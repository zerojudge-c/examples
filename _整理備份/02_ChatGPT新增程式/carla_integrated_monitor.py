#!/usr/bin/env python3
"""CARLA 0.9.16 unified cooperative perception DEMO (observation only).

Existing Ego + raw LiDAR + geometric blind zone BEV + GT comparison + collision
logging + optional 5G/MEC alert UDP. Does not control, teleport, destroy vehicles,
call world.tick(), or change world/TM settings. Creates/destroys only own sensors.

Needs blind_zone_lidar_v2.py in the SAME folder.
  py -3.10 carla_integrated_monitor.py --ego-id 200
  py -3.10 carla_integrated_monitor.py --ego-id 200 --headless
  py -3.10 carla_integrated_monitor.py --ego-id 200 --alert-udp-host 127.0.0.1 --alert-udp-port 50101

The GT actor states used for evaluation/blind-zone geometry must NOT be called
LiDAR-only perception. UDP alerts are a NEW optional JSON stream; existing
Unity vehicle bridge on port 50100 is independent and its schema is unchanged.
"""
import argparse
import csv
import json
import math
import socket
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import carla
import cv2
import numpy as np
import blind_zone_lidar_v2 as blind


def gt_front(world, ego, radius):
    p = ego.get_transform()
    forward = p.get_forward_vector()
    origin = p.location
    ev = ego.get_velocity()
    own_speed = ev.x * forward.x + ev.y * forward.y
    best = None
    for car in world.get_actors().filter('vehicle.*'):
        if car.id == ego.id or not car.is_alive:
            continue
        try:
            q = car.get_location()
            dx, dy = q.x - origin.x, q.y - origin.y
            a = dx * forward.x + dy * forward.y
            b = abs(-dx * forward.y + dy * forward.x)
            if a <= 0 or a > radius or b > 2.4:
                continue
            clearance = max(0., a - ego.bounding_box.extent.x - car.bounding_box.extent.x)
            ov = car.get_velocity()
            closing = own_speed - (ov.x * forward.x + ov.y * forward.y)
            ttc = clearance / closing if closing > 0.5 else None
            if best is None or clearance < best['gap']:
                best = dict(id=car.id, gap=clearance, closing=closing, ttc=ttc)
        except RuntimeError:
            continue
    return best


def lidar_corridor(points, radius):
    """Unclassified obstacle-range heuristic; never uses actor positions."""
    if points.size == 0:
        return None, 0
    # LiDAR is mounted 2.4m high. Keep vertical surfaces and reject ground.
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    mask = (x > 3.8) & (x < radius) & (np.abs(y) < 1.6) & (z > -1.9) & (z < 0.7)
    hits = x[mask]
    if hits.size < 12:
        return None, int(hits.size)
    # Minimum occupied 1-m longitudinal bin with at least 8 points.
    bins = np.floor(hits).astype(np.int32)
    values, counts = np.unique(bins, return_counts=True)
    qualifying = values[counts >= 8]
    if not len(qualifying):
        return None, int(hits.size)
    first = int(qualifying[0])
    group = hits[(hits >= first) & (hits < first + 1.0)]
    return float(np.median(group)), int(hits.size)


def estimate_ttc(history, distance, half_length):
    if distance is None or len(history) < 3:
        return None, None
    start, end = history[0], history[-1]
    dt = end[0] - start[0]
    if dt < 0.15:
        return None, None
    closing = (start[1] - end[1]) / dt
    if closing <= 0.7:
        return closing, None
    return closing, max(0.0, distance - half_length) / closing


def emit_status_console(s, impact):
    def fmt(v, decimals=2):
        return '--' if v is None else ('%%.%df' % decimals) % v
    print('t=%6.1fs | LiDAR=%s m TTC=%s %s | GT car=%s TTC=%s %s | blind=%d%s' % (
        s['wall_s'], fmt(s['lidar_m'], 1), fmt(s['lidar_ttc']),
        'WARNING' if s['lidar_warning'] else '',
        s['gt_id'] if s['gt_id'] is not None else '--', fmt(s['gt_ttc']),
        'WARNING' if s['gt_warning'] else '', s['blind_count'],
        (' | COLLISION actor=%s' % impact) if impact else ''), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ego-id', type=int, required=True)
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=2000)
    ap.add_argument('--radius', type=float, default=50)
    ap.add_argument('--warn-ttc', type=float, default=4)
    ap.add_argument('--headless', action='store_true', help='No OpenCV window, write CSV only')
    ap.add_argument('--alert-udp-host', default=None, help='Optional separate alert listener IP; not vehicle stream')
    ap.add_argument('--alert-udp-port', type=int, default=50101)
    ap.add_argument('--sample-hz', type=float, default=10)
    args = ap.parse_args()
    if args.sample_hz <= 0:
        ap.error('--sample-hz must be positive')
    client = carla.Client(args.host, args.port)
    client.set_timeout(10)
    world = client.get_world()
    ego = world.get_actor(args.ego_id)
    if ego is None or not ego.type_id.startswith('vehicle.'):
        print('ERROR: Ego missing. List actors again; IDs can change.')
        return 2

    lock = threading.Lock()
    latest = {'frame': -1, 'stamp': None, 'points': np.empty((0, 4), dtype=np.float32)}
    impacts = deque()
    sensors = []
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if args.alert_udp_host else None
    csvfile = Path(__file__).with_name('integrated_carla_%s.csv' % datetime.now().strftime('%Y%m%d_%H%M%S'))

    def lidar_callback(data):
        raw = np.frombuffer(data.raw_data, dtype=np.float32)
        if raw.size == 0 or raw.size % 4:
            return
        arr = raw.reshape((-1, 4)).copy()
        with lock:
            latest.update(frame=int(data.frame), stamp=float(data.timestamp), points=arr)

    def collision_callback(event):
        with lock:
            impacts.append((int(event.frame), int(event.other_actor.id), float(event.timestamp)))

    history = deque(maxlen=5)
    prev_range = None
    prev_frame = -1
    prev_flags = (False, False, False)
    recent_warnings = {'lidar': deque(maxlen=500), 'gt': deque(maxlen=500)}
    print('WORLD:', world.get_map().name, 'EGO:', ego.id, ego.type_id)
    print('READ ONLY: no moving cars, no world.tick(), no world setting changes.')
    print('LiDAR = unclassified point-cloud estimate; GT/blind zone = actor ground truth.')
    print('Press Q / ESC in the image or Ctrl+C here to save CSV and exit.')
    try:
        bp = world.get_blueprint_library().find('sensor.lidar.ray_cast')
        config = {'channels': '32', 'range': str(args.radius),
                  'points_per_second': '100000', 'rotation_frequency': '20',
                  'upper_fov': '10', 'lower_fov': '-30', 'sensor_tick': '0.1'}
        for key, value in config.items():
            bp.set_attribute(key, value)
        lidar = world.spawn_actor(bp, carla.Transform(carla.Location(z=2.4)), attach_to=ego)
        sensors.append(lidar)
        collision = world.spawn_actor(world.get_blueprint_library().find('sensor.other.collision'),
                                      carla.Transform(), attach_to=ego)
        sensors.append(collision)
        lidar.listen(lidar_callback)
        collision.listen(collision_callback)
        print('MONITOR READY. CSV:', csvfile, flush=True)
        with csvfile.open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['wall_elapsed_s', 'sim_time_s', 'lidar_frame', 'ego_kmh',
                             'lidar_corridor_distance_m', 'lidar_points', 'lidar_closing_mps_est',
                             'lidar_ttc_s_est', 'lidar_warning', 'gt_vehicle_id',
                             'gt_clearance_m', 'gt_closing_mps', 'gt_ttc_s',
                             'gt_warning', 'blind_vehicle_ids', 'collision_actor_id',
                             'collision_frame', 'collision_sim_time_s'])
            start = time.monotonic()
            last_print = -1
            while ego.is_alive:
                cycle_start = time.monotonic()
                now = cycle_start - start
                with lock:
                    sample = latest.copy()
                    event = impacts.popleft() if impacts else None
                lidar_distance, count = lidar_corridor(sample['points'], args.radius)
                if sample['frame'] != prev_frame:
                    prev_frame = sample['frame']
                    if lidar_distance is None:
                        history.clear()
                    else:
                        # Avoid mixing different obstacles when the corridor suddenly jumps.
                        if prev_range is not None and abs(lidar_distance-prev_range) > 3.0:
                            history.clear()
                        history.append((sample['stamp'], lidar_distance))
                    prev_range = lidar_distance
                lidar_closing, lidar_ttc = estimate_ttc(history, lidar_distance, ego.bounding_box.extent.x)
                lidar_warn = lidar_ttc is not None and lidar_ttc <= args.warn_ttc
                truth = gt_front(world, ego, args.radius)
                gt_ttc = truth['ttc'] if truth else None
                gt_warn = gt_ttc is not None and gt_ttc <= args.warn_ttc
                projections = []
                ego_transform = ego.get_transform()
                for car in world.get_actors().filter('vehicle.*'):
                    if car.id == ego.id or not car.is_alive:
                        continue
                    projection = blind.get_vehicle_projection(car, ego_transform)
                    if projection is not None and projection['distance'] <= args.radius:
                        projections.append(projection)
                blind_results = blind.detect_blind_vehicles(projections)
                blind_ids = sorted({x['target']['id'] for x in blind_results})
                e_v = ego.get_velocity()
                ego_kmh = 3.6 * math.sqrt(e_v.x**2 + e_v.y**2 + e_v.z**2)
                sim_stamp = sample['stamp']
                state = dict(wall_s=round(now, 3), sim_time_s=sim_stamp,
                             lidar_m=lidar_distance, lidar_ttc=lidar_ttc,
                             lidar_warning=bool(lidar_warn), gt_id=truth['id'] if truth else None,
                             gt_ttc=gt_ttc, gt_warning=bool(gt_warn),
                             blind_count=len(blind_ids), blind_ids=blind_ids)
                if lidar_warn:
                    recent_warnings['lidar'].append(now)
                if gt_warn:
                    recent_warnings['gt'].append(now)
                if event is not None:
                    print('>>> COLLISION: other_actor=%d frame=%d sensor_sim_time=%.3f' %
                          (event[1], event[0], event[2]), flush=True)
                    for name, ts in recent_warnings.items():
                        recent = [t for t in ts if 0 <= now-t <= 10.]
                        print('>>> %s 10-second lead: %s' % (name.upper(),
                              ('%.3fs' % (now-min(recent))) if recent else 'NONE'), flush=True)
                    recent_warnings = {'lidar': deque(maxlen=500), 'gt': deque(maxlen=500)}
                writer.writerow([round(now, 3), sim_stamp, sample['frame'], round(ego_kmh, 2),
                                 '' if lidar_distance is None else round(lidar_distance, 3), count,
                                 '' if lidar_closing is None else round(lidar_closing, 3),
                                 '' if lidar_ttc is None else round(lidar_ttc, 3), int(lidar_warn),
                                 '' if truth is None else truth['id'],
                                 '' if truth is None else round(truth['gap'], 3),
                                 '' if truth is None else round(truth['closing'], 3),
                                 '' if gt_ttc is None else round(gt_ttc, 3), int(gt_warn),
                                 ';'.join(map(str, blind_ids)), '' if event is None else event[1],
                                 '' if event is None else event[0], '' if event is None else event[2]])
                flags = (bool(lidar_warn), bool(gt_warn), bool(blind_ids))
                if (int(now) != last_print or event or flags != prev_flags):
                    emit_status_console(state, event[1] if event else None)
                    last_print = int(now)
                # Alert packets are OUT-OF-BAND; do not change vehicle positioning schema.
                if sender and (event or flags != prev_flags):
                    payload = json.dumps({
                        'type': 'carla_warning', 'version': 1, 'ego_actor_id': ego.id,
                        'sim_time_s': sim_stamp, 'lidar_warning': bool(lidar_warn),
                        'gt_warning': bool(gt_warn), 'blind_vehicle_ids': blind_ids,
                        'lidar_ttc_s': lidar_ttc, 'gt_ttc_s': gt_ttc,
                        'collision_actor_id': event[1] if event else None,
                        'perception_note': 'lidar=unclassified; blind_and_gt=actor_truth'
                    }, ensure_ascii=True, separators=(',', ':')).encode('utf-8')
                    if len(payload) < 1100:
                        sender.sendto(payload, (args.alert_udp_host, args.alert_udp_port))
                prev_flags = flags
                if not args.headless:
                    image = blind.render_bev(sample['points'], projections, blind_results)
                    bar_color = (10, 30, 210) if lidar_warn else ((10, 175, 230) if gt_warn else (35, 100, 35))
                    cv2.rectangle(image, (0, 95), (blind.IMAGE_SIZE, 180), (15, 15, 15), -1)
                    cv2.putText(image, 'EGO #%d | speed %.1f km/h' % (ego.id, ego_kmh),
                                (15, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (240, 240, 240), 1)
                    cv2.putText(image, 'LiDAR %.1fm / TTC %s | %s' % (
                        lidar_distance if lidar_distance is not None else -1.,
                        '--' if lidar_ttc is None else '%.2fs' % lidar_ttc,
                        'WARNING' if lidar_warn else 'no warning'),
                                (15, 138), cv2.FONT_HERSHEY_SIMPLEX, 0.51, bar_color, 2)
                    cv2.putText(image, 'GT vehicle %s | TTC %s | blind=%d' % (
                        truth['id'] if truth else '--',
                        '--' if gt_ttc is None else '%.2fs' % gt_ttc, len(blind_ids)),
                                (15, 163), cv2.FONT_HERSHEY_SIMPLEX, 0.51,
                                (50, 210, 255) if gt_warn else (225, 225, 225), 1)
                    cv2.imshow('CARLA Integrated: LiDAR + Blind + TTC + Collision', image)
                    key = cv2.waitKey(1) & 0xff
                    if key in (ord('q'), 27):
                        break
                if event or int(now) != int(max(0., now-1./args.sample_hz)):
                    f.flush()
                time.sleep(max(0., 1./args.sample_hz - (time.monotonic()-cycle_start)))
    except KeyboardInterrupt:
        print('\nStopped by user.', flush=True)
    finally:
        for sensor in sensors:
            try:
                sensor.stop()
                sensor.destroy()
            except RuntimeError:
                pass
        if sender:
            sender.close()
        if not args.headless:
            cv2.destroyAllWindows()
        print('CSV SAVED:', csvfile, '\nExisting CARLA actors remain untouched.', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
