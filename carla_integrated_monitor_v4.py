#!/usr/bin/env python3
"""CARLA 0.9.16 V4 LiDAR/GT spatial matching (observation only).

Existing Ego + raw LiDAR + geometric blind zone BEV + GT comparison + collision
logging + optional 5G/MEC alert UDP. Does not control, teleport, destroy vehicles,
call world.tick(), or change world/TM settings. Creates/destroys only own sensors.

Needs blind_zone_lidar_v2.py in the SAME folder.
  py -3.10 carla_integrated_monitor_v4.py --ego-id 200
  py -3.10 carla_integrated_monitor_v4.py --ego-id 200 --headless
  py -3.10 carla_integrated_monitor_v4.py --ego-id 200 --alert-udp-host 127.0.0.1 --alert-udp-port 50101

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
    if hits.size < 6:
        return None, int(hits.size)
    # Minimum occupied 1-m longitudinal bin with at least 8 points.
    bins = np.floor(hits).astype(np.int32)
    values, counts = np.unique(bins, return_counts=True)
    qualifying = values[counts >= 4]
    if not len(qualifying):
        return None, int(hits.size)
    first = int(qualifying[0])
    group = hits[(hits >= first) & (hits < first + 1.0)]
    return float(np.median(group)), int(hits.size)



def match_lidar_to_gt_vehicle(points, lidar_transform, target, radius, min_hits=4,
                              padding_m=0.35):
    """GT-assisted spatial verification; NEVER a LiDAR-only object ID.

    LiDAR returns are transformed through the sensor/world/actor/box coordinate
    systems.  A hit is associated only if its 3D location intersects the target
    oriented bounding box (with modest tolerance).  We compare each matched
    ray's first GT box intersection with its actual return along the same ray,
    so the distance reference is sensor-to-surface, not vehicle clearance.
    This is an approximate, time-unsynchronized association for evaluation.
    """
    result = {'status': 'UNKNOWN', 'target_id': None, 'hit_count': 0,
              'lidar_m': None, 'gt_m': None, 'error_m': None,
              'matched_points': np.empty((0, 3), dtype=np.float32)}
    if target is None or points.size == 0:
        return result
    result['target_id'] = target.id
    try:
        bbox = target.bounding_box
        # sensor -> world -> actor -> oriented bounding-box frame
        s_to_box = (np.asarray(carla.Transform(bbox.location, bbox.rotation).get_inverse_matrix(), dtype=np.float64)
                    @ np.asarray(target.get_transform().get_inverse_matrix(), dtype=np.float64)
                    @ np.asarray(lidar_transform.get_matrix(), dtype=np.float64))
        coords = np.asarray(points[:, :3], dtype=np.float64)
        corridor = ((coords[:, 0] > 3.8) & (coords[:, 0] < radius) &
                    (np.abs(coords[:, 1]) < 1.6) &
                    (coords[:, 2] > -1.9) & (coords[:, 2] < 0.7))
        if not np.any(corridor):
            return result
        pts = coords[corridor]
        xyz_box = pts @ s_to_box[:3, :3].T + s_to_box[:3, 3]
        ext = np.array([bbox.extent.x, bbox.extent.y, bbox.extent.z], dtype=np.float64)
        inside = np.all(np.abs(xyz_box) <= ext + padding_m, axis=1)
        pts = pts[inside]
        xyz_box = xyz_box[inside]
        result['hit_count'] = len(pts)
        result['matched_points'] = pts.astype(np.float32)
        if len(pts) < min_hits:
            return result

        # Ray/box slab intersection using exact, unpadded CARLA OBB.
        origin = s_to_box[:3, 3]
        ray = xyz_box - origin
        bounds_lo, bounds_hi = -ext, ext
        low = np.empty_like(ray)
        high = np.empty_like(ray)
        low.fill(-np.inf)
        high.fill(np.inf)
        valid = np.ones(len(ray), dtype=bool)
        for axis in range(3):
            component = ray[:, axis]
            moving = np.abs(component) > 1e-9
            valid &= moving | ((origin[axis] >= bounds_lo[axis]) & (origin[axis] <= bounds_hi[axis]))
            low[moving, axis] = (bounds_lo[axis] - origin[axis]) / component[moving]
            high[moving, axis] = (bounds_hi[axis] - origin[axis]) / component[moving]
        t_enter = np.maximum(0., np.max(np.minimum(low, high), axis=1))
        t_exit = np.min(np.maximum(low, high), axis=1)
        valid &= (t_exit >= t_enter) & (t_enter >= 0) & (t_enter <= 1.5)
        if np.count_nonzero(valid) < min_hits:
            return result
        lidar_x = pts[valid, 0]
        gt_x = lidar_x * t_enter[valid]
        if np.median(np.abs(lidar_x - np.median(lidar_x))) > 1.5:
            return result
        result.update(status='YES (GT-assisted)',
                      hit_count=int(np.count_nonzero(valid)),
                      lidar_m=float(np.median(lidar_x)),
                      gt_m=float(np.median(gt_x)),
                      error_m=float(np.median(np.abs(lidar_x - gt_x))),
                      matched_points=pts[valid].astype(np.float32))
    except (RuntimeError, ValueError, TypeError, AttributeError) as exc:
        # Keep the observation-only monitor alive if an NPC disappears mid-frame.
        result['status'] = 'UNKNOWN'
    return result


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
    ap.add_argument('--event-cooldown', type=float, default=30., help='Minimum seconds between events for the same actor')
    ap.add_argument('--warning-window', type=float, default=10., help='Look-back window before impact in simulation seconds')
    args = ap.parse_args()
    if args.sample_hz <= 0:
        ap.error('--sample-hz must be positive')
    if args.event_cooldown < 0 or args.warning_window <= 0:
        ap.error('invalid event-cooldown or warning-window')
    client = carla.Client(args.host, args.port)
    client.set_timeout(10)
    world = client.get_world()
    ego = world.get_actor(args.ego_id)
    if ego is None or not ego.type_id.startswith('vehicle.'):
        print('ERROR: Ego missing. List actors again; IDs can change.')
        return 2

    lock = threading.Lock()
    latest = {'frame': -1, 'stamp': None, 'points': np.empty((0, 4), dtype=np.float32)}
    impacts = deque(maxlen=2000)
    raw_impacts = [0]
    enqueued_at = {}
    episodes = {}  # actor -> last count and contact state
    crash_count = 0
    last_crash = None
    last_lidar_lead = None
    last_gt_lead = None
    sensors = []
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if args.alert_udp_host else None
    tag = datetime.now().strftime('%Y%m%d_%H%M%S')
    csvfile = Path(__file__).with_name('integrated_carla_v4_%s.csv' % tag)
    eventfile = Path(__file__).with_name('accident_events_v4_%s.csv' % tag)

    def lidar_callback(data):
        raw = np.frombuffer(data.raw_data, dtype=np.float32)
        if raw.size == 0 or raw.size % 4:
            return
        arr = raw.reshape((-1, 4)).copy()
        with lock:
            latest.update(frame=int(data.frame), stamp=float(data.timestamp), points=arr)

    def collision_callback(event):
        actor_id = int(event.other_actor.id)
        stamp = float(event.timestamp)
        with lock:
            raw_impacts[0] += 1
            # Cap callback traffic while retaining the FIRST contact notification.
            if stamp - enqueued_at.get(actor_id, -1e10) < 0.2:
                return
            enqueued_at[actor_id] = stamp
            impacts.append((int(event.frame), actor_id, stamp))

    history = deque(maxlen=5)
    prev_range = None
    prev_frame = -1
    prev_flags = (False, False, False)
    recent_warnings = {'lidar': deque(maxlen=2000), 'gt': deque(maxlen=2000)}
    print('WORLD:', world.get_map().name, 'EGO:', ego.id, ego.type_id)
    print('READ ONLY: no moving cars, no world.tick(), no world setting changes.')
    print('LiDAR = unclassified point-cloud estimate; GT/blind zone = actor ground truth.')
    print('V4 ACCIDENT COUNTER: first distinct contact per actor; waits for separation and cooldown.')
    print('Counts start at 0 each run. LiDAR warning is NOT actor-identified.')
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
        print('EVENT CSV:', eventfile, flush=True)
        with csvfile.open('w', newline='', encoding='utf-8-sig') as f, \
             eventfile.open('w', newline='', encoding='utf-8-sig') as ef:
            writer = csv.writer(f)
            eventwriter = csv.writer(ef)
            eventwriter.writerow(['event_number', 'ego_id', 'target_id', 'collision_frame',
                                  'collision_sim_time_s', 'lidar_lead_s_unclassified',
                                  'gt_lead_s_target_matched', 'raw_collision_callback_total'])
            writer.writerow(['wall_elapsed_s', 'sim_time_s', 'lidar_frame', 'ego_kmh',
                             'lidar_corridor_distance_m', 'lidar_points', 'lidar_closing_mps_est',
                             'lidar_ttc_s_est', 'lidar_warning', 'gt_vehicle_id',
                             'gt_clearance_m', 'gt_closing_mps', 'gt_ttc_s',
                             'gt_warning', 'blind_vehicle_ids', 'collision_actor_id',
                             'collision_frame', 'collision_sim_time_s',
                             'unique_crash_count', 'raw_callback_count',
                             'match_status', 'match_gt_actor_id', 'match_hits',
                             'matched_lidar_range_m', 'matched_gt_reference_m', 'matched_error_m'])
            start = time.monotonic()
            last_print = -1
            while ego.is_alive:
                cycle_start = time.monotonic()
                now = cycle_start - start
                with lock:
                    sample = latest.copy()
                    pending = list(impacts)
                    impacts.clear()
                    raw_total = raw_impacts[0]
                event = None
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
                matched = {'status': 'UNKNOWN', 'target_id': None, 'hit_count': 0,
                           'lidar_m': None, 'gt_m': None, 'error_m': None,
                           'matched_points': np.empty((0, 3), dtype=np.float32)}
                if truth is not None and sample['stamp'] is not None:
                    selected = world.get_actor(truth['id'])
                    if selected is not None and selected.is_alive:
                        matched = match_lidar_to_gt_vehicle(
                            sample['points'], lidar.get_transform(), selected, args.radius)
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
                if sim_stamp is not None:
                    if lidar_warn:
                        recent_warnings['lidar'].append(sim_stamp)
                    if gt_warn and truth is not None:
                        recent_warnings['gt'].append((sim_stamp, truth['id']))
                    while recent_warnings['lidar'] and sim_stamp - recent_warnings['lidar'][0] > args.warning_window + 1:
                        recent_warnings['lidar'].popleft()
                    while recent_warnings['gt'] and sim_stamp - recent_warnings['gt'][0][0] > args.warning_window + 1:
                        recent_warnings['gt'].popleft()
                # Reset episode only after physical separation, never while cars remain touching.
                for actor_id, episode in list(episodes.items()):
                    if not episode['active']:
                        continue
                    other = world.get_actor(actor_id)
                    separated = False
                    try:
                        if other is None or not other.is_alive:
                            separated = True
                        else:
                            ego_pos = ego.get_location()
                            other_pos = other.get_location()
                            distance_xy = math.hypot(ego_pos.x - other_pos.x,
                                                     ego_pos.y - other_pos.y)
                            extent = (math.hypot(ego.bounding_box.extent.x, ego.bounding_box.extent.y)
                                      + math.hypot(other.bounding_box.extent.x, other.bounding_box.extent.y))
                            separated = distance_xy > extent + 2.0
                    except RuntimeError:
                        separated = True
                    if separated:
                        if episode['separated_since'] is None:
                            episode['separated_since'] = sim_stamp
                        elif sim_stamp is not None and episode['separated_since'] is not None and sim_stamp - episode['separated_since'] >= 2.0:
                            episode['active'] = False
                    else:
                        episode['separated_since'] = None

                # Process limited sensor notifications; only new episodes increment the counter.
                for candidate in pending:
                    frame, actor_id, hit_stamp = candidate
                    episode = episodes.get(actor_id)
                    if episode is not None and episode['active']:
                        continue
                    if episode is not None and hit_stamp - episode['last_counted'] < args.event_cooldown:
                        continue
                    episodes[actor_id] = {'active': True, 'last_counted': hit_stamp,
                                          'separated_since': None}
                    crash_count += 1
                    event = candidate
                    last_crash = actor_id
                    lidar_times = [t for t in recent_warnings['lidar']
                                   if 0 <= hit_stamp - t <= args.warning_window]
                    gt_times = [t for t, target in recent_warnings['gt']
                                if target == actor_id and 0 <= hit_stamp - t <= args.warning_window]
                    last_lidar_lead = round(hit_stamp - min(lidar_times), 3) if lidar_times else None
                    last_gt_lead = round(hit_stamp - min(gt_times), 3) if gt_times else None
                    eventwriter.writerow([crash_count, ego.id, actor_id, frame, round(hit_stamp, 3),
                                          '' if last_lidar_lead is None else last_lidar_lead,
                                          '' if last_gt_lead is None else last_gt_lead, raw_total])
                    ef.flush()
                    print('>>> NEW COLLISION #%d: actor=%d frame=%d | LiDAR lead=%s | GT lead=%s' % (
                        crash_count, actor_id, frame,
                        '--' if last_lidar_lead is None else '%.3fs (unclassified)' % last_lidar_lead,
                        '--' if last_gt_lead is None else '%.3fs (ID matched)' % last_gt_lead), flush=True)
                writer.writerow([round(now, 3), sim_stamp, sample['frame'], round(ego_kmh, 2),
                                 '' if lidar_distance is None else round(lidar_distance, 3), count,
                                 '' if lidar_closing is None else round(lidar_closing, 3),
                                 '' if lidar_ttc is None else round(lidar_ttc, 3), int(lidar_warn),
                                 '' if truth is None else truth['id'],
                                 '' if truth is None else round(truth['gap'], 3),
                                 '' if truth is None else round(truth['closing'], 3),
                                 '' if gt_ttc is None else round(gt_ttc, 3), int(gt_warn),
                                 ';'.join(map(str, blind_ids)), '' if event is None else event[1],
                                 '' if event is None else event[0], '' if event is None else event[2],
                                  crash_count, raw_total,
                                  matched['status'],
                                  '' if matched['target_id'] is None else matched['target_id'],
                                  matched['hit_count'],
                                  '' if matched['lidar_m'] is None else round(matched['lidar_m'], 3),
                                  '' if matched['gt_m'] is None else round(matched['gt_m'], 3),
                                  '' if matched['error_m'] is None else round(matched['error_m'], 3)])
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
                    # LiDAR diagnostic overlay: cyan = real filtered LiDAR returns.
                    # This NEVER uses CARLA actor positions to color LiDAR points.
                    cloud = sample['points']
                    if cloud.size:
                        cx, cy, cz = cloud[:, 0], cloud[:, 1], cloud[:, 2]
                        corridor_mask = ((cx > 3.8) & (cx < args.radius) &
                                         (np.abs(cy) < 1.6) & (cz > -1.9) & (cz < 0.7))
                        corridor_cloud = cloud[corridor_mask]
                        for pt in corridor_cloud:
                            px, py = blind.ego_to_pixel(float(pt[0]), float(pt[1]))
                            if 0 <= px < image.shape[1] and 0 <= py < image.shape[0]:
                                cv2.circle(image, (px, py), 2, (255, 255, 0), -1,
                                           lineType=cv2.LINE_AA)
                    # GT-assisted matched returns are magenta. Cyan still means
                    # unclassified LiDAR corridor returns.
                    for pt in matched['matched_points']:
                        px, py = blind.ego_to_pixel(float(pt[0]), float(pt[1]))
                        if 0 <= px < image.shape[1] and 0 <= py < image.shape[0]:
                            cv2.circle(image, (px, py), 3, (255, 0, 255), -1,
                                       lineType=cv2.LINE_AA)
                    # This ring marks the estimated range, not an identified vehicle.
                    if lidar_distance is not None:
                        px, py = blind.ego_to_pixel(float(lidar_distance), 0.0)
                        if 0 <= px < image.shape[1] and 0 <= py < image.shape[0]:
                            cv2.circle(image, (px, py), 9, (0, 255, 0), 2,
                                       lineType=cv2.LINE_AA)
                    # Compact diagnostic label in the upper-right, away from NPC icons.
                    diag = 'LiDAR front hits: %d' % count
                    dw = cv2.getTextSize(diag, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)[0][0]
                    dx = max(8, image.shape[1] - dw - 20)
                    cv2.rectangle(image, (dx-8, 8), (image.shape[1]-6, 37), (12, 18, 20), -1)
                    cv2.putText(image, diag, (dx, 28), cv2.FONT_HERSHEY_SIMPLEX,
                                0.50, (255, 255, 0), 1, cv2.LINE_AA)
                    # Upper-right GT-assisted point association panel.
                    text_match = [
                        'LiDAR / GT MATCH (evaluation)',
                        'GT target: %s' % ('--' if matched['target_id'] is None else 'NPC %d' % matched['target_id']),
                        'Matched hits: %d | Match: %s' % (matched['hit_count'], matched['status']),
                        'LiDAR range: %s' % ('--' if matched['lidar_m'] is None else '%.2f m' % matched['lidar_m']),
                        'GT reference: %s' % ('--' if matched['gt_m'] is None else '%.2f m' % matched['gt_m']),
                        'Abs range error: %s' % ('--' if matched['error_m'] is None else '%.2f m' % matched['error_m']),
                    ]
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    scale = 0.43
                    max_w = max(cv2.getTextSize(item, font, scale, 1)[0][0] for item in text_match)
                    match_right = image.shape[1] - 8
                    match_left = max(8, match_right - max_w - 22)
                    match_top = 46
                    match_bottom = match_top + 6 + 22 * len(text_match)
                    cv2.rectangle(image, (match_left, match_top),
                                  (match_right, match_bottom), (14, 17, 20), -1)
                    for idx, line in enumerate(text_match):
                        color = (130, 245, 140) if matched['status'].startswith('YES') and idx == 2 else (225, 225, 225)
                        cv2.putText(image, line, (match_left + 8, match_top + 19 + idx * 22),
                                    font, scale, color, 1, cv2.LINE_AA)
                    bar_color = (10, 30, 210) if lidar_warn else ((10, 175, 230) if gt_warn else (35, 100, 35))
                    # Compact panel: fit the text, never cover the full BEV width.
                    panel_lines = [
                        'EGO #%d | speed %.1f km/h' % (ego.id, ego_kmh),
                        'LiDAR %.1fm / TTC %s | %s' % (
                            lidar_distance if lidar_distance is not None else -1.,
                            '--' if lidar_ttc is None else '%.2fs' % lidar_ttc,
                            'WARNING' if lidar_warn else 'no warning'),
                        'GT vehicle %s | TTC %s | blind=%d' % (
                            truth['id'] if truth else '--',
                            '--' if gt_ttc is None else '%.2fs' % gt_ttc, len(blind_ids)),
                    ]
                    text_width = max(cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.53, 2)[0][0]
                                     for line in panel_lines)
                    x1, y1 = 8, 95
                    x2 = min(image.shape[1] - 1, x1 + text_width + 24)
                    y2 = min(image.shape[0] - 1, 180)
                    # Blend only inside this little rectangle; remainder stays untouched.
                    region = image[y1:y2, x1:x2]
                    if region.size:
                        backdrop = np.full_like(region, (15, 15, 15))
                        cv2.addWeighted(backdrop, 0.77, region, 0.23, 0, dst=region)
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
                    # Place stats in lower-right area without replacing original TTC overlay.
                    stats = [
                        'ACCIDENT MONITOR V3',
                        'Confirmed episodes: %d' % crash_count,
                        'Last NPC: %s' % ('--' if last_crash is None else last_crash),
                        'LiDAR lead*: %s' % ('--' if last_lidar_lead is None else '%.2fs' % last_lidar_lead),
                        'GT lead (ID): %s' % ('--' if last_gt_lead is None else '%.2fs' % last_gt_lead),
                        '*LiDAR obstacle unclassified'
                    ]
                    ih, iw = image.shape[:2]
                    panel_w, panel_h = min(340, iw), 149
                    x0 = max(0, iw - panel_w)
                    y0 = max(185, ih - panel_h)
                    cv2.rectangle(image, (x0, y0), (iw-1, min(ih-1, y0+panel_h)), (15, 15, 15), -1)
                    for ix, line in enumerate(stats):
                        cv2.putText(image, line, (x0+9, y0+21+ix*23),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.47, (245, 245, 245), 1)
                    cv2.imshow('CARLA Integrated V4: LiDAR / GT MATCH + Accidents', image)
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
        print('CSV SAVED:', csvfile, '\nEVENTS SAVED:', eventfile, '\nExisting CARLA actors remain untouched.', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
