#!/usr/bin/env python3
"""Attach read-only LiDAR/TTC monitoring to an existing CARLA 0.9.16 Ego.

Does not move/control any vehicle, advance the CARLA world, or alter settings.
LiDAR is an unclassified forward obstacle heuristic (not object detection).
CARLA actor-based TTC is a separate ground-truth comparison signal.

Usage:
  py -3.10 carla_lidar_ttc_monitor.py --ego-id 200
  py -3.10 carla_lidar_ttc_monitor.py --ego-id 200 --warn-ttc 4
Start this BEFORE triggering the existing collision scenario in another CMD.
"""
import argparse
import csv
import math
import statistics
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import carla
import numpy as np


def velocity_xy(actor):
    v = actor.get_velocity()
    return v.x, v.y


def gt_front_vehicle(world, ego, radius):
    """Actor state reference, NOT information obtained through LiDAR."""
    p = ego.get_location()
    yaw = math.radians(ego.get_transform().rotation.yaw)
    fx, fy = math.cos(yaw), math.sin(yaw)
    evx, evy = velocity_xy(ego)
    v_ego_forward = evx * fx + evy * fy
    best = None
    for other in world.get_actors().filter('vehicle.*'):
        if other.id == ego.id or not other.is_alive:
            continue
        q = other.get_location()
        dx, dy = q.x-p.x, q.y-p.y
        forward = dx*fx + dy*fy
        lateral = abs(-dx*fy + dy*fx)
        if not (0 < forward < radius and lateral < 2.0):
            continue
        try:
            clearance = max(0.0, forward - ego.bounding_box.extent.x - other.bounding_box.extent.x)
            ovx, ovy = velocity_xy(other)
            closing = v_ego_forward - (ovx*fx + ovy*fy)
            ttc = clearance / closing if closing > 0.5 else math.inf
            if best is None or clearance < best[1]:
                best = (other.id, clearance, closing, ttc)
        except RuntimeError:
            continue
    return best


def main():
    ap = argparse.ArgumentParser(description='Read-only LiDAR + actor-reference TTC monitor for existing CARLA Ego')
    ap.add_argument('--ego-id', type=int, required=True)
    ap.add_argument('--warn-ttc', type=float, default=4.0)
    ap.add_argument('--radius', type=float, default=50.0)
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=2000)
    args = ap.parse_args()
    client = carla.Client(args.host, args.port)
    client.set_timeout(10)
    world = client.get_world()
    ego = world.get_actor(args.ego_id)
    if ego is None or not ego.type_id.startswith('vehicle.'):
        print('ERROR: Ego ID not present. Check with CARLA_All_Vehicles_To_5G.py --list')
        return 2

    bp = world.get_blueprint_library().find('sensor.lidar.ray_cast')
    for name, value in {'channels':'32','range':str(args.radius),'points_per_second':'100000',
                        'rotation_frequency':'10','upper_fov':'10','lower_fov':'-30'}.items():
        bp.set_attribute(name, value)
    sensor = None
    collision_sensor = None
    lock = threading.Lock()
    latest = {'frame':-1, 'stamp':None, 'range':None, 'points':0}
    collisions = deque()
    log_file = Path(__file__).with_name('lidar_ttc_log_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.csv')

    def on_lidar(measurement):
        # LiDAR local coordinate frame: x=forward, y=lateral, z=up.
        arr = np.frombuffer(measurement.raw_data, dtype=np.float32)
        if arr.size == 0 or arr.size % 4:
            return
        pts = arr.reshape((-1, 4))
        # Heuristic: narrow central corridor; exclude road surface and overhead objects.
        # This may still include poles, guardrails and unrelated obstacles.
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        selected = x[(x > 2.5) & (x < args.radius) & (np.abs(y) < 1.45) & (z > -1.8) & (z < 0.4)]
        # 10th percentile avoids a lone random close point. Require enough returns.
        dist = float(np.percentile(selected, 10)) if selected.size >= 8 else None
        with lock:
            latest.update(frame=measurement.frame, stamp=float(measurement.timestamp),
                          range=dist, points=int(selected.size))

    def on_collision(event):
        with lock:
            collisions.append((event.frame, event.other_actor.id, time.monotonic()))

    history = deque(maxlen=5)
    last_seen_frame = -1
    print('Connected:', world.get_map().name, 'Ego:', ego.id, ego.type_id, flush=True)
    print('Does not move cars / world.tick / alter synchronous settings.', flush=True)
    print('LiDAR distance is an UNCLASSIFIED corridor estimate; actor TTC is labeled ground truth.', flush=True)
    print('Run your collision script in ANOTHER CMD after this displays MONITOR READY.', flush=True)
    try:
        sensor = world.spawn_actor(bp, carla.Transform(carla.Location(x=0.0, y=0.0, z=2.4)), attach_to=ego)
        collision_sensor = world.spawn_actor(world.get_blueprint_library().find('sensor.other.collision'), carla.Transform(), attach_to=ego)
        sensor.listen(on_lidar)
        collision_sensor.listen(on_collision)
        print('MONITOR READY. Press Ctrl+C to stop monitoring. CSV:', log_file, flush=True)
        with log_file.open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.writer(stream)
            writer.writerow(['wall_elapsed_s', 'lidar_frame', 'lidar_corridor_distance_m', 'lidar_points',
                             'lidar_closing_mps_est', 'lidar_ttc_s_est', 'lidar_warning',
                             'gt_vehicle_id', 'gt_clearance_m', 'gt_closing_mps', 'gt_ttc_s',
                             'gt_warning', 'collision_actor_id'])
            start = time.monotonic()
            last_print = -1
            last_warning = False
            while ego.is_alive:
                now = time.monotonic()
                with lock:
                    snap = latest.copy()
                    impact = collisions.popleft() if collisions else None
                # Only append NEW sensor frames. Never estimate closure from repeated frames.
                if snap['frame'] != last_seen_frame:
                    last_seen_frame = snap['frame']
                    if snap['range'] is not None:
                        history.append((snap['stamp'], snap['range']))
                    else:
                        history.clear()
                lidar_close = None
                lidar_ttc = None
                if len(history) >= 3:
                    # Averaged finite difference; still vulnerable to obstacle switching/noise.
                    t0, r0 = history[0]
                    t1, r1 = history[-1]
                    if t1 > t0 + 0.05:
                        lidar_close = (r0-r1)/(t1-t0)
                        if lidar_close > 0.7 and snap['range'] is not None:
                            lidar_ttc = max(0.0, snap['range'] - ego.bounding_box.extent.x) / lidar_close
                lidar_warn = lidar_ttc is not None and lidar_ttc <= args.warn_ttc
                truth = gt_front_vehicle(world, ego, args.radius)
                gt_warn = truth is not None and truth[3] <= args.warn_ttc
                writer.writerow([round(now-start, 3), snap['frame'],
                                 '' if snap['range'] is None else round(snap['range'], 2), snap['points'],
                                 '' if lidar_close is None else round(lidar_close, 2),
                                 '' if lidar_ttc is None else round(lidar_ttc, 2), int(lidar_warn),
                                 '' if truth is None else truth[0],
                                 '' if truth is None else round(truth[1], 2),
                                 '' if truth is None else round(truth[2], 2),
                                 '' if truth is None or math.isinf(truth[3]) else round(truth[3], 2),
                                 int(gt_warn), '' if impact is None else impact[1]])
                elapsed_sec = int(now-start)
                if elapsed_sec != last_print or (lidar_warn and not last_warning) or impact:
                    last_print = elapsed_sec
                    print('t=%3ds | LiDAR=%s m TTC=%s %s | GT car=%s TTC=%s %s' % (
                        elapsed_sec,
                        '--' if snap['range'] is None else '%.1f' % snap['range'],
                        '--' if lidar_ttc is None else '%.2f' % lidar_ttc,
                        'WARNING' if lidar_warn else '',
                        '--' if truth is None else truth[0],
                        '--' if truth is None or math.isinf(truth[3]) else '%.2f' % truth[3],
                        'WARNING' if gt_warn else ''), flush=True)
                if impact:
                    print('>>> COLLISION EVENT with actor ID', impact[1], '(frame', impact[0], ')', flush=True)
                last_warning = lidar_warn
                if elapsed_sec % 5 == 0:
                    stream.flush()
                time.sleep(0.1)
    except KeyboardInterrupt:
        print('\nMonitoring stopped.', flush=True)
    finally:
        for s in (sensor, collision_sensor):
            if s is not None:
                try:
                    s.stop()
                    s.destroy()
                except RuntimeError:
                    pass
        print('Saved:', log_file, flush=True)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
