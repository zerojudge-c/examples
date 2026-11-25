#!/usr/bin/env python3
"""Live CARLA 0.9.16: Ego drives into an EXISTING car after a measurable approach.

No world reload, no world.tick, no spawning/deleting vehicles, no changes to
other traffic. Ego is moved behind a suitable existing car on a clear straight
segment. The target is temporarily held stationary; Ego accelerates. This is a
controlled scenario, not a LiDAR/5G perception algorithm.

Usage: py -3.10 carla_live_warning_collision.py --ego-id 200
"""
import argparse
import csv
import math
import threading
import time
from datetime import datetime
from pathlib import Path

import carla


def speed_kmh(actor):
    v = actor.get_velocity()
    return math.sqrt(v.x*v.x + v.y*v.y + v.z*v.z) * 3.6


def point_distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def along_and_lateral(ego_loc, target_loc, target_yaw):
    theta = math.radians(target_yaw)
    fx, fy = math.cos(theta), math.sin(theta)
    dx, dy = target_loc.x - ego_loc.x, target_loc.y - ego_loc.y
    return dx * fx + dy * fy, abs(-dx * fy + dy * fx)


def candidate_start(world, target, distance, vehicles):
    try:
        target_loc = target.get_location()
        w = world.get_map().get_waypoint(target_loc, project_to_road=True,
                                          lane_type=carla.LaneType.Driving)
        if w is None or w.is_junction:
            return None
        if point_distance(w.transform.location, target_loc) > 1.8:
            return None
        # Verify that the entire approach is a non-junction, relatively straight lane.
        sequence = [w]
        cursor = w
        for _ in range(int(math.ceil(distance / 2.0))):
            prev = cursor.previous(2.0)
            if len(prev) != 1:
                return None
            cursor = prev[0]
            if (cursor.is_junction or cursor.road_id != w.road_id or
                    cursor.lane_id != w.lane_id):
                return None
            turn = (cursor.transform.rotation.yaw - w.transform.rotation.yaw + 180) % 360 - 180
            if abs(turn) > 5:
                return None
            sequence.append(cursor)
        start = sequence[-1].transform
        start.location.z = w.transform.location.z + 0.30
        start_loc = start.location
        along, lateral = along_and_lateral(start_loc, target_loc, w.transform.rotation.yaw)
        if along < distance * 0.85 or lateral > 1.4:
            return None
        # The entire approach needs to be clear, not only the spawn position.
        for car in vehicles:
            if car.id == target.id or not car.is_alive:
                continue
            loc = car.get_location()
            if point_distance(loc, start_loc) < 6.5:
                return None
            # Block vehicles that sit between the planned start and target.
            forward, side = along_and_lateral(start_loc, loc, w.transform.rotation.yaw)
            if 0 < forward < along and side < 3.5:
                return None
        return start
    except RuntimeError:
        return None


def run():
    parser = argparse.ArgumentParser(description='Live Ego rear-end collision with TTC warning, CSV and auto-release')
    parser.add_argument('--ego-id', type=int, required=True, help='Existing Ego actor ID from --list')
    parser.add_argument('--other-id', type=int, default=None, help='Optional existing target car ID')
    parser.add_argument('--gap', type=float, default=35, help='Start distance in metres (25..55)')
    parser.add_argument('--warn-ttc', type=float, default=4.0, help='Warning threshold in seconds')
    parser.add_argument('--timeout', type=float, default=25, help='Max experiment duration, wall seconds')
    parser.add_argument('--hold-after', type=float, default=2.0, help='Seconds to hold cars after result, then auto release')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--tm-port', type=int, default=8000)
    args = parser.parse_args()
    if not 25 <= args.gap <= 55:
        parser.error('--gap must be 25..55')

    client = carla.Client(args.host, args.port)
    client.set_timeout(12)
    world = client.get_world()
    ego = world.get_actor(args.ego_id)
    if ego is None or not ego.type_id.startswith('vehicle.'):
        print('ERROR: Ego actor not found. Recheck ID with CARLA_All_Vehicles_To_5G.py --list')
        return 2
    actors = list(world.get_actors().filter('vehicle.*'))
    targets = [v for v in actors if v.id != ego.id and v.is_alive
               and v.attributes.get('role_name') != 'hero'
               and v.bounding_box.extent.x > 1.5]
    if args.other_id is not None:
        targets = [v for v in targets if v.id == args.other_id]
    else:
        targets.sort(key=lambda v: point_distance(v.get_location(), ego.get_location()))

    chosen = start = None
    for target in targets:
        candidate = candidate_start(world, target, args.gap,
                                    [a for a in actors if a.id != ego.id])
        if candidate is not None:
            chosen, start = target, candidate
            break
    if chosen is None:
        print('No clear, straight approach available. Traffic can remain open.')
        print('Retry later, or use --gap 25; optionally use --other-id <id>.')
        return 2
    target = chosen
    print('WORLD:', world.get_map().name, flush=True)
    print('EGO:', ego.id, ego.type_id, 'TARGET:', target.id, target.type_id, flush=True)
    print('Existing vehicles remain; only Ego is repositioned. Target is held stationary.', flush=True)
    print('No world.tick() or change to simulation settings.', flush=True)

    impacts = []
    impacts_lock = threading.Lock()
    sensor = None
    armed = threading.Event()
    log_rows = []
    first_warning_time = None
    collision_time = None
    valid_hit = False
    start_wall = None
    start_pos = None
    file_path = Path(__file__).with_name('carla_auto_release_collision_%s.csv' % datetime.now().strftime('%Y%m%d_%H%M%S'))
    try:
        ego.set_autopilot(False, args.tm_port)
        target.set_autopilot(False, args.tm_port)
        ego.apply_control(carla.VehicleControl(brake=1, hand_brake=True))
        target.apply_control(carla.VehicleControl(brake=1, hand_brake=True))
        target.set_target_velocity(carla.Vector3D(0, 0, 0))
        ego.set_transform(start)
        ego.set_target_velocity(carla.Vector3D(0, 0, 0))
        ego.set_target_angular_velocity(carla.Vector3D(0, 0, 0))
        sensor = world.spawn_actor(world.get_blueprint_library().find('sensor.other.collision'),
                                   carla.Transform(), attach_to=ego)

        def on_collision(event):
            if not armed.is_set():
                return
            if event.other_actor.id == target.id:
                with impacts_lock:
                    impacts.append((event.frame, time.monotonic()))

        sensor.listen(on_collision)
        print('Stabilizing physics and arming Collision Sensor...', flush=True)
        for _ in range(15):
            target.apply_control(carla.VehicleControl(brake=1, hand_brake=True))
            ego.apply_control(carla.VehicleControl(brake=1, hand_brake=True))
            time.sleep(0.07)
        start_pos = ego.get_location()
        initial_along, initial_lateral = along_and_lateral(start_pos, target.get_location(), start.rotation.yaw)
        if initial_along < 20 or initial_lateral > 2.0:
            print('ABORT: the cars are no longer safely lined up after placement.')
            return 3
        print('READY: separation %.1f m; beginning Ego approach...' % initial_along, flush=True)
        start_wall = time.monotonic()
        armed.set()
        last_print = -1
        while time.monotonic() - start_wall < args.timeout:
            if not ego.is_alive or not target.is_alive:
                print('ABORT: Ego or target disappeared')
                break
            now = time.monotonic()
            elapsed = now - start_wall
            ego_loc = ego.get_location()
            target_loc = target.get_location()
            moved = point_distance(ego_loc, start_pos)
            forward, lateral = along_and_lateral(ego_loc, target_loc, start.rotation.yaw)
            # Approximate surface gap using actor bounding-box half-lengths.
            clearance = max(0.0, forward - ego.bounding_box.extent.x - target.bounding_box.extent.x)
            closing_mps = (speed_kmh(ego) - speed_kmh(target)) / 3.6
            ttc = clearance / closing_mps if closing_mps > 0.5 and forward > 0 and lateral < 3 else float('inf')
            warning = (ttc <= args.warn_ttc and moved >= 2)
            if warning and first_warning_time is None:
                first_warning_time = elapsed
                print('>>> WARNING: TTC %.2f s | Ego moved %.1f m | gap %.1f m' %
                      (ttc, moved, clearance), flush=True)
            with impacts_lock:
                impact = impacts[0] if impacts else None
            if impact:
                collision_time = elapsed
                v = speed_kmh(ego)
                valid_hit = moved >= 10 and v >= 5
                print('>>> CONTACT WITH TARGET: frame=%d, Ego moved=%.1fm, speed=%.1fkm/h' %
                      (impact[0], moved, v), flush=True)
                if valid_hit:
                    print('>>> VERIFIED DRIVING COLLISION: approach >=10m, impact speed >=5km/h', flush=True)
                else:
                    print('>>> CONTACT NOT VERIFIED AS DRIVING COLLISION (too early/slow)', flush=True)
            log_rows.append([round(elapsed, 3), round(moved, 2), round(forward, 2),
                             round(lateral, 2), round(clearance, 2), round(speed_kmh(ego), 2),
                             round(speed_kmh(target), 2), '' if math.isinf(ttc) else round(ttc, 2),
                             int(warning), int(bool(impact)), int(valid_hit)])
            second = int(elapsed)
            if second != last_print:
                last_print = second
                print('t=%2ds | moved=%5.1fm | gap=%5.1fm | Ego=%4.1fkm/h | TTC=%s' %
                      (second, moved, clearance, speed_kmh(ego), '--' if math.isinf(ttc) else '%.2fs' % ttc), flush=True)
            if impact:
                break
            target.apply_control(carla.VehicleControl(brake=1, hand_brake=True))
            ego.apply_control(carla.VehicleControl(throttle=0.75, steer=0, brake=0, hand_brake=False))
            time.sleep(0.05)
        else:
            print('TIMEOUT: no confirmed contact during this run.', flush=True)
        ego.apply_control(carla.VehicleControl(brake=1, hand_brake=True))
        target.apply_control(carla.VehicleControl(brake=1, hand_brake=True))
        if first_warning_time is not None and collision_time is not None:
            print('WARNING LEAD TIME: %.2f seconds' % (collision_time - first_warning_time), flush=True)
        elif collision_time is not None:
            print('WARNING LEAD TIME: unavailable (no warning before contact)', flush=True)
        print('Releasing cars automatically after %.1f seconds. Other traffic keeps moving.' % max(0., args.hold_after), flush=True)
        time.sleep(max(0., args.hold_after))
    except KeyboardInterrupt:
        print('\nStopping experiment; releasing vehicles.', flush=True)
    finally:
        armed.clear()
        if sensor is not None:
            try:
                sensor.stop()
                sensor.destroy()
            except RuntimeError:
                pass
        for car in (ego, target):
            try:
                if car.is_alive:
                    car.apply_control(carla.VehicleControl(brake=1))
                    car.set_autopilot(True, args.tm_port)
            except RuntimeError:
                pass
        with file_path.open('w', newline='', encoding='utf-8-sig') as output:
            writer = csv.writer(output)
            writer.writerow(['elapsed_s','ego_travel_m','forward_distance_m','lateral_offset_m',
                             'surface_gap_m','ego_kmh','target_kmh','ttc_s',
                             'warning','target_contact','verified_driving_collision'])
            writer.writerows(log_rows)
        print('CSV SAVED:', file_path, flush=True)
        print('No vehicles deleted. Ego/target returned to autopilot.', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(run())
