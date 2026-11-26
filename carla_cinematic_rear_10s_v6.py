#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CARLA 0.9.16 - 10-s cinematic scripted rear-contact demonstration.

Works with the EXISTING CARLA world, Ego, and one existing NPC. Searches a
clear, nearly straight road, places the two cars in line, and moves both along
that fixed road axis. The cars visually touch after approximately 10 CARLA
simulation seconds and remain frozen. This is a KINEMATIC VISUAL DEMO; it is
NOT a natural physics crash and should not be used to claim LiDAR TTC accuracy.

No world.tick(), map reload, world settings changes, or extra vehicles.

Example:
 py -3.10 carla_cinematic_rear_10s.py --ego-id 112 --seconds 10 --cinematic
 py -3.10 carla_cinematic_rear_10s.py --ego-id 112 --resume
"""
import argparse
import csv
import json
import math
import random
import threading
import time
from datetime import datetime
from pathlib import Path

import carla

STATE = Path(__file__).with_name('carla_cinematic_rear_10s_state.json')
KNOWN_OLDER_STATES = ('carla_straight_forced_10s_state.json',
                      'carla_forced_contact_10s_state.json',
                      'carla_rear_chase_v4_state.json',
                      'carla_rear_chase_v5_state.json')


def transform_at(x, y, z, yaw):
    return carla.Transform(carla.Location(x=x, y=y, z=z), carla.Rotation(yaw=yaw))


def planar(a, b):
    return math.hypot(a.x-b.x, a.y-b.y)


def angle_error(a, b):
    return abs((a-b+180.0) % 360.0-180.0)


def stop_and_freeze(actor):
    if actor and actor.is_alive:
        actor.apply_control(carla.VehicleControl(throttle=0, steer=0, brake=1,
                                                 hand_brake=True))
        actor.set_target_velocity(carla.Vector3D(0, 0, 0))
        actor.set_target_angular_velocity(carla.Vector3D(0, 0, 0))
        actor.set_simulate_physics(False)


def restore_actor(actor, tm_port):
    if actor and actor.is_alive:
        actor.set_simulate_physics(True)
        actor.apply_control(carla.VehicleControl(throttle=0, steer=0, brake=0,
                                                 hand_brake=False))
        actor.set_autopilot(True, tm_port)


def restore_saved(world, ego_id, tm_port):
    if not STATE.exists():
        print('No saved cinematic-contact state. Nothing to resume.')
        return 0
    data = json.loads(STATE.read_text(encoding='utf-8'))
    if data.get('map') != world.get_map().name or data.get('ego') != ego_id:
        print('Saved map/Ego does not match current scene: no actors modified.')
        return 2
    ego = world.get_actor(data['ego'])
    npc = world.get_actor(data['npc'])
    if ego and ego.is_alive and npc and npc.is_alive:
        # Separate BEFORE restoring physics to prevent an instant second impact.
        e = ego.get_transform()
        yaw = math.radians(e.rotation.yaw)
        separation = (float(ego.bounding_box.extent.x) +
                      float(npc.bounding_box.extent.x) + 7.0)
        npc.set_transform(transform_at(e.location.x + math.cos(yaw)*separation,
                                       e.location.y + math.sin(yaw)*separation,
                                       e.location.z, e.rotation.yaw))
    for name, actor in (('Ego', ego), ('NPC', npc)):
        if actor and actor.is_alive:
            restore_actor(actor, tm_port)
            print('RESTORED %s #%d' % (name, actor.id))
    STATE.unlink(missing_ok=True)
    print('RESUME COMPLETE')
    return 0


def find_straight_stage(world, ego, npc, start_gap, npc_distance):
    """Find a clear and visually straight corridor, without altering traffic."""
    mp = world.get_map()
    candidates = list(mp.generate_waypoints(6.0))
    random.shuffle(candidates)
    others = []
    for actor in world.get_actors():
        if not actor.is_alive or actor.id in (ego.id, npc.id):
            continue
        if not (actor.type_id.startswith('vehicle.') or
                actor.type_id.startswith('walker.pedestrian.')):
            continue
        try:
            pos = actor.get_location()
            radius = (max(actor.bounding_box.extent.x, actor.bounding_box.extent.y)
                      if actor.type_id.startswith('vehicle.') else 0.6)
            others.append((pos, radius))
        except RuntimeError:
            pass

    length = start_gap + npc_distance + 12.0
    for entry in candidates:
        if entry.is_junction or entry.lane_type != carla.LaneType.Driving:
            continue
        origin = entry.transform.location
        yaw = entry.transform.rotation.yaw
        rad = math.radians(yaw)
        fx, fy = math.cos(rad), math.sin(rad)
        cursor = entry
        valid = True
        for _ in range(int(math.ceil(length / 3.0))):
            next_wps = cursor.next(3.0)
            if len(next_wps) != 1:
                valid = False
                break
            cursor = next_wps[0]
            loc = cursor.transform.location
            dx, dy = loc.x-origin.x, loc.y-origin.y
            deviation = abs(-dx*fy + dy*fx)
            if (cursor.is_junction or cursor.lane_type != carla.LaneType.Driving
                    or deviation > 0.45 or angle_error(cursor.transform.rotation.yaw, yaw) > 2.5
                    or abs(loc.z-origin.z) > 0.9):
                valid = False
                break
        if not valid:
            continue
        for pos, radius in others:
            if abs(pos.z-origin.z) > 3.0:
                continue
            dx, dy = pos.x-origin.x, pos.y-origin.y
            ahead = dx*fx + dy*fy
            lateral = abs(-dx*fy + dy*fx)
            if -8.0-radius <= ahead <= length+radius and lateral < 4.5+radius:
                valid = False
                break
        if valid:
            return (origin.x, origin.y, origin.z+0.45,
                    yaw, entry.road_id, entry.lane_id)
    return None


def set_cinematic_view(world, x, y, z, yaw, distance=15.0):
    """Set spectator ONCE. Avoid per-frame camera commands and their overhead."""
    rad = math.radians(yaw)
    fx, fy = math.cos(rad), math.sin(rad)
    # Elevated, rear-side three-quarter perspective, looking toward the cars.
    cam_x = x - fx*distance - fy*8.0
    cam_y = y - fy*distance + fx*8.0
    cam_z = z + 8.5
    aim_x = x + fx*10.0
    aim_y = y + fy*10.0
    dx, dy, dz = aim_x-cam_x, aim_y-cam_y, z+0.8-cam_z
    yaw_cam = math.degrees(math.atan2(dy, dx))
    pitch_cam = math.degrees(math.atan2(dz, math.hypot(dx, dy)))
    world.get_spectator().set_transform(carla.Transform(
        carla.Location(x=cam_x, y=cam_y, z=cam_z),
        carla.Rotation(pitch=pitch_cam, yaw=yaw_cam)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ego-id', required=True, type=int)
    p.add_argument('--npc-id', type=int, default=None)
    p.add_argument('--gt-target-file', default='chase_target.txt',
                   help='Write the selected NPC ID here for the V5 monitor to hot-load')
    p.add_argument('--seconds', type=float, default=10.0)
    p.add_argument('--gap', type=float, default=20.0)
    p.add_argument('--npc-kmh', type=float, default=7.0,
                   help='Visual motion speed of NPC along straight line')
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=2000)
    p.add_argument('--tm-port', type=int, default=8000)
    p.add_argument('--update-hz', type=float, default=30.0,
                   help='Maximum pose updates per CARLA SIM second, 10..60')
    p.add_argument('--timeout-wall', type=float, default=120.0)
    p.add_argument('--cinematic', action='store_true',
                   help='Position spectator camera once at a static 3/4 angle')
    p.add_argument('--diagnose', action='store_true')
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    if not (5 <= a.seconds <= 30 and 12 <= a.gap <= 35 and
            0 <= a.npc_kmh <= 12 and 10 <= a.update_hz <= 60 and
            30 <= a.timeout_wall <= 300):
        p.error('Require --seconds 5..30 --gap 12..35 --npc-kmh 0..12 '
                '--update-hz 10..60 --timeout-wall 30..300')

    client = carla.Client(a.host, a.port)
    client.set_timeout(10.0)
    world = client.get_world()
    mp_name = world.get_map().name
    print('MAP:', mp_name, '| no world.tick(), no world reload/settings changes', flush=True)
    if a.resume:
        return restore_saved(world, a.ego_id, a.tm_port)
    if STATE.exists():
        print('Previous cinematic demo frozen. Run --resume first.')
        return 2
    for filename in KNOWN_OLDER_STATES:
        if Path(__file__).with_name(filename).exists():
            print('Found previous frozen state:', filename)
            print('Restore that demo with its matching script first.')
            return 2

    ego = world.get_actor(a.ego_id)
    if ego is None or not ego.is_alive or not ego.type_id.startswith('vehicle.'):
        print('ERROR: Ego #%d does not exist.' % a.ego_id)
        return 2
    if a.npc_id is not None:
        npc = world.get_actor(a.npc_id)
        if not npc or not npc.is_alive or not npc.type_id.startswith('vehicle.') or npc.id == ego.id:
            print('ERROR: --npc-id does not refer to an existing other vehicle.')
            return 2
    else:
        pool = [v for v in world.get_actors().filter('vehicle.*')
                if v.id != ego.id and v.is_alive and
                v.attributes.get('role_name', '') not in ('hero', 'ego') and
                v.bounding_box.extent.x >= 1.3]
        if not pool:
            print('ERROR: No suitable existing NPC vehicle.')
            return 2
        npc = random.choice(pool)

    # The NPC moves forward while Ego closes the gap. Both are kinematic during
    # the visual demo, so the presentation is stable and intentionally scripted.
    npc_travel = a.npc_kmh / 3.6 * a.seconds
    site = find_straight_stage(world, ego, npc, a.gap, npc_travel)
    if site is None:
        print('ABORT BEFORE MOVING: no clear straight corridor, other traffic unchanged.')
        return 1

    # Automatically synchronize the actual selected NPC ID with the V5 monitor.
    target_file = Path(a.gt_target_file)
    if not target_file.is_absolute():
        target_file = Path(__file__).with_name(a.gt_target_file)
    try:
        target_file.parent.mkdir(parents=True, exist_ok=True)
        target_file.write_text(str(npc.id), encoding='ascii')
        print('GT TARGET FILE UPDATED:', str(target_file), '-> NPC', npc.id, flush=True)
    except OSError as exc:
        print('WARNING: could not update GT target file:', exc, flush=True)

    ox, oy, oz, yaw, road, lane = site
    theta = math.radians(yaw)
    fx, fy = math.cos(theta), math.sin(theta)
    full_body = float(ego.bounding_box.extent.x) + float(npc.bounding_box.extent.x)
    start_gap = max(a.gap, full_body+3.0)
    final_gap = max(0.3, full_body-0.45)  # deliberate overlap only at finish
    print('STAGING: Ego=%d NPC=%d | road=%d lane=%d' %
          (ego.id, npc.id, road, lane), flush=True)
    print('CINEMATIC 10s: start center_gap=%.2fm | end center_gap=%.2fm' %
          (start_gap, final_gap), flush=True)
    print('NOTE: geometric overlap is SCRIPTED; collision sensor may not fire.', flush=True)

    event = threading.Event()
    armed = threading.Event()
    sensor = None
    rows = []
    frozen = False
    reached = False
    wall_start = time.monotonic()
    sim_start = None
    last_log_second = -1
    last_move_t = -1e6
    old_ego_pose = ego.get_transform()
    old_npc_pose = npc.get_transform()
    output = Path(__file__).with_name('cinematic_rear_10s_%s.csv' %
                                     datetime.now().strftime('%Y%m%d_%H%M%S'))

    try:
        for vehicle in (ego, npc):
            vehicle.set_autopilot(False, a.tm_port)
            vehicle.set_simulate_physics(True)
            vehicle.apply_control(carla.VehicleControl(throttle=0, steer=0, brake=1))
            vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))
            vehicle.set_target_angular_velocity(carla.Vector3D(0, 0, 0))
        ego.set_transform(transform_at(ox, oy, oz, yaw))
        npc.set_transform(transform_at(ox+fx*start_gap, oy+fy*start_gap, oz, yaw))
        ego.set_simulate_physics(False)
        npc.set_simulate_physics(False)
        if a.cinematic:
            set_cinematic_view(world, ox, oy, oz, yaw)
            print('CINEMATIC CAMERA: fixed side/rear perspective enabled.', flush=True)

        bp = world.get_blueprint_library().find('sensor.other.collision')
        sensor = world.spawn_actor(bp, carla.Transform(), attach_to=ego)
        def on_collision(ev):
            if (armed.is_set() and ev.other_actor and ev.other_actor.id == npc.id):
                event.set()
        sensor.listen(on_collision)
        armed.set()
        sim_start = world.get_snapshot().timestamp.elapsed_seconds
        print('START: both vehicles move along one straight line for ~%.1f SIM seconds.' %
              a.seconds, flush=True)

        # Passive snapshot waiting. This script does not advance or own the world clock.
        while time.monotonic() - wall_start < a.timeout_wall:
            if not ego.is_alive or not npc.is_alive:
                print('STOP: a test actor disappeared.')
                break
            try:
                snap = world.wait_for_tick(seconds=2.0)
            except RuntimeError as exc:
                print('WORLD TICK UNAVAILABLE:', exc)
                break
            if snap is None:
                print('WAITING: CARLA world clock has not advanced.', flush=True)
                continue
            t = max(0.0, float(snap.timestamp.elapsed_seconds) - sim_start)
            if t < a.seconds and t-last_move_t < 1.0/a.update_hz:
                continue
            last_move_t = t
            progress = min(1.0, t/a.seconds)
            # Linear spacing gives a constant relative approach speed, so
            # distance-vs-time is understandable for the LiDAR bird's-eye demo.
            center_gap = start_gap + (final_gap-start_gap)*progress
            forward_m = npc_travel*progress
            npc_x, npc_y = ox+fx*(start_gap+forward_m), oy+fy*(start_gap+forward_m)
            ego_x, ego_y = npc_x-fx*center_gap, npc_y-fy*center_gap
            npc.set_transform(transform_at(npc_x, npc_y, oz, yaw))
            ego.set_transform(transform_at(ego_x, ego_y, oz, yaw))
            reached = progress >= 1.0
            rows.append((round(t, 4), round(center_gap, 4),
                         round(forward_m, 4), int(reached), int(event.is_set())))
            if int(t) != last_log_second:
                print('SIM %5.2fs | distance %5.2fm | %3.0f%% | sensor=%s' %
                      (t, center_gap, progress*100, event.is_set()), flush=True)
                last_log_second = int(t)
            if reached:
                break

        if reached:
            stop_and_freeze(ego)
            stop_and_freeze(npc)
            STATE.write_text(json.dumps({'map': mp_name, 'ego': ego.id,
                                         'npc': npc.id, 'mode': 'scripted_kinematic_overlap'}),
                             encoding='utf-8')
            frozen = True
            print('FORCED GEOMETRIC CONTACT at %.2f CARLA SIM seconds' % t, flush=True)
            print('COLLISION SENSOR CONFIRMED:', 'YES' if event.is_set() else 'NO', flush=True)
            print('BOTH VEHICLES FROZEN. Run --resume to release.', flush=True)
            return 0
        else:
            print('NO CONTACT: CARLA clock stopped or demo interrupted.')
            return 1
    except (RuntimeError, KeyboardInterrupt) as exc:
        print('STOP:', exc)
        return 3
    finally:
        armed.clear()
        if sensor is not None:
            try:
                sensor.stop()
                sensor.destroy()
            except RuntimeError:
                pass
        if not frozen:
            for actor in (ego, npc):
                try:
                    restore_actor(actor, a.tm_port)
                except RuntimeError:
                    pass
        with output.open('w', newline='', encoding='utf-8-sig') as handle:
            writer = csv.writer(handle)
            writer.writerow(['elapsed_sim_s', 'center_gap_m', 'npc_motion_m',
                             'scripted_contact', 'collision_sensor_confirmed'])
            writer.writerows(rows)
        print('CSV SAVED:', output, flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
