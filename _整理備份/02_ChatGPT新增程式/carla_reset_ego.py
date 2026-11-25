#!/usr/bin/env python3
"""Reposition an existing CARLA Ego on a free road spawn and resume autopilot.
Does not spawn/destroy cars, tick the world, or modify world settings.
Run only after stopping collision-control scripts (Ctrl+C).
"""
import argparse
import math
import carla


def d2(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ego-id', type=int, required=True)
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=2000)
    ap.add_argument('--tm-port', type=int, default=8000)
    ap.add_argument('--clearance', type=float, default=12.0, help='minimum XY distance from every other vehicle')
    args = ap.parse_args()
    client = carla.Client(args.host, args.port)
    client.set_timeout(12.0)
    world = client.get_world()
    ego = world.get_actor(args.ego_id)
    if ego is None or not ego.type_id.startswith('vehicle.'):
        print('ERROR: Ego ID not found. Run unity\\CARLA_All_Vehicles_To_5G.py --list')
        return 2
    vehicles = [a for a in world.get_actors().filter('vehicle.*') if a.id != ego.id and a.is_alive]
    walkers = [a for a in world.get_actors().filter('walker.pedestrian.*') if a.is_alive]
    old = ego.get_location()
    candidates = []
    for transform in world.get_map().get_spawn_points():
        loc = transform.location
        if d2(loc, old) < 20.0:
            continue
        if any(d2(loc, a.get_location()) < args.clearance for a in vehicles):
            continue
        if any(d2(loc, a.get_location()) < 5.0 for a in walkers):
            continue
        try:
            wp = world.get_map().get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        except RuntimeError:
            continue
        if wp is None or wp.is_junction:
            continue
        candidates.append(transform)
    if not candidates:
        print('NO SAFE SPAWN FOUND: Nothing changed. Try again after traffic moves.')
        return 3
    # Choose farthest available spawn to avoid immediately returning to the crash scene.
    t = max(candidates, key=lambda p: d2(p.location, old))
    t.location.z += 0.35
    print('WORLD:', world.get_map().name)
    print('EGO:', ego.id, ego.type_id)
    print('Relocating Ego only; other cars, pedestrians and world settings remain unchanged.')
    ego.set_autopilot(False, args.tm_port)
    ego.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))
    ego.set_target_velocity(carla.Vector3D(0, 0, 0))
    ego.set_target_angular_velocity(carla.Vector3D(0, 0, 0))
    ego.set_transform(t)
    ego.apply_control(carla.VehicleControl(throttle=0, brake=0, hand_brake=False))
    ego.set_autopilot(True, args.tm_port)
    print('SUCCESS: Ego moved to an open road and autopilot re-enabled.')
    print('Ego ID is unchanged:', ego.id)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
