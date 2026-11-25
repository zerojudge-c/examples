"""CARLA 0.9.16: Ego rear-ends a slower moving cooperative car.

Run with CarlaUE4.exe open and other traffic scripts stopped:
    py -3.10 ego_collision_test.py
Optional:
    py -3.10 ego_collision_test.py --gap 40 --ego-speed 32 --lead-speed 8

This uses one synchronous tick owner and collision sensors. No teleporting after spawn.
"""
import argparse
import math
import random
import threading
import time

import carla


def heading_difference(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def select_straight_route(world, gap):
    """Find one long, gently curving drivable lane, with spawn points at both ends."""
    car_map = world.get_map()
    starts = list(car_map.get_spawn_points())
    random.Random(23).shuffle(starts)
    for start in starts:
        wp = car_map.get_waypoint(start.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        if wp is None or wp.is_junction:
            continue
        current = wp
        route = [wp]
        distance = 0.0
        ok = True
        while distance < gap + 30.0:
            next_options = current.next(2.0)
            if len(next_options) != 1:
                ok = False
                break
            nxt = next_options[0]
            if nxt.is_junction or nxt.road_id != wp.road_id or nxt.lane_id != wp.lane_id:
                ok = False
                break
            if heading_difference(nxt.transform.rotation.yaw, wp.transform.rotation.yaw) > 6.0:
                ok = False
                break
            distance += current.transform.location.distance(nxt.transform.location)
            route.append(nxt)
            current = nxt
        if not ok:
            continue
        lead = min(route, key=lambda node: abs(node.transform.location.distance(wp.transform.location) - gap))
        # Vehicle position should be the center of the lane, slightly above road surface.
        ego_t = carla.Transform(wp.transform.location + carla.Location(z=0.35), wp.transform.rotation)
        lead_t = carla.Transform(lead.transform.location + carla.Location(z=0.35), lead.transform.rotation)
        if ego_t.location.distance(lead_t.location) < gap - 3:
            continue
        return ego_t, lead_t
    raise RuntimeError('No suitable straight lane found. Try a different town or --gap 25.')


def vehicle_blueprint(world, name, role, color):
    bp = world.get_blueprint_library().find(name)
    bp.set_attribute('role_name', role)
    if bp.has_attribute('color'):
        bp.set_attribute('color', color)
    return bp


def speed_kmh(actor):
    v = actor.get_velocity()
    return math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2) * 3.6


def main():
    ap = argparse.ArgumentParser(description='Ego hits a slow-moving vehicle in CARLA')
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=2000)
    ap.add_argument('--gap', type=float, default=38.0, help='Initial separation in metres (25-60)')
    ap.add_argument('--ego-speed', type=float, default=35.0, help='Ego target speed in km/h')
    ap.add_argument('--lead-speed', type=float, default=8.0, help='Lead vehicle target speed in km/h')
    ap.add_argument('--max-seconds', type=float, default=25.0)
    args = ap.parse_args()
    if not 25 <= args.gap <= 60:
        ap.error('--gap must be 25-60 metres')
    if args.lead_speed < 0 or args.ego_speed <= args.lead_speed:
        ap.error('ego speed must exceed the nonnegative lead speed')

    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)
    world = client.get_world()
    old_settings = world.get_settings()
    if old_settings.synchronous_mode:
        raise RuntimeError('The world is already synchronous. Stop other traffic/tick scripts, then restart CARLA.')

    cars = []
    sensors = []
    collision_detected = threading.Event()
    collision_data = {}
    try:
        ego_t, lead_t = select_straight_route(world, args.gap)
        ego = world.try_spawn_actor(vehicle_blueprint(world, 'vehicle.tesla.model3', 'hero', '0,80,220'), ego_t)
        if ego is None:
            raise RuntimeError('Ego spawn point occupied. Stop prior vehicle scripts or restart CARLA.')
        cars.append(ego)
        lead = world.try_spawn_actor(vehicle_blueprint(world, 'vehicle.audi.tt', 'car_01', '220,50,40'), lead_t)
        if lead is None:
            raise RuntimeError('Lead spawn point occupied. Stop prior vehicle scripts or restart CARLA.')
        cars.append(lead)

        sensor_bp = world.get_blueprint_library().find('sensor.other.collision')
        for owner, other in ((ego, lead), (lead, ego)):
            sensor = world.spawn_actor(sensor_bp, carla.Transform(), attach_to=owner)
            sensors.append(sensor)
            def callback(event, vehicle_id=owner.id, counterpart_id=other.id):
                if event.other_actor.id == counterpart_id:
                    collision_data['frame'] = event.frame
                    collision_data['vehicle_id'] = vehicle_id
                    collision_detected.set()
            sensor.listen(callback)

        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        # Spectator is positioned once; you can freely move it afterwards.
        middle = carla.Location(
            x=(ego_t.location.x + lead_t.location.x) / 2,
            y=(ego_t.location.y + lead_t.location.y) / 2,
            z=(ego_t.location.z + lead_t.location.z) / 2 + 36,
        )
        world.get_spectator().set_transform(carla.Transform(middle, carla.Rotation(pitch=-90)))
        print('Ego(hero):', ego.id, '| Lead(car_01):', lead.id)
        print('Both cars drive in the same lane. Ego is faster and will rear-end the lead car.')
        print('Camera is free. Press Ctrl+C to end and remove both cars.')

        # At every step, drive forward along the initially selected near-straight lane.
        # set_target_velocity is intentional for reproducible test speeds. It does not
        # teleport the vehicles or disable collision physics.
        ticks = 0
        initial_yaw = math.radians(ego_t.rotation.yaw)
        forward = carla.Vector3D(math.cos(initial_yaw), math.sin(initial_yaw), 0.0)
        lead_forward_yaw = math.radians(lead_t.rotation.yaw)
        lead_forward = carla.Vector3D(math.cos(lead_forward_yaw), math.sin(lead_forward_yaw), 0.0)
        # Bring both cars to their desired movement velocities each tick.
        # These cars do NOT use Traffic Manager, which would brake to avoid collision.
        n_steps = int(args.max_seconds / 0.05)
        for ticks in range(n_steps):
            if collision_detected.is_set():
                break
            ego.set_target_velocity(carla.Vector3D(forward.x * args.ego_speed / 3.6,
                                                   forward.y * args.ego_speed / 3.6, 0))
            lead.set_target_velocity(carla.Vector3D(lead_forward.x * args.lead_speed / 3.6,
                                                    lead_forward.y * args.lead_speed / 3.6, 0))
            world.tick()
            if ticks % 20 == 0:
                gap = ego.get_location().distance(lead.get_location())
                print(f't={ticks * 0.05:4.1f}s | Ego={speed_kmh(ego):4.1f} km/h | '
                      f'car_01={speed_kmh(lead):4.1f} km/h | gap={gap:5.1f} m')
        if collision_detected.is_set():
            print('>>> COLLISION CONFIRMED by sensor! frame=', collision_data.get('frame'))
        else:
            print('>>> No collision detected. Try another map or increase --max-seconds.')

        # Keep accident scene stable without destroying vehicles until Ctrl+C.
        for actor in cars:
            actor.set_target_velocity(carla.Vector3D(0, 0, 0))
            actor.apply_control(carla.VehicleControl(throttle=0, brake=1, hand_brake=True))
        print('Accident scene retained. Ctrl+C cleans up.')
        while True:
            world.tick()
            time.sleep(0.05)
    except KeyboardInterrupt:
        print('\nStopping...')
    finally:
        for sensor in sensors:
            try:
                sensor.stop()
                sensor.destroy()
            except RuntimeError:
                pass
        for actor in cars:
            try:
                actor.destroy()
            except RuntimeError:
                pass
        world.apply_settings(old_settings)
        print('Cleaned up cars and sensors; restored world settings.')


if __name__ == '__main__':
    main()
