"""Temporary CARLA detection test. Spawn one NPC near (but behind) ego car_01.

Run this on the Windows machine hosting CARLA 0.9.16, while the existing
CARLA simulation and generate_traffic.py are running. This script does NOT tick
the world or alter the ego car. It only destroys the NPC it creates.
"""

import math
import time
import carla

HOST = '127.0.0.1'
PORT = 2000
TM_PORT = 8000
TEST_DURATION_S = 30


def distance_xy(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def main():
    client = carla.Client(HOST, PORT)
    client.set_timeout(20.0)
    world = client.get_world()
    ego = None
    test_car = None

    try:
        # Look up ego by ROS ID, avoiding dependence on old Actor IDs.
        vehicles = list(world.get_actors().filter('vehicle.*'))
        for actor in vehicles:
            if actor.attributes.get('ros_name', '') == 'car_01':
                ego = actor
                break
        if ego is None:
            print('STOP: No CARLA vehicle with ros_name=car_01 was found.')
            return

        ego_tf = ego.get_transform()
        ego_pos = ego_tf.location
        ex = math.cos(math.radians(ego_tf.rotation.yaw))
        ey = math.sin(math.radians(ego_tf.rotation.yaw))

        # Choose a lane-aligned spawn point BEHIND the moving ego.
        # Check separation from all vehicles, not only the ego.
        choices = []
        for index, point in enumerate(world.get_map().get_spawn_points()):
            d = distance_xy(point.location, ego_pos)
            if not (22 <= d <= 40):
                continue
            dx = point.location.x - ego_pos.x
            dy = point.location.y - ego_pos.y
            behind = dx * ex + dy * ey
            if behind > -15:
                continue
            angle = math.radians(point.rotation.yaw - ego_tf.rotation.yaw)
            alignment = math.cos(angle)
            if alignment < 0.5:
                continue
            clearance = min(distance_xy(point.location, v.get_location())
                            for v in vehicles)
            if clearance < 15:
                continue
            choices.append((abs(d - 30), index, point, d, clearance))

        if not choices:
            print('SKIPPED: No suitable point 22-40 m behind car_01 right now.')
            print('No vehicles were created. The moving ego may change the available points.')
            return

        choices.sort(key=lambda item: item[0])
        _, index, point, d, clearance = choices[0]
        # Re-read positions just before spawning; the ego is moving.
        latest_ego = ego.get_transform()
        d_now = distance_xy(point.location, latest_ego.location)
        all_vehicles = list(world.get_actors().filter('vehicle.*'))
        closest_now = min(distance_xy(point.location, v.get_location())
                          for v in all_vehicles)
        if not (20 <= d_now <= 45) or closest_now < 15:
            print('SKIPPED: Spawning became unsafe as cars moved.')
            print(f'Ego distance now: {d_now:.1f} m, vehicle clearance: {closest_now:.1f} m')
            return

        blueprint = world.get_blueprint_library().find('vehicle.audi.tt')
        blueprint.set_attribute('role_name', 'autopilot')
        if blueprint.has_attribute('ros_name'):
            blueprint.set_attribute('ros_name', 'test_npc')

        print(f'Chosen spawn INDEX={index}, current ego distance={d_now:.1f}m, '
              f'clearance={closest_now:.1f}m', flush=True)
        test_car = world.try_spawn_actor(blueprint, point)
        if test_car is None:
            print('SKIPPED: CARLA rejected the spawn point (may be occupied).')
            return

        print(f'TEST VEHICLE CREATED: ID={test_car.id}', flush=True)
        test_car.set_autopilot(True, TM_PORT)
        print(f'Watching for {TEST_DURATION_S}s; this test vehicle will be cleaned up automatically.',
              flush=True)
        start = time.monotonic()
        while time.monotonic() - start < TEST_DURATION_S:
            if not ego.is_alive or not test_car.is_alive:
                print('One vehicle is no longer alive; ending test.')
                break
            d = distance_xy(ego.get_location(), test_car.get_location())
            print(f'car_01 <-> test NPC: {d:.1f}m | within 50m: {d <= 50}', flush=True)
            time.sleep(1)
    except KeyboardInterrupt:
        print('Interrupted by user; cleaning up own test car.', flush=True)
    except Exception as exc:
        print(f'Test failed: {exc}', flush=True)
    finally:
        if test_car is not None:
            try:
                actor_id = test_car.id
                test_car.destroy()
                print(f'CLEANED UP TEST VEHICLE ID={actor_id}', flush=True)
            except Exception as exc:
                print(f'Cleanup warning: {exc}', flush=True)
                print('If necessary, manually check and remove ONLY the test_npc actor.')


if __name__ == '__main__':
    main()
