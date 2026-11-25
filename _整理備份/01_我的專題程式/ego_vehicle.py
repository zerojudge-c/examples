import carla
import random


def get_speed(vehicle):
    velocity = vehicle.get_velocity()

    speed = (
        velocity.x ** 2
        + velocity.y ** 2
        + velocity.z ** 2
    ) ** 0.5

    return speed * 3.6


def main():

    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)

    world = client.get_world()

    print("Connected to CARLA")
    print("Map:", world.get_map().name)

    # 使用 generate_traffic.py 同一個 Traffic Manager
    tm_port = 8000
    traffic_manager = client.get_trafficmanager(tm_port)

    blueprint_library = world.get_blueprint_library()

    ego_bp = blueprint_library.find("vehicle.tesla.model3")
    ego_bp.set_attribute("role_name", "hero")

    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    ego_vehicle = None

    for spawn_point in spawn_points:
        ego_vehicle = world.try_spawn_actor(
            ego_bp,
            spawn_point
        )

        if ego_vehicle is not None:
            break

    if ego_vehicle is None:
        print("Failed to spawn Ego Vehicle")
        return

    print("Ego Vehicle spawned!")
    print("ID:", ego_vehicle.id)

    # 加入 Traffic Manager
    ego_vehicle.set_autopilot(True, tm_port)

    # 比一般 Traffic 慢 10%
    traffic_manager.vehicle_percentage_speed_difference(
        ego_vehicle,
        10.0
    )

    spectator = world.get_spectator()

    print("Ego Vehicle is now driving.")
    print("Press Ctrl+C to stop.")

    frame_count = 0

    try:
        while True:

            # 很重要：
            # generate_traffic.py 負責 tick
            # Ego client 只等待 tick
            snapshot = world.wait_for_tick()

            actor_snapshot = snapshot.find(ego_vehicle.id)

            if actor_snapshot is None:
                continue

            transform = actor_snapshot.get_transform()

            location = transform.location
            rotation = transform.rotation

            # 第三人稱跟車視角
            spectator_transform = carla.Transform(
                transform.transform(
                    carla.Location(
                        x=-8.0,
                        z=4.0
                    )
                ),
                carla.Rotation(
                    pitch=-15.0,
                    yaw=rotation.yaw
                )
            )

            spectator.set_transform(
                spectator_transform
            )

            # 降低 console print 頻率
            frame_count += 1

            if frame_count % 10 == 0:

                print(
                    f"\rLocation: "
                    f"x={location.x:.2f}, "
                    f"y={location.y:.2f}, "
                    f"z={location.z:.2f}, "
                    f"speed={get_speed(ego_vehicle):.1f} km/h",
                    end=""
                )

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:

        if ego_vehicle is not None:
            ego_vehicle.destroy()

        print("Ego Vehicle destroyed.")


if __name__ == "__main__":
    main()