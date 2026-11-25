import carla
import time
import numpy as np


def lidar_callback(data):
    # CARLA LiDAR raw_data:
    # x, y, z, intensity
    points = np.frombuffer(data.raw_data, dtype=np.float32)
    points = points.reshape((-1, 4))

    xyz = points[:, :3]

    print(
        f"[LiDAR] frame={data.frame} "
        f"points={len(points)} "
        f"sample=({xyz[0][0]:.2f}, "
        f"{xyz[0][1]:.2f}, "
        f"{xyz[0][2]:.2f})"
    )


def find_ego_vehicle(world):
    vehicles = world.get_actors().filter("vehicle.*")

    print(f"目前世界中共有 {len(vehicles)} 台車")

    for vehicle in vehicles:
        role_name = vehicle.attributes.get("role_name", "")

        if role_name in ["hero", "ego"]:
            print(
                f"找到 Ego Vehicle: "
                f"id={vehicle.id}, "
                f"type={vehicle.type_id}, "
                f"role_name={role_name}"
            )
            return vehicle

    print("找不到 role_name=hero 或 ego 的車")

    print("\n目前車輛 role_name：")
    for vehicle in vehicles[:10]:
        print(
            vehicle.id,
            vehicle.type_id,
            vehicle.attributes.get("role_name", "")
        )

    return None


def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)

    world = client.get_world()

    print("Connected to CARLA")
    print("Map:", world.get_map().name)

    # --------------------------------------------------
    # 找 Ego Vehicle
    # --------------------------------------------------

    ego_vehicle = find_ego_vehicle(world)

    if ego_vehicle is None:
        print(
            "\n請確認你的 Ego Vehicle blueprint "
            "有設定 role_name='hero' 或 'ego'"
        )
        return

    # --------------------------------------------------
    # 建立 LiDAR
    # --------------------------------------------------

    blueprint_library = world.get_blueprint_library()

    lidar_bp = blueprint_library.find(
        "sensor.lidar.ray_cast"
    )

    # 基本 LiDAR 設定
    lidar_bp.set_attribute(
        "channels",
        "32"
    )

    lidar_bp.set_attribute(
        "range",
        "50"
    )

    lidar_bp.set_attribute(
        "points_per_second",
        "100000"
    )

    lidar_bp.set_attribute(
        "rotation_frequency",
        "10"
    )

    lidar_bp.set_attribute(
        "upper_fov",
        "10"
    )

    lidar_bp.set_attribute(
        "lower_fov",
        "-30"
    )

    # --------------------------------------------------
    # LiDAR 安裝位置
    #
    # Ego 車頂中央附近
    # --------------------------------------------------

    lidar_transform = carla.Transform(
        carla.Location(
            x=0.0,
            y=0.0,
            z=2.4
        )
    )

    lidar = world.spawn_actor(
        lidar_bp,
        lidar_transform,
        attach_to=ego_vehicle
    )

    print(
        f"LiDAR 建立成功: id={lidar.id}"
    )

    print("開始接收 LiDAR 點雲...")
    print("按 Ctrl+C 結束")

    # --------------------------------------------------
    # 開始監聽
    # --------------------------------------------------

    lidar.listen(lidar_callback)

    try:
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n停止 LiDAR")

    finally:
        if lidar is not None:
            lidar.stop()
            lidar.destroy()

        print("LiDAR 已移除")


if __name__ == "__main__":
    main()