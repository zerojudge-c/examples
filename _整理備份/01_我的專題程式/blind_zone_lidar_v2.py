import carla
import numpy as np
import cv2

import math
import threading
import time


# ============================================================
# Settings
# ============================================================

HOST = "localhost"
PORT = 2000

MAX_DISTANCE = 50.0

# Blind 判定條件
OCCLUSION_THRESHOLD = 0.60
MIN_DISTANCE_GAP = 2.0

# LiDAR
LIDAR_RANGE = 50.0
LIDAR_CHANNELS = 32
LIDAR_POINTS_PER_SECOND = 100000
LIDAR_ROTATION_FREQUENCY = 10.0
LIDAR_SENSOR_TICK = 0.1
LIDAR_HEIGHT = 2.4

# BEV
IMAGE_SIZE = 800
METERS_RANGE = 50.0

# LiDAR 高度過濾
# LiDAR 本身在車頂約 z=2.4 m
# 地面通常會落在約 -2.4 附近
# 第一階段先保留完整 LiDAR
USE_Z_FILTER = False

MIN_Z = -2.0
MAX_Z = 3.0

# 幾何計算更新頻率
GEOMETRY_UPDATE_INTERVAL = 0.01


# ============================================================
# LiDAR shared data
# ============================================================

latest_points = np.empty((0, 4), dtype=np.float32)

lidar_lock = threading.Lock()


def lidar_callback(lidar_data):
    """
    CARLA LiDAR callback。

    每一個點：
        x
        y
        z
        intensity
    """

    global latest_points

    points = np.frombuffer(
        lidar_data.raw_data,
        dtype=np.float32
    )

    points = points.reshape((-1, 4)).copy()

    with lidar_lock:
        latest_points = points


# ============================================================
# Angle functions
# ============================================================

def wrap_angle(angle):
    """
    限制在 [-pi, pi]
    """

    return math.atan2(
        math.sin(angle),
        math.cos(angle)
    )


def angle_diff(a, b):
    """
    最短角度差
    """

    return wrap_angle(a - b)


# ============================================================
# World -> Ego coordinate
# ============================================================

def world_to_ego(point, ego_transform):
    """
    CARLA World coordinate
            ↓
        Ego coordinate

    Ego coordinate:

            +X
         Forward
            ↑
            │
      -Y ← Ego → +Y

    """

    ego_location = ego_transform.location

    dx = point.x - ego_location.x
    dy = point.y - ego_location.y

    yaw = math.radians(
        ego_transform.rotation.yaw
    )

    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)

    x_local = (
        cos_yaw * dx
        + sin_yaw * dy
    )

    y_local = (
        -sin_yaw * dx
        + cos_yaw * dy
    )

    return x_local, y_local


# ============================================================
# Vehicle angular projection
# ============================================================

def get_vehicle_projection(actor, ego_transform):
    """
    計算一台車在 Ego 看出去的：

        distance
        center angle
        angular half width

    angular half width 是利用 CARLA Bounding Box
    的 8 個角計算。
    """

    try:

        location = actor.get_location()

        x, y = world_to_ego(
            location,
            ego_transform
        )

        distance = math.hypot(x, y)

        if distance < 0.5:
            return None

        center_angle = math.atan2(
            y,
            x
        )

        bbox = actor.bounding_box

        vertices = bbox.get_world_vertices(
            actor.get_transform()
        )

        angular_offsets = []

        for vertex in vertices:

            vx, vy = world_to_ego(
                vertex,
                ego_transform
            )

            if math.hypot(vx, vy) < 0.1:
                continue

            corner_angle = math.atan2(
                vy,
                vx
            )

            offset = angle_diff(
                corner_angle,
                center_angle
            )

            angular_offsets.append(
                abs(offset)
            )

        if not angular_offsets:
            return None

        half_width = max(
            angular_offsets
        )

        # 用 bbox 平面尺寸估計車子的半徑
        radius = math.hypot(
            bbox.extent.x,
            bbox.extent.y
        )

        return {
            "actor": actor,
            "id": actor.id,
            "type": actor.type_id,

            "location": location,

            "x": x,
            "y": y,

            "distance": distance,

            "angle": center_angle,
            "half_width": half_width,

            "radius": radius
        }

    except RuntimeError:
        # Actor 剛好被 CARLA 刪掉
        return None


# ============================================================
# Angular overlap
# ============================================================

def angular_overlap_ratio(target, blocker):
    """
    計算 blocker 遮住 target 的角度比例。

    target:

         |-------------|

    blocker:

             |------|

    overlap ratio:

          overlap
        -------------
        target width
    """

    target_half = target["half_width"]
    blocker_half = blocker["half_width"]

    relative_angle = angle_diff(
        blocker["angle"],
        target["angle"]
    )

    # target 以自己中心角為 0
    target_left = -target_half
    target_right = target_half

    blocker_left = (
        relative_angle
        - blocker_half
    )

    blocker_right = (
        relative_angle
        + blocker_half
    )

    overlap_left = max(
        target_left,
        blocker_left
    )

    overlap_right = min(
        target_right,
        blocker_right
    )

    overlap = max(
        0.0,
        overlap_right - overlap_left
    )

    target_width = (
        2.0 * target_half
    )

    if target_width < 1e-6:
        return 0.0

    return overlap / target_width


# ============================================================
# Blind Zone Detection
# ============================================================

def detect_blind_vehicles(projections):

    results = []

    for target in projections:

        best_blocker = None
        best_overlap = 0.0

        for blocker in projections:

            if target["id"] == blocker["id"]:
                continue

            # blocker 必須比 target 靠近 Ego
            if blocker["distance"] >= (
                target["distance"]
                - MIN_DISTANCE_GAP
            ):
                continue

            overlap = angular_overlap_ratio(
                target,
                blocker
            )

            if overlap > best_overlap:

                best_overlap = overlap
                best_blocker = blocker

        if (
            best_blocker is not None
            and
            best_overlap >= OCCLUSION_THRESHOLD
        ):

            results.append({
                "target": target,
                "blocker": best_blocker,
                "overlap": best_overlap
            })

    return results


# ============================================================
# BEV coordinate conversion
# ============================================================

def ego_to_pixel(x, y):
    """
    Ego coordinate -> Image pixel

                  forward
                     ↑
                     |
                     |
              Ego = center
    """

    scale = (
        IMAGE_SIZE
        / (METERS_RANGE * 2.0)
    )

    center = IMAGE_SIZE // 2

    px = int(
        center + y * scale
    )

    py = int(
        center - x * scale
    )

    return px, py


# ============================================================
# Draw LiDAR points
# ============================================================

def draw_lidar_points(image, points):

    if len(points) == 0:
        return

    # ----------------------------------------------------
    # Optional Z filter
    # ----------------------------------------------------

    if USE_Z_FILTER:

        mask = (
            (points[:, 2] > MIN_Z)
            &
            (points[:, 2] < MAX_Z)
        )

        points = points[mask]

    if len(points) == 0:
        return

    # ----------------------------------------------------
    # LiDAR coordinates
    # ----------------------------------------------------

    x = points[:, 0]
    y = points[:, 1]

    # 只限制 BEV 顯示範圍
    valid = (
        (x >= -METERS_RANGE)
        &
        (x <= METERS_RANGE)
        &
        (y >= -METERS_RANGE)
        &
        (y <= METERS_RANGE)
    )

    x = x[valid]
    y = y[valid]

    if len(x) == 0:
        return

    # ----------------------------------------------------
    # Ego coordinate -> Pixel
    # ----------------------------------------------------

    scale = (
        IMAGE_SIZE
        / (METERS_RANGE * 2.0)
    )

    center = IMAGE_SIZE // 2

    px = (
        center + y * scale
    ).astype(np.int32)

    py = (
        center - x * scale
    ).astype(np.int32)

    valid_pixel = (
        (px >= 0)
        &
        (px < IMAGE_SIZE)
        &
        (py >= 0)
        &
        (py < IMAGE_SIZE)
    )

    px = px[valid_pixel]
    py = py[valid_pixel]

    # ----------------------------------------------------
    # 建立獨立 LiDAR layer
    # ----------------------------------------------------

    lidar_layer = np.zeros_like(image)

    lidar_layer[
        py,
        px
    ] = (255, 255, 255)

    # ----------------------------------------------------
    # 讓點比單一 pixel 稍微粗一些
    #
    # 比逐點 cv2.circle 快很多
    # ----------------------------------------------------

    kernel = np.ones(
        (2, 2),
        dtype=np.uint8
    )

    lidar_layer = cv2.dilate(
        lidar_layer,
        kernel,
        iterations=1
    )

    # 疊到主畫面
    np.maximum(
        image,
        lidar_layer,
        out=image
    )


# ============================================================
# Draw grid
# ============================================================

def draw_grid(image):

    center = IMAGE_SIZE // 2

    scale = (
        IMAGE_SIZE
        / (METERS_RANGE * 2.0)
    )

    # 10m、20m、30m...
    for distance in range(
        10,
        int(METERS_RANGE) + 1,
        10
    ):

        radius = int(
            distance * scale
        )

        cv2.circle(
            image,
            (center, center),
            radius,
            (45, 45, 45),
            1
        )

        cv2.putText(
            image,
            f"{distance}m",
            (
                center + 5,
                center - radius + 15
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (100, 100, 100),
            1
        )

    # 十字
    cv2.line(
        image,
        (center, 0),
        (center, IMAGE_SIZE),
        (35, 35, 35),
        1
    )

    cv2.line(
        image,
        (0, center),
        (IMAGE_SIZE, center),
        (35, 35, 35),
        1
    )


# ============================================================
# Draw Blind Zone
# ============================================================

def draw_blind_zones(
    image,
    blind_results
):

    if not blind_results:
        return

    overlay = image.copy()

    for result in blind_results:

        blocker = result["blocker"]

        left_angle = (
            blocker["angle"]
            - blocker["half_width"]
        )

        right_angle = (
            blocker["angle"]
            + blocker["half_width"]
        )

        # Blind zone 從 blocker 後方開始
        start_distance = (
            blocker["distance"]
            + blocker["radius"]
        )

        end_distance = METERS_RANGE

        left_start = ego_to_pixel(
            math.cos(left_angle)
            * start_distance,

            math.sin(left_angle)
            * start_distance
        )

        left_end = ego_to_pixel(
            math.cos(left_angle)
            * end_distance,

            math.sin(left_angle)
            * end_distance
        )

        right_start = ego_to_pixel(
            math.cos(right_angle)
            * start_distance,

            math.sin(right_angle)
            * start_distance
        )

        right_end = ego_to_pixel(
            math.cos(right_angle)
            * end_distance,

            math.sin(right_angle)
            * end_distance
        )

        polygon = np.array(
            [
                left_start,
                left_end,
                right_end,
                right_start
            ],
            dtype=np.int32
        )

        cv2.fillPoly(
            overlay,
            [polygon],
            (0, 180, 255)
        )

        # Blind Zone 邊線
        cv2.line(
            image,
            left_start,
            left_end,
            (0, 255, 255),
            1
        )

        cv2.line(
            image,
            right_start,
            right_end,
            (0, 255, 255),
            1
        )

    # 半透明
    cv2.addWeighted(
        overlay,
        0.18,
        image,
        0.82,
        0,
        image
    )


# ============================================================
# Draw vehicles
# ============================================================

def draw_vehicles(
    image,
    projections,
    blind_results
):

    blind_target_ids = {
        result["target"]["id"]
        for result in blind_results
    }

    blocker_ids = {
        result["blocker"]["id"]
        for result in blind_results
    }

    for vehicle in projections:

        px, py = ego_to_pixel(
            vehicle["x"],
            vehicle["y"]
        )

        if not (
            0 <= px < IMAGE_SIZE
            and
            0 <= py < IMAGE_SIZE
        ):
            continue

        vehicle_id = vehicle["id"]

        # 預設：附近車輛
        color = (120, 120, 120)
        radius = 4

        # blocker
        if vehicle_id in blocker_ids:
            color = (0, 255, 255)
            radius = 6

        # blind target
        if vehicle_id in blind_target_ids:
            color = (0, 0, 255)
            radius = 7

        cv2.circle(
            image,
            (px, py),
            radius,
            color,
            -1
        )

        # 只標 Blind / Blocker ID，避免畫面太亂
        if (
            vehicle_id in blind_target_ids
            or
            vehicle_id in blocker_ids
        ):

            cv2.putText(
                image,
                str(vehicle_id),
                (px + 7, py - 7),

                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,

                color,
                1
            )


# ============================================================
# Draw Ego
# ============================================================

def draw_ego(image):

    center = IMAGE_SIZE // 2

    # Ego
    cv2.circle(
        image,
        (center, center),
        8,
        (255, 0, 0),
        -1
    )

    # 車頭方向
    cv2.arrowedLine(
        image,
        (center, center),

        (
            center,
            center - 35
        ),

        (255, 0, 0),
        2,

        tipLength=0.3
    )

    cv2.putText(
        image,
        "EGO",
        (
            center + 10,
            center + 5
        ),

        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,

        (255, 0, 0),
        1
    )


# ============================================================
# BEV Rendering
# ============================================================

def render_bev(
    points,
    projections,
    blind_results
):

    image = np.zeros(
        (
            IMAGE_SIZE,
            IMAGE_SIZE,
            3
        ),
        dtype=np.uint8
    )

    draw_grid(image)

    # LiDAR
    draw_lidar_points(
        image,
        points
    )

    # Blind Zone
    draw_blind_zones(
        image,
        blind_results
    )

    # Vehicle GT
    draw_vehicles(
        image,
        projections,
        blind_results
    )

    draw_ego(image)

    # --------------------------------------------------------
    # Information
    # --------------------------------------------------------

    cv2.putText(
        image,
        f"LiDAR points: {len(points)}",
        (15, 25),

        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,

        (255, 255, 255),
        1
    )

    cv2.putText(
        image,
        f"Vehicles: {len(projections)}",
        (15, 48),

        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,

        (255, 255, 255),
        1
    )

    cv2.putText(
        image,
        f"Blind targets: {len(blind_results)}",
        (15, 71),

        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,

        (0, 0, 255),
        1
    )

    cv2.putText(
        image,
        "Yellow = blocker / blind zone",
        (15, IMAGE_SIZE - 45),

        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,

        (0, 255, 255),
        1
    )

    cv2.putText(
        image,
        "Red = geometrically occluded vehicle",
        (15, IMAGE_SIZE - 22),

        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,

        (0, 0, 255),
        1
    )

    return image


# ============================================================
# Find Ego
# ============================================================

def find_ego_vehicle(world):

    vehicles = world.get_actors().filter(
        "vehicle.*"
    )

    for vehicle in vehicles:

        role_name = vehicle.attributes.get(
            "role_name",
            ""
        )

        if role_name == "hero":
            return vehicle

    return None


# ============================================================
# LiDAR creation
# ============================================================

def create_lidar(world, ego):

    blueprint_library = (
        world.get_blueprint_library()
    )

    lidar_bp = blueprint_library.find(
        "sensor.lidar.ray_cast"
    )

    lidar_bp.set_attribute(
        "range",
        str(LIDAR_RANGE)
    )

    lidar_bp.set_attribute(
        "channels",
        str(LIDAR_CHANNELS)
    )

    lidar_bp.set_attribute(
        "points_per_second",
        str(LIDAR_POINTS_PER_SECOND)
    )

    lidar_bp.set_attribute(
        "rotation_frequency",
        str(LIDAR_ROTATION_FREQUENCY)
    )

    lidar_bp.set_attribute(
        "sensor_tick",
        str(LIDAR_SENSOR_TICK)
    )

    lidar_transform = carla.Transform(
        carla.Location(
            x=0.0,
            y=0.0,
            z=LIDAR_HEIGHT
        )
    )

    lidar = world.spawn_actor(
        lidar_bp,
        lidar_transform,
        attach_to=ego
    )

    lidar.listen(
        lidar_callback
    )

    return lidar


# ============================================================
# Main
# ============================================================

def main():

    client = carla.Client(
        HOST,
        PORT
    )

    client.set_timeout(10.0)

    world = client.get_world()

    print("Connected to CARLA")
    print("Map:", world.get_map().name)

    ego = find_ego_vehicle(
        world
    )

    if ego is None:

        print(
            "找不到 role_name=hero 的 Ego Vehicle"
        )

        return

    print(
        f"Ego Vehicle: "
        f"id={ego.id}, "
        f"type={ego.type_id}"
    )

    lidar = None

    try:

        lidar = create_lidar(
            world,
            ego
        )

        print(
            f"LiDAR created: id={lidar.id}"
        )

        print()
        print(
            "V2 LiDAR + Geometry Blind Zone started"
        )
        print(
            "按 Q 或 ESC 結束"
        )
        print()

        projections = []
        blind_results = []

        last_geometry_update = 0.0

        while True:

            now = time.time()

            # =================================================
            # Geometry Update
            # =================================================

            if (
                now - last_geometry_update
                >= GEOMETRY_UPDATE_INTERVAL
            ):

                ego_transform = (
                    ego.get_transform()
                )

                vehicles = (
                    world
                    .get_actors()
                    .filter("vehicle.*")
                )

                new_projections = []

                for vehicle in vehicles:

                    if vehicle.id == ego.id:
                        continue

                    projection = (
                        get_vehicle_projection(
                            vehicle,
                            ego_transform
                        )
                    )

                    if projection is None:
                        continue

                    if (
                        projection["distance"]
                        > MAX_DISTANCE
                    ):
                        continue

                    new_projections.append(
                        projection
                    )

                projections = (
                    new_projections
                )

                blind_results = (
                    detect_blind_vehicles(
                        projections
                    )
                )

                last_geometry_update = now

            # =================================================
            # Get latest LiDAR frame
            # =================================================

            with lidar_lock:
                points = latest_points

            # =================================================
            # Render BEV
            # =================================================

            image = render_bev(
                points,
                projections,
                blind_results
            )

            cv2.imshow(
                "LiDAR Blind Zone V2",
                image
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if key == ord("q"):
                break

            if key == 27:  # ESC
                break

            # 不需要 full speed
            time.sleep(0.005)

    except KeyboardInterrupt:

        pass

    finally:

        print()
        print("Stopping...")

        if lidar is not None:

            try:
                lidar.stop()
            except RuntimeError:
                pass

            try:
                lidar.destroy()
            except RuntimeError:
                pass

        cv2.destroyAllWindows()

        print("Done")


if __name__ == "__main__":
    main()