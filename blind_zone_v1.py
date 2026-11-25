import carla
import math
import time


# =========================================================
# Settings
# =========================================================

HOST = "localhost"
PORT = 2000

MAX_DISTANCE = 50.0

# 遠方目標有多少比例被前車角度遮住，就算 Blind
OCCLUSION_THRESHOLD = 0.60

# 避免兩台幾乎並排時被判成前後遮蔽
MIN_DISTANCE_GAP = 2.0

# Debug Visualization
DRAW_DEBUG = True

# 每隔多久印一次結果
PRINT_INTERVAL = 0.5


# =========================================================
# Basic Geometry
# =========================================================

def wrap_angle(angle):
    """
    將角度限制在 [-pi, pi]
    """
    return math.atan2(math.sin(angle), math.cos(angle))


def angle_diff(a, b):
    """
    a - b 的最短角度差
    """
    return wrap_angle(a - b)


def world_to_ego(point, ego_transform):
    """
    World coordinate -> Ego 2D coordinate

    Ego:
        +X = forward
        +Y = side
    """

    ego_loc = ego_transform.location
    yaw = math.radians(ego_transform.rotation.yaw)

    dx = point.x - ego_loc.x
    dy = point.y - ego_loc.y

    # inverse rotation
    x_local = math.cos(yaw) * dx + math.sin(yaw) * dy
    y_local = -math.sin(yaw) * dx + math.cos(yaw) * dy

    return x_local, y_local


# =========================================================
# Vehicle Projection
# =========================================================

def get_vehicle_projection(actor, ego_transform):
    """
    將一台車投影到 Ego 視角。

    回傳：
        distance
        center_angle
        angular_half_width
        center_x
        center_y
    """

    location = actor.get_location()

    center_x, center_y = world_to_ego(
        location,
        ego_transform
    )

    distance = math.hypot(center_x, center_y)

    if distance < 0.01:
        return None

    center_angle = math.atan2(center_y, center_x)

    # -----------------------------------------------------
    # 使用 CARLA Bounding Box 的 8 個角
    # -----------------------------------------------------

    bbox = actor.bounding_box
    actor_transform = actor.get_transform()

    vertices = bbox.get_world_vertices(actor_transform)

    angular_offsets = []

    for vertex in vertices:

        x, y = world_to_ego(
            vertex,
            ego_transform
        )

        # 非常接近 Ego 時避免奇怪結果
        if abs(x) < 0.001 and abs(y) < 0.001:
            continue

        corner_angle = math.atan2(y, x)

        offset = angle_diff(
            corner_angle,
            center_angle
        )

        angular_offsets.append(abs(offset))

    if len(angular_offsets) == 0:
        return None

    half_width = max(angular_offsets)

    return {
        "actor": actor,

        "id": actor.id,

        "type": actor.type_id,

        "location": location,

        "distance": distance,

        "angle": center_angle,

        "half_width": half_width,

        "x": center_x,
        "y": center_y,
    }


# =========================================================
# Angular Occlusion
# =========================================================

def angular_overlap_ratio(target, blocker):
    """
    計算 blocker 在 Ego 視角下，
    覆蓋 target 的比例。

    例如：

        target angular width
        ----------------

              blocker
              --------

    回傳 0.0 ~ 1.0
    """

    target_half = target["half_width"]
    blocker_half = blocker["half_width"]

    # 將 blocker 的中心角轉成相對 target 的角度
    d = angle_diff(
        blocker["angle"],
        target["angle"]
    )

    # target interval:
    #
    # [-target_half, +target_half]

    target_left = -target_half
    target_right = target_half

    # blocker interval
    blocker_left = d - blocker_half
    blocker_right = d + blocker_half

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

    target_width = 2.0 * target_half

    if target_width < 1e-6:
        return 0.0

    return overlap / target_width


# =========================================================
# Blind Zone Detection
# =========================================================

def detect_blind_vehicles(projections):
    """
    回傳所有被其他車遮蔽的 vehicle。

    結果：

    [
        {
            "target": ...,
            "blocker": ...,
            "overlap": ...
        }
    ]
    """

    results = []

    for target in projections:

        best_blocker = None
        best_overlap = 0.0

        for blocker in projections:

            if blocker["id"] == target["id"]:
                continue

            # ---------------------------------------------
            # blocker 必須比 target 更靠近 Ego
            # ---------------------------------------------

            if blocker["distance"] >= (
                target["distance"] - MIN_DISTANCE_GAP
            ):
                continue

            overlap = angular_overlap_ratio(
                target,
                blocker
            )

            if overlap > best_overlap:

                best_overlap = overlap
                best_blocker = blocker

        # ---------------------------------------------
        # Blind 判定
        # ---------------------------------------------

        if (
            best_blocker is not None
            and best_overlap >= OCCLUSION_THRESHOLD
        ):

            results.append({
                "target": target,
                "blocker": best_blocker,
                "overlap": best_overlap
            })

    return results


# =========================================================
# Debug Visualization
# =========================================================

def draw_blind_zone(world, ego_transform, blocker, target):
    """
    用兩條線畫出 blocker 後方的 Blind Zone。
    """

    if not DRAW_DEBUG:
        return

    debug = world.debug

    ego_loc = ego_transform.location

    z = ego_loc.z + 1.5

    center_angle_world = (
        math.radians(ego_transform.rotation.yaw)
        + blocker["angle"]
    )

    left_angle = (
        center_angle_world
        - blocker["half_width"]
    )

    right_angle = (
        center_angle_world
        + blocker["half_width"]
    )

    start_distance = blocker["distance"] + 1.0
    end_distance = MAX_DISTANCE

    # Blind zone 左右邊界
    left_start = carla.Location(
        x=ego_loc.x
        + math.cos(left_angle) * start_distance,

        y=ego_loc.y
        + math.sin(left_angle) * start_distance,

        z=z
    )

    left_end = carla.Location(
        x=ego_loc.x
        + math.cos(left_angle) * end_distance,

        y=ego_loc.y
        + math.sin(left_angle) * end_distance,

        z=z
    )

    right_start = carla.Location(
        x=ego_loc.x
        + math.cos(right_angle) * start_distance,

        y=ego_loc.y
        + math.sin(right_angle) * start_distance,

        z=z
    )

    right_end = carla.Location(
        x=ego_loc.x
        + math.cos(right_angle) * end_distance,

        y=ego_loc.y
        + math.sin(right_angle) * end_distance,

        z=z
    )

    # 黃色：Blind Zone 邊界
    debug.draw_line(
        left_start,
        left_end,
        thickness=0.08,
        color=carla.Color(255, 255, 0),
        life_time=0.1
    )

    debug.draw_line(
        right_start,
        right_end,
        thickness=0.08,
        color=carla.Color(255, 255, 0),
        life_time=0.1
    )

    # blocker = 黃色點
    blocker_loc = blocker["location"]

    blocker_point = carla.Location(
        x=blocker_loc.x,
        y=blocker_loc.y,
        z=blocker_loc.z + 2.0
    )

    debug.draw_point(
        blocker_point,
        size=0.15,
        color=carla.Color(255, 255, 0),
        life_time=0.1
    )

    # target = 紅色點
    target_loc = target["location"]

    target_point = carla.Location(
        x=target_loc.x,
        y=target_loc.y,
        z=target_loc.z + 2.0
    )

    debug.draw_point(
        target_point,
        size=0.20,
        color=carla.Color(255, 0, 0),
        life_time=0.1
    )

    debug.draw_string(
        target_point,
        f"BLIND {target['id']}",
        draw_shadow=True,
        color=carla.Color(255, 0, 0),
        life_time=0.1
    )


# =========================================================
# Ego Search
# =========================================================

def find_ego_vehicle(world):

    vehicles = world.get_actors().filter("vehicle.*")

    for vehicle in vehicles:

        role_name = vehicle.attributes.get(
            "role_name",
            ""
        )

        if role_name == "hero":
            return vehicle

    return None


# =========================================================
# Main
# =========================================================

def main():

    client = carla.Client(
        HOST,
        PORT
    )

    client.set_timeout(10.0)

    world = client.get_world()

    print("Connected to CARLA")
    print("Map:", world.get_map().name)

    ego = find_ego_vehicle(world)

    if ego is None:
        print("找不到 role_name=hero 的 Ego Vehicle")
        return

    print(
        f"Ego Vehicle: "
        f"id={ego.id}, "
        f"type={ego.type_id}"
    )

    print()
    print("Blind Zone Detection V1 started")
    print(
        f"MAX_DISTANCE = {MAX_DISTANCE} m"
    )
    print(
        f"OCCLUSION_THRESHOLD = "
        f"{OCCLUSION_THRESHOLD}"
    )
    print()

    last_print = 0.0

    try:

        while True:

            world.wait_for_tick()

            ego_transform = ego.get_transform()

            vehicles = world.get_actors().filter(
                "vehicle.*"
            )

            projections = []

            # =============================================
            # 建立所有車輛的 Ego 視角投影
            # =============================================

            for vehicle in vehicles:

                if vehicle.id == ego.id:
                    continue

                projection = get_vehicle_projection(
                    vehicle,
                    ego_transform
                )

                if projection is None:
                    continue

                if (
                    projection["distance"]
                    > MAX_DISTANCE
                ):
                    continue

                projections.append(
                    projection
                )

            # =============================================
            # Blind Zone Detection
            # =============================================

            blind_results = detect_blind_vehicles(
                projections
            )

            # =============================================
            # Debug Visualization
            # =============================================

            for result in blind_results:

                draw_blind_zone(
                    world,
                    ego_transform,
                    result["blocker"],
                    result["target"]
                )

            # =============================================
            # Console
            # =============================================

            now = time.time()

            if now - last_print >= PRINT_INTERVAL:

                print(
                    f"\rNearby={len(projections):2d} "
                    f"Blind={len(blind_results):2d}",
                    end=""
                )

                if blind_results:

                    print()

                    for result in blind_results:

                        target = result["target"]
                        blocker = result["blocker"]

                        print(
                            f"  [BLIND] "
                            f"target={target['id']} "
                            f"({target['distance']:.1f}m) "
                            f"<- blocker={blocker['id']} "
                            f"({blocker['distance']:.1f}m) "
                            f"overlap="
                            f"{result['overlap']:.2f}"
                        )

                last_print = now

    except KeyboardInterrupt:

        print()
        print("Blind Zone Detection stopped")


if __name__ == "__main__":
    main()