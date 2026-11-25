import carla
import numpy as np
import pygame
import threading
import math
import time
from collections import deque


# ============================================================
# 顯示設定
# ============================================================

WIDTH = 800
HEIGHT = 800

# 顯示 Ego 周圍 ±50 m
VIEW_RANGE = 50.0

# 保留最近幾幀 LiDAR
HISTORY_SIZE = 5

# 顯示 FPS
DISPLAY_FPS = 30


# ============================================================
# LiDAR 設定
# ============================================================

LIDAR_CHANNELS = 32
LIDAR_RANGE = 50.0
LIDAR_POINTS_PER_SECOND = 100000
LIDAR_ROTATION_FREQUENCY = 10.0
LIDAR_UPPER_FOV = 10.0
LIDAR_LOWER_FOV = -30.0
LIDAR_HEIGHT = 2.4

# LiDAR 顯示高度過濾
MIN_Z = -2.0
MAX_Z = 2.0


# ============================================================
# Blind Zone 幾何判定設定
# ============================================================

# 最遠只處理這個範圍內的車輛
MAX_DETECTION_DISTANCE = 50.0

# Target 的視角寬度被 Blocker 蓋住多少比例，就判定為 Blind
OCCLUSION_THRESHOLD = 0.60

# 避免兩台幾乎並排行駛時被判成前後遮蔽
MIN_DISTANCE_GAP = 2.0

# 幾何判定更新間隔（秒）
GEOMETRY_UPDATE_INTERVAL = 0.10


# ============================================================
# 顏色
# ============================================================

COLOR_BG = (0, 0, 0)
COLOR_GRID = (45, 45, 45)
COLOR_AXIS = (40, 40, 40)

COLOR_EGO = (0, 255, 0)

COLOR_NORMAL_VEHICLE = (120, 120, 120)
COLOR_BLOCKER = (255, 220, 0)
COLOR_BLIND_TARGET = (255, 60, 60)

COLOR_BLIND_FILL = (180, 140, 0, 55)
COLOR_BLIND_EDGE = (255, 220, 0)


# ============================================================
# LiDAR frame history
# ============================================================

point_history = deque(maxlen=HISTORY_SIZE)
points_lock = threading.Lock()


# ============================================================
# LiDAR callback
# ============================================================

def lidar_callback(data):

    points = np.frombuffer(
        data.raw_data,
        dtype=np.float32
    ).reshape((-1, 4))

    xyz = points[:, :3].copy()

    with points_lock:
        point_history.append(xyz)


# ============================================================
# 找 Ego Vehicle
# ============================================================

def find_ego_vehicle(world):

    vehicles = world.get_actors().filter("vehicle.*")

    for vehicle in vehicles:

        role_name = vehicle.attributes.get(
            "role_name",
            ""
        )

        if role_name in ["hero", "ego"]:

            print(
                f"找到 Ego Vehicle: "
                f"id={vehicle.id}, "
                f"type={vehicle.type_id}, "
                f"role_name={role_name}"
            )

            return vehicle

    return None


# ============================================================
# LiDAR / Ego 座標 → pygame 座標
# ============================================================

def ego_to_screen(x, y):

    """
    Ego local coordinates:

        +X = 車頭方向
        +Y = 右方

    Bird View:

            Front (+X)
                ↑
                |
        -Y ←  EGO  → +Y
    """

    scale = WIDTH / (VIEW_RANGE * 2.0)

    screen_x = WIDTH // 2 + int(y * scale)
    screen_y = HEIGHT // 2 - int(x * scale)

    return screen_x, screen_y


# ============================================================
# 角度工具
# ============================================================

def wrap_angle(angle):

    return math.atan2(
        math.sin(angle),
        math.cos(angle)
    )


def angle_diff(a, b):

    return wrap_angle(a - b)


# ============================================================
# World → Ego local coordinate
# ============================================================

def world_to_ego(point, ego_transform):

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
# Vehicle → Ego angular projection
# ============================================================

def get_vehicle_projection(actor, ego_transform):

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

            vertex_angle = math.atan2(
                vy,
                vx
            )

            offset = angle_diff(
                vertex_angle,
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
        # Actor 可能剛好被 CARLA 移除
        return None


# ============================================================
# Angular overlap
# ============================================================

def angular_overlap_ratio(target, blocker):

    target_half = target["half_width"]
    blocker_half = blocker["half_width"]

    relative_angle = angle_diff(
        blocker["angle"],
        target["angle"]
    )

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

    target_width = 2.0 * target_half

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

            # Blocker 必須比 Target 更接近 Ego
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
# 建立附近車輛 projection
# ============================================================

def build_vehicle_projections(world, ego_vehicle):

    ego_transform = ego_vehicle.get_transform()

    vehicles = world.get_actors().filter(
        "vehicle.*"
    )

    projections = []

    for vehicle in vehicles:

        if vehicle.id == ego_vehicle.id:
            continue

        projection = get_vehicle_projection(
            vehicle,
            ego_transform
        )

        if projection is None:
            continue

        if (
            projection["distance"]
            > MAX_DETECTION_DISTANCE
        ):
            continue

        projections.append(
            projection
        )

    return projections


# ============================================================
# 畫背景 / Grid
# ============================================================

def draw_background(screen):

    screen.fill(
        COLOR_BG
    )

    center_x = WIDTH // 2
    center_y = HEIGHT // 2

    scale = WIDTH / (
        VIEW_RANGE * 2.0
    )

    for distance in [
        10,
        20,
        30,
        40,
        50
    ]:

        if distance > VIEW_RANGE:
            continue

        radius = int(
            distance * scale
        )

        pygame.draw.circle(
            screen,
            COLOR_GRID,
            (center_x, center_y),
            radius,
            1
        )

    pygame.draw.line(
        screen,
        COLOR_AXIS,
        (center_x, 0),
        (center_x, HEIGHT)
    )

    pygame.draw.line(
        screen,
        COLOR_AXIS,
        (0, center_y),
        (WIDTH, center_y)
    )


# ============================================================
# 畫 Blind Zone
# ============================================================

def draw_blind_zones(screen, blind_results):

    if not blind_results:
        return

    overlay = pygame.Surface(
        (WIDTH, HEIGHT),
        pygame.SRCALPHA
    )

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

        # Blind Zone 從 blocker 後方開始
        start_distance = (
            blocker["distance"]
            + blocker["radius"]
        )

        end_distance = VIEW_RANGE

        left_start = ego_to_screen(
            math.cos(left_angle)
            * start_distance,
            math.sin(left_angle)
            * start_distance
        )

        left_end = ego_to_screen(
            math.cos(left_angle)
            * end_distance,
            math.sin(left_angle)
            * end_distance
        )

        right_start = ego_to_screen(
            math.cos(right_angle)
            * start_distance,
            math.sin(right_angle)
            * start_distance
        )

        right_end = ego_to_screen(
            math.cos(right_angle)
            * end_distance,
            math.sin(right_angle)
            * end_distance
        )

        polygon = [
            left_start,
            left_end,
            right_end,
            right_start
        ]

        pygame.draw.polygon(
            overlay,
            COLOR_BLIND_FILL,
            polygon
        )

        pygame.draw.line(
            screen,
            COLOR_BLIND_EDGE,
            left_start,
            left_end,
            1
        )

        pygame.draw.line(
            screen,
            COLOR_BLIND_EDGE,
            right_start,
            right_end,
            1
        )

    screen.blit(
        overlay,
        (0, 0)
    )


# ============================================================
# 畫 LiDAR history
# ============================================================

def draw_lidar_history(screen, frames):

    total_points = 0

    if len(frames) == 0:
        return total_points

    brightness_levels = np.linspace(
        50,
        255,
        len(frames)
    )

    for frame_index, points in enumerate(frames):

        brightness = int(
            brightness_levels[frame_index]
        )

        color = (
            brightness,
            brightness,
            brightness
        )

        mask = (
            (points[:, 2] > MIN_Z)
            &
            (points[:, 2] < MAX_Z)
        )

        filtered = points[mask]

        total_points += len(filtered)

        for point in filtered:

            x = point[0]
            y = point[1]

            if (
                abs(x) > VIEW_RANGE
                or
                abs(y) > VIEW_RANGE
            ):
                continue

            sx, sy = ego_to_screen(
                x,
                y
            )

            if (
                0 <= sx < WIDTH
                and
                0 <= sy < HEIGHT
            ):

                pygame.draw.circle(
                    screen,
                    color,
                    (sx, sy),
                    1
                )

    return total_points


# ============================================================
# 畫 Ground Truth Vehicles
# ============================================================

def draw_vehicle_markers(
    screen,
    projections,
    blind_results,
    font_small
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

        sx, sy = ego_to_screen(
            vehicle["x"],
            vehicle["y"]
        )

        if not (
            0 <= sx < WIDTH
            and
            0 <= sy < HEIGHT
        ):
            continue

        vehicle_id = vehicle["id"]

        color = COLOR_NORMAL_VEHICLE
        radius = 3
        show_label = False

        if vehicle_id in blocker_ids:
            color = COLOR_BLOCKER
            radius = 6
            show_label = True

        if vehicle_id in blind_target_ids:
            color = COLOR_BLIND_TARGET
            radius = 7
            show_label = True

        pygame.draw.circle(
            screen,
            color,
            (sx, sy),
            radius,
            2
        )

        if show_label:

            label = font_small.render(
                str(vehicle_id),
                True,
                color
            )

            screen.blit(
                label,
                (sx + 8, sy - 8)
            )


# ============================================================
# 畫 Ego
# ============================================================

def draw_ego(screen):

    center_x = WIDTH // 2
    center_y = HEIGHT // 2

    ego_width = 18
    ego_length = 36

    ego_rect = pygame.Rect(
        center_x - ego_width // 2,
        center_y - ego_length // 2,
        ego_width,
        ego_length
    )

    pygame.draw.rect(
        screen,
        COLOR_EGO,
        ego_rect,
        2
    )

    pygame.draw.line(
        screen,
        COLOR_EGO,
        (center_x, center_y),
        (
            center_x,
            center_y - 55
        ),
        3
    )


# ============================================================
# UI
# ============================================================

def draw_ui(
    screen,
    font,
    frames,
    projections,
    blind_results
):

    latest_count = 0

    if len(frames) > 0:
        latest_count = len(
            frames[-1]
        )

    texts = [
        (
            f"Latest LiDAR: {latest_count} points",
            (255, 255, 255)
        ),
        (
            f"History: {len(frames)}/{HISTORY_SIZE} frames",
            (255, 255, 255)
        ),
        (
            f"Vehicles: {len(projections)}",
            (255, 255, 255)
        ),
        (
            f"Blind targets: {len(blind_results)}",
            COLOR_BLIND_TARGET
        ),
        (
            f"Occlusion threshold: {OCCLUSION_THRESHOLD:.2f}",
            (255, 255, 255)
        )
    ]

    y = 10

    for text, color in texts:

        surface = font.render(
            text,
            True,
            color
        )

        screen.blit(
            surface,
            (10, y)
        )

        y += 24

    front = font.render(
        "FRONT",
        True,
        COLOR_EGO
    )

    screen.blit(
        front,
        (
            WIDTH // 2 - 25,
            10
        )
    )

    legend_y = HEIGHT - 58

    legend1 = font.render(
        "Yellow: blocker / blind zone",
        True,
        COLOR_BLOCKER
    )

    legend2 = font.render(
        "Red: geometrically blind target",
        True,
        COLOR_BLIND_TARGET
    )

    screen.blit(
        legend1,
        (10, legend_y)
    )

    screen.blit(
        legend2,
        (10, legend_y + 24)
    )


# ============================================================
# 建立 LiDAR
# ============================================================

def create_lidar(world, ego_vehicle):

    bp_lib = world.get_blueprint_library()

    lidar_bp = bp_lib.find(
        "sensor.lidar.ray_cast"
    )

    lidar_bp.set_attribute(
        "channels",
        str(LIDAR_CHANNELS)
    )

    lidar_bp.set_attribute(
        "range",
        str(LIDAR_RANGE)
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
        "upper_fov",
        str(LIDAR_UPPER_FOV)
    )

    lidar_bp.set_attribute(
        "lower_fov",
        str(LIDAR_LOWER_FOV)
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
        attach_to=ego_vehicle
    )

    lidar.listen(
        lidar_callback
    )

    return lidar


# ============================================================
# 主程式
# ============================================================

def main():

    client = carla.Client(
        "140.130.35.3",
        2000
    )

    client.set_timeout(10.0)

    world = client.get_world()

    print("Connected to CARLA")
    print("Map:", world.get_map().name)

    # --------------------------------------------------------
    # Ego
    # --------------------------------------------------------

    ego_vehicle = find_ego_vehicle(
        world
    )

    if ego_vehicle is None:
        print("找不到 Ego Vehicle")
        return

    # --------------------------------------------------------
    # LiDAR
    # --------------------------------------------------------

    lidar = None

    try:

        lidar = create_lidar(
            world,
            ego_vehicle
        )

        print(
            f"LiDAR 建立成功: id={lidar.id}"
        )

        print(
            "Blind Zone Detection started"
        )

        print(
            "按 ESC 關閉"
        )

        # ====================================================
        # pygame
        # ====================================================

        pygame.init()

        screen = pygame.display.set_mode(
            (WIDTH, HEIGHT)
        )

        pygame.display.set_caption(
            "CARLA Blind Zone Detection"
        )

        clock = pygame.time.Clock()

        font = pygame.font.SysFont(
            "Arial",
            18
        )

        font_small = pygame.font.SysFont(
            "Arial",
            14
        )

        # ====================================================
        # Geometry cache
        # ====================================================

        projections = []
        blind_results = []

        last_geometry_update = 0.0

        running = True

        while running:

            # ------------------------------------------------
            # Event
            # ------------------------------------------------

            for event in pygame.event.get():

                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:

                    if event.key == pygame.K_ESCAPE:
                        running = False

            # ------------------------------------------------
            # Update geometry
            # ------------------------------------------------

            now = time.time()

            if (
                now - last_geometry_update
                >= GEOMETRY_UPDATE_INTERVAL
            ):

                projections = (
                    build_vehicle_projections(
                        world,
                        ego_vehicle
                    )
                )

                blind_results = (
                    detect_blind_vehicles(
                        projections
                    )
                )

                last_geometry_update = now

            # ------------------------------------------------
            # Copy LiDAR history
            # ------------------------------------------------

            with points_lock:
                frames = list(
                    point_history
                )

            # ------------------------------------------------
            # Render order
            # ------------------------------------------------
            #
            # 1. Background / Grid
            # 2. Blind Zone
            # 3. LiDAR history
            # 4. Vehicle GT markers
            # 5. Ego
            # 6. UI
            #
            # 這樣 Blind Zone 是底層，
            # 不會把 LiDAR 點雲蓋掉。
            # ------------------------------------------------

            draw_background(
                screen
            )

            draw_blind_zones(
                screen,
                blind_results
            )

            draw_lidar_history(
                screen,
                frames
            )

            draw_vehicle_markers(
                screen,
                projections,
                blind_results,
                font_small
            )

            draw_ego(
                screen
            )

            draw_ui(
                screen,
                font,
                frames,
                projections,
                blind_results
            )

            pygame.display.flip()

            clock.tick(
                DISPLAY_FPS
            )

    except KeyboardInterrupt:

        pass

    finally:

        print("停止 LiDAR...")

        if lidar is not None:

            try:
                lidar.stop()
            except RuntimeError:
                pass

            try:
                lidar.destroy()
            except RuntimeError:
                pass

        pygame.quit()

        print("完成")


if __name__ == "__main__":
    main()
