import carla
import numpy as np
import pygame
import threading
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

# LiDAR frame history
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
# LiDAR 座標 → pygame 座標
# ============================================================

def lidar_to_screen(x, y):

    """
    CARLA LiDAR local coordinates:

        +X = 車頭方向
        +Y = 右方
        +Z = 上方

    Bird View:

            Front (+X)
                ↑
                |
        -Y ←  EGO  → +Y
    """

    scale = WIDTH / (VIEW_RANGE * 2)

    screen_x = WIDTH // 2 + int(y * scale)
    screen_y = HEIGHT // 2 - int(x * scale)

    return screen_x, screen_y


# ============================================================
# 主程式
# ============================================================

def main():

    # --------------------------------------------------------
    # CARLA
    # --------------------------------------------------------

    client = carla.Client(
        "localhost",
        2000
    )

    client.set_timeout(10.0)

    world = client.get_world()

    print("Connected to CARLA")
    print("Map:", world.get_map().name)

    # --------------------------------------------------------
    # 找 Ego
    # --------------------------------------------------------

    ego_vehicle = find_ego_vehicle(world)

    if ego_vehicle is None:
        print("找不到 Ego Vehicle")
        return

    # --------------------------------------------------------
    # LiDAR Blueprint
    # --------------------------------------------------------

    bp_lib = world.get_blueprint_library()

    lidar_bp = bp_lib.find(
        "sensor.lidar.ray_cast"
    )

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

    # --------------------------------------------------------
    # LiDAR 安裝位置
    # --------------------------------------------------------

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

    lidar.listen(
        lidar_callback
    )

    # ========================================================
    # pygame
    # ========================================================

    pygame.init()

    screen = pygame.display.set_mode(
        (WIDTH, HEIGHT)
    )

    pygame.display.set_caption(
        "CARLA Ego LiDAR Bird View"
    )

    clock = pygame.time.Clock()

    font = pygame.font.Font(None,18)

    running = True

    try:

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
            # Background
            # ------------------------------------------------

            screen.fill(
                (0, 0, 0)
            )

            center_x = WIDTH // 2
            center_y = HEIGHT // 2

            scale = WIDTH / (
                VIEW_RANGE * 2
            )

            # ------------------------------------------------
            # 距離圓
            # ------------------------------------------------

            for distance in [
                10,
                20,
                30,
                40,
                50
            ]:

                radius = int(
                    distance * scale
                )

                pygame.draw.circle(
                    screen,
                    (45, 45, 45),
                    (center_x, center_y),
                    radius,
                    1
                )

            # ------------------------------------------------
            # 中心座標軸
            # ------------------------------------------------

            pygame.draw.line(
                screen,
                (40, 40, 40),
                (center_x, 0),
                (center_x, HEIGHT)
            )

            pygame.draw.line(
                screen,
                (40, 40, 40),
                (0, center_y),
                (WIDTH, center_y)
            )

            # =================================================
            # 取得最近幾幀
            # =================================================

            with points_lock:
                frames = list(point_history)

            total_points = 0

            # -------------------------------------------------
            # 每一幀不同亮度
            #
            # oldest → darkest
            # newest → brightest
            # -------------------------------------------------

            if len(frames) > 0:

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

                    # -----------------------------------------
                    # 高度過濾
                    #
                    # LiDAR 高約 2.4 m
                    # 地面大約會在 z ≈ -2.4
                    #
                    # 所以暫時排除大部分地面
                    # -----------------------------------------

                    mask = (
                        (points[:, 2] > -2.0)
                        &
                        (points[:, 2] < 2.0)
                    )

                    filtered = points[mask]

                    total_points += len(filtered)

                    # -----------------------------------------
                    # 畫 LiDAR point
                    # -----------------------------------------

                    for point in filtered:

                        x = point[0]
                        y = point[1]

                        if (
                            abs(x) > VIEW_RANGE
                            or
                            abs(y) > VIEW_RANGE
                        ):
                            continue

                        sx, sy = lidar_to_screen(
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

            # =================================================
            # Ego Vehicle
            # =================================================

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
                (0, 255, 0),
                ego_rect,
                2
            )

            # ------------------------------------------------
            # 車頭方向
            # ------------------------------------------------

            pygame.draw.line(
                screen,
                (0, 255, 0),
                (center_x, center_y),
                (
                    center_x,
                    center_y - 55
                ),
                3
            )

            # =================================================
            # UI
            # =================================================

            latest_count = 0

            if len(frames) > 0:
                latest_count = len(
                    frames[-1]
                )

            text1 = font.render(
                f"Latest LiDAR: {latest_count} points",
                True,
                (255, 255, 255)
            )

            text2 = font.render(
                f"History: {len(frames)}/{HISTORY_SIZE} frames",
                True,
                (255, 255, 255)
            )

            text3 = font.render(
                f"View: +/- {VIEW_RANGE:.0f} m",
                True,
                (255, 255, 255)
            )

            text4 = font.render(
                "FRONT",
                True,
                (0, 255, 0)
            )

            screen.blit(
                text1,
                (10, 10)
            )

            screen.blit(
                text2,
                (10, 35)
            )

            screen.blit(
                text3,
                (10, 60)
            )

            screen.blit(
                text4,
                (
                    center_x - 25,
                    10
                )
            )

            pygame.display.flip()

            # pygame 顯示 30 FPS
            clock.tick(30)

    finally:

        print("停止 LiDAR...")

        lidar.stop()
        lidar.destroy()

        pygame.quit()

        print("完成")


if __name__ == "__main__":
    main()