#!/usr/bin/env python

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Generate stable CARLA traffic for cooperative-perception experiments.

Identity contract:
- Every cooperative vehicle uses CARLA role_name="hero" so Traffic Manager does
  not remove it through its stuck-vehicle cleanup path.
- Every cooperative vehicle uses a unique ros_name="car_01" ... "car_N".
- ros_name is the cooperative-perception logical vehicle identity.
- CARLA actor.id remains runtime/debug metadata only.

The built-in monitor is observation-only:
- It NEVER destroys, teleports, recycles, or respawns a managed vehicle.
- It reports missing vehicles, stopped vehicles, and long-stopped vehicles.
- If a managed vehicle disappears, treat that run as abnormal and investigate it.

This script intentionally does not implement communication-dropout injection.
"""

import argparse
import logging
import math
import time

import carla
from carla.command import DestroyActor, SpawnActor
from numpy import random


def get_actor_blueprints(world, actor_filter, generation):
    bps = world.get_blueprint_library().filter(actor_filter)

    if generation.lower() == "all":
        return bps

    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        if int_generation in [1, 2, 3]:
            return [
                x for x in bps
                if int(x.get_attribute("generation")) == int_generation
            ]
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return []
    except Exception:
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return []


def build_vehicle_spec(blueprints, vehicle_id):
    """Freeze vehicle type/color/driver for one logical CP vehicle."""
    template = random.choice(blueprints)

    spec = {
        "vehicle_id": vehicle_id,
        "blueprint_id": template.id,
        "color": None,
        "driver_id": None,
        "actor_id": None,
        "low_speed_since_sim": None,
        "stuck_warned": False,
    }

    if template.has_attribute("color"):
        values = template.get_attribute("color").recommended_values
        if values:
            spec["color"] = str(random.choice(values))

    if template.has_attribute("driver_id"):
        values = template.get_attribute("driver_id").recommended_values
        if values:
            spec["driver_id"] = str(random.choice(values))

    return spec


def make_blueprint(world, spec):
    bp = world.get_blueprint_library().find(spec["blueprint_id"])

    if not bp.has_attribute("role_name"):
        raise RuntimeError(
            "%s does not expose role_name" % spec["blueprint_id"]
        )

    if not bp.has_attribute("ros_name"):
        raise RuntimeError(
            "%s does not expose ros_name; this stable-identity mode requires it"
            % spec["blueprint_id"]
        )

    # Traffic Manager lifecycle protection.
    bp.set_attribute("role_name", "hero")

    # Cooperative-perception logical identity.
    bp.set_attribute("ros_name", spec["vehicle_id"])

    if spec["color"] is not None and bp.has_attribute("color"):
        bp.set_attribute("color", spec["color"])

    if spec["driver_id"] is not None and bp.has_attribute("driver_id"):
        bp.set_attribute("driver_id", spec["driver_id"])

    return bp


def spawn_initial_vehicle(
    world,
    traffic_manager,
    spec,
    spawn_points,
    preferred_index,
    tm_port,
    car_lights_on,
):
    """Spawn one CP participant. Initial setup only; no runtime respawn."""
    candidates = []

    if preferred_index < len(spawn_points):
        candidates.append(spawn_points[preferred_index])

    for point in spawn_points:
        if preferred_index < len(spawn_points) and point == spawn_points[preferred_index]:
            continue
        candidates.append(point)

    for transform in candidates:
        bp = make_blueprint(world, spec)
        actor = world.try_spawn_actor(bp, transform)

        if actor is None:
            continue

        actor.set_autopilot(True, tm_port)

        if car_lights_on:
            traffic_manager.update_vehicle_lights(actor, True)

        spec["actor_id"] = actor.id

        logging.info(
            "%s -> actor_id=%d | role_name=hero | ros_name=%s",
            spec["vehicle_id"],
            actor.id,
            spec["vehicle_id"],
        )
        return actor

    logging.error("Could not spawn %s", spec["vehicle_id"])
    return None


def scan_managed_vehicles(world, expected_vehicle_ids):
    """Find managed vehicles by ros_name, never by actor.id or role_name."""
    found = {}
    duplicates = {}

    for actor in world.get_actors().filter("vehicle.*"):
        vehicle_id = actor.attributes.get("ros_name", "")
        if vehicle_id not in expected_vehicle_ids:
            continue

        if vehicle_id in found:
            duplicates.setdefault(vehicle_id, [found[vehicle_id].id]).append(actor.id)
            continue

        found[vehicle_id] = actor

    for vehicle_id, actor_ids in sorted(duplicates.items()):
        logging.error(
            "[MONITOR] duplicate ros_name=%s actors=%s",
            vehicle_id,
            actor_ids,
        )

    return found, duplicates


def is_red_light_stop(actor):
    try:
        if not actor.is_at_traffic_light():
            return False
        return actor.get_traffic_light_state() == carla.TrafficLightState.Red
    except RuntimeError:
        return False


def run_vehicle_monitor(world, vehicle_specs, args):
    expected_ids = set(vehicle_specs.keys())
    current, duplicates = scan_managed_vehicles(world, expected_ids)

    snapshot = world.get_snapshot()
    sim_time = snapshot.timestamp.elapsed_seconds

    moving = 0
    stopped = 0
    transitioning = 0
    stuck_ids = []

    for vehicle_id, actor in current.items():
        spec = vehicle_specs[vehicle_id]
        actor_snapshot = snapshot.find(actor.id)

        if actor_snapshot is None:
            transitioning += 1
            continue

        velocity = actor_snapshot.get_velocity()
        speed = math.sqrt(
            velocity.x * velocity.x
            + velocity.y * velocity.y
            + velocity.z * velocity.z
        )

        spec["actor_id"] = actor.id

        if speed >= args.stopped_speed:
            moving += 1
            spec["low_speed_since_sim"] = None
            spec["stuck_warned"] = False
            continue

        stopped += 1

        if is_red_light_stop(actor):
            spec["low_speed_since_sim"] = None
            spec["stuck_warned"] = False
            continue

        if spec["low_speed_since_sim"] is None:
            spec["low_speed_since_sim"] = sim_time
            continue

        stopped_duration = sim_time - spec["low_speed_since_sim"]

        if args.stuck_warning_s > 0 and stopped_duration >= args.stuck_warning_s:
            stuck_ids.append(vehicle_id)

            if not spec["stuck_warned"]:
                logging.warning(
                    "[MONITOR] %s actor_id=%d has been stopped for %.1f s "
                    "(speed=%.2f m/s). No recovery action will be taken.",
                    vehicle_id,
                    actor.id,
                    stopped_duration,
                    speed,
                )
                spec["stuck_warned"] = True

    missing = sorted(expected_ids - set(current.keys()))

    return {
        "online": len(current),
        "expected": len(expected_ids),
        "moving": moving,
        "stopped": stopped,
        "transitioning": transitioning,
        "stuck": sorted(stuck_ids),
        "missing": missing,
        "duplicates": duplicates,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--host", metavar="H", default="140.130.35.3",
        help="IP of the CARLA server"
    )
    parser.add_argument(
        "-p", "--port", metavar="P", default=2000, type=int,
        help="CARLA RPC port (default: 2000)"
    )
    parser.add_argument(
        "-n", "--number-of-vehicles", metavar="N", default=30, type=int,
        help="Number of cooperative vehicles (default: 30)"
    )
    parser.add_argument(
        "-w", "--number-of-walkers", metavar="W", default=10, type=int,
        help="Number of walkers (default: 10)"
    )
    parser.add_argument(
        "--safe", action="store_true",
        help="Use only blueprints whose base_type is car"
    )
    parser.add_argument(
        "--filterv", metavar="PATTERN", default="vehicle.*",
        help="Vehicle blueprint filter (default: vehicle.*)"
    )
    parser.add_argument(
        "--generationv", metavar="G", default="All",
        help="Vehicle generation: 1, 2, 3 or All"
    )
    parser.add_argument(
        "--filterw", metavar="PATTERN", default="walker.pedestrian.*",
        help="Pedestrian blueprint filter"
    )
    parser.add_argument(
        "--generationw", metavar="G", default="2",
        help="Pedestrian generation: 1, 2, 3 or All"
    )
    parser.add_argument(
        "--tm-port", metavar="P", default=8000, type=int,
        help="Traffic Manager port (default: 8000)"
    )
    parser.add_argument(
        "--asynch", action="store_true",
        help="Activate asynchronous mode execution"
    )
    parser.add_argument(
        "--hybrid", action="store_true",
        help="Activate Traffic Manager hybrid physics mode (not recommended here)"
    )
    parser.add_argument(
        "-s", "--seed", metavar="S", type=int,
        help="Traffic Manager/random seed"
    )
    parser.add_argument(
        "--seedw", metavar="S", default=0, type=int,
        help="Pedestrian seed"
    )
    parser.add_argument(
        "--car-lights-on", action="store_true", default=False,
        help="Enable automatic car light management"
    )
    parser.add_argument(
        "--no-rendering", action="store_true", default=False,
        help="Activate no rendering mode"
    )
    parser.add_argument(
        "--vehicle-prefix", default="car_",
        help="CP vehicle ros_name prefix (default: car_)"
    )

    # Compatibility flags from CARLA's generate_traffic.py. They are intentionally
    # ignored in this experiment because all vehicles are heroes and this is Town10.
    parser.add_argument(
        "--hero", action="store_true", default=False,
        help="Compatibility flag; ignored because all CP vehicles are role_name=hero"
    )
    parser.add_argument(
        "--respawn", action="store_true", default=False,
        help="Compatibility flag; ignored in stable CP mode"
    )

    # Observation-only monitor.
    parser.add_argument(
        "--no-monitor", action="store_true",
        help="Disable the observation-only CP vehicle monitor"
    )
    parser.add_argument(
        "--monitor-interval", type=float, default=1.0,
        help="Seconds between registry/health checks (default: 1.0)"
    )
    parser.add_argument(
        "--status-interval", type=float, default=10.0,
        help="Seconds between status lines (default: 10.0)"
    )
    parser.add_argument(
        "--stopped-speed", type=float, default=0.3,
        help="Speed below which a vehicle is considered stopped, m/s (default: 0.3)"
    )
    parser.add_argument(
        "--stuck-warning-s", type=float, default=180.0,
        help="Warn after a non-red-light vehicle stays stopped this long; no recovery action is taken"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        format="%(levelname)s: %(message)s",
        level=logging.INFO,
    )

    if args.number_of_vehicles <= 0:
        raise ValueError("--number-of-vehicles must be > 0")

    if args.hybrid:
        logging.warning(
            "--hybrid is enabled. Stable CP mode is intended to run without hybrid physics."
        )

    if args.respawn:
        logging.warning(
            "--respawn is ignored. This experiment does not use CARLA dormant-vehicle respawn."
        )

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)

    world = None
    synchronous_master = False
    walkers_list = []
    all_id = []
    all_actors = []
    vehicle_specs = {}
    expected_vehicle_ids = set()

    random.seed(args.seed if args.seed is not None else int(time.time()))

    try:
        world = client.get_world()
        traffic_manager = client.get_trafficmanager(args.tm_port)
        traffic_manager.set_global_distance_to_leading_vehicle(2.5)

        if args.hybrid:
            traffic_manager.set_hybrid_physics_mode(True)
            traffic_manager.set_hybrid_physics_radius(70.0)

        if args.seed is not None:
            traffic_manager.set_random_device_seed(args.seed)

        settings = world.get_settings()

        if not args.asynch:
            traffic_manager.set_synchronous_mode(True)
            if not settings.synchronous_mode:
                synchronous_master = True
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 0.05
            else:
                synchronous_master = False
        else:
            print(
                "You are currently in asynchronous mode. "
                "If traffic behaves incorrectly, consider synchronous mode."
            )

        if args.no_rendering:
            settings.no_rendering_mode = True

        world.apply_settings(settings)

        blueprints = get_actor_blueprints(world, args.filterv, args.generationv)
        if not blueprints:
            raise ValueError("Couldn't find any vehicles with the specified filters")

        blueprints_walkers = get_actor_blueprints(world, args.filterw, args.generationw)
        if not blueprints_walkers:
            raise ValueError("Couldn't find any walkers with the specified filters")

        if args.safe:
            blueprints = [
                x for x in blueprints
                if x.has_attribute("base_type")
                and str(x.get_attribute("base_type")) == "car"
            ]

        identity_capable = [
            bp for bp in blueprints
            if bp.has_attribute("role_name") and bp.has_attribute("ros_name")
        ]

        skipped_count = len(blueprints) - len(identity_capable)
        if skipped_count > 0:
            logging.warning(
                "Excluded %d vehicle blueprints without role_name/ros_name support",
                skipped_count,
            )

        blueprints = sorted(identity_capable, key=lambda bp: bp.id)

        if not blueprints:
            raise RuntimeError(
                "No vehicle blueprint exposes both role_name and ros_name. "
                "Stable CP identity cannot be created."
            )

        spawn_points = list(world.get_map().get_spawn_points())
        number_of_spawn_points = len(spawn_points)

        if args.number_of_vehicles > number_of_spawn_points:
            logging.warning(
                "requested %d vehicles, but map only exposes %d spawn points",
                args.number_of_vehicles,
                number_of_spawn_points,
            )
            args.number_of_vehicles = number_of_spawn_points

        random.shuffle(spawn_points)

        role_width = max(2, len(str(args.number_of_vehicles)))

        for index in range(args.number_of_vehicles):
            vehicle_id = f"{args.vehicle_prefix}{index + 1:0{role_width}d}"
            vehicle_specs[vehicle_id] = build_vehicle_spec(blueprints, vehicle_id)

        expected_vehicle_ids = set(vehicle_specs.keys())

        # ----------------------------------------------------
        # Initial CP vehicle spawn
        # ----------------------------------------------------
        spawned_count = 0

        for index, vehicle_id in enumerate(sorted(vehicle_specs.keys())):
            actor = spawn_initial_vehicle(
                world=world,
                traffic_manager=traffic_manager,
                spec=vehicle_specs[vehicle_id],
                spawn_points=spawn_points,
                preferred_index=index,
                tm_port=args.tm_port,
                car_lights_on=args.car_lights_on,
            )
            if actor is not None:
                spawned_count += 1

        if spawned_count != args.number_of_vehicles:
            raise RuntimeError(
                "Initial CP vehicle spawn incomplete: %d / %d"
                % (spawned_count, args.number_of_vehicles)
            )

        # ----------------------------------------------------
        # Spawn walkers (CARLA example behavior)
        # ----------------------------------------------------
        percentage_pedestrians_running = 0.0
        percentage_pedestrians_crossing = 0.0

        if args.seedw:
            world.set_pedestrians_seed(args.seedw)
            random.seed(args.seedw)

        walker_spawn_points = []
        for _ in range(args.number_of_walkers):
            spawn_point = carla.Transform()
            loc = world.get_random_location_from_navigation()
            if loc is not None:
                spawn_point.location = loc
                walker_spawn_points.append(spawn_point)

        batch = []
        walker_speed = []

        for spawn_point in walker_spawn_points:
            walker_bp = random.choice(blueprints_walkers)
            probability = random.randint(0, 101)

            if walker_bp.has_attribute("is_invincible"):
                walker_bp.set_attribute("is_invincible", "false")

            if walker_bp.has_attribute("can_use_wheelchair") and probability < 11:
                walker_bp.set_attribute("use_wheelchair", "true")

            if walker_bp.has_attribute("speed"):
                if random.random() > percentage_pedestrians_running:
                    walker_speed.append(
                        walker_bp.get_attribute("speed").recommended_values[1]
                    )
                else:
                    walker_speed.append(
                        walker_bp.get_attribute("speed").recommended_values[2]
                    )
            else:
                print("Walker has no speed")
                walker_speed.append(0.0)

            batch.append(SpawnActor(walker_bp, spawn_point))

        results = client.apply_batch_sync(batch, True)
        walker_speed_ok = []

        for i, result in enumerate(results):
            if result.error:
                logging.error(result.error)
            else:
                walkers_list.append({"id": result.actor_id})
                walker_speed_ok.append(walker_speed[i])

        walker_speed = walker_speed_ok

        batch = []
        walker_controller_bp = world.get_blueprint_library().find("controller.ai.walker")

        for item in walkers_list:
            batch.append(
                SpawnActor(
                    walker_controller_bp,
                    carla.Transform(),
                    item["id"],
                )
            )

        results = client.apply_batch_sync(batch, True)
        valid_walkers = []
        valid_speeds = []

        for i, result in enumerate(results):
            if result.error:
                logging.error(result.error)
            else:
                walkers_list[i]["con"] = result.actor_id
                valid_walkers.append(walkers_list[i])
                valid_speeds.append(walker_speed[i])

        walkers_list = valid_walkers
        walker_speed = valid_speeds

        for item in walkers_list:
            all_id.append(item["con"])
            all_id.append(item["id"])

        all_actors = world.get_actors(all_id)

        # Force at least one complete world update before registry validation.
        if args.asynch or not synchronous_master:
            world.wait_for_tick()
        else:
            world.tick()

        world.set_pedestrians_cross_factor(percentage_pedestrians_crossing)

        for i in range(0, len(all_id), 2):
            controller = all_actors[i]
            controller.start()
            controller.go_to_location(world.get_random_location_from_navigation())
            controller.set_max_speed(float(walker_speed[int(i / 2)]))

        # Give actor attributes one more completed world update before the first
        # registry report. This avoids the stale startup registry observed with
        # immediate post-spawn validation.
        if args.asynch or not synchronous_master:
            world.wait_for_tick()
        else:
            world.tick()

        connected, duplicates = scan_managed_vehicles(world, expected_vehicle_ids)
        hero_count = sum(
            1
            for actor in connected.values()
            if actor.attributes.get("role_name", "") == "hero"
        )

        print("\n===== Cooperative Vehicle Registry =====")
        for vehicle_id in sorted(connected.keys()):
            actor = connected[vehicle_id]
            print(
                "%-8s actor_id=%-5d role_name=%-4s %s"
                % (
                    vehicle_id,
                    actor.id,
                    actor.attributes.get("role_name", ""),
                    actor.type_id,
                )
            )

        missing_initial = sorted(expected_vehicle_ids - set(connected.keys()))

        print("CP vehicles: %d / %d" % (len(connected), len(expected_vehicle_ids)))
        print("Unique ros_name: %d" % len(connected))
        print("role_name=hero: %d / %d" % (hero_count, len(expected_vehicle_ids)))
        print("Missing ros_name: %d" % len(missing_initial))
        print("Duplicate ros_name: %d" % len(duplicates))

        if missing_initial or duplicates or hero_count != len(expected_vehicle_ids):
            raise RuntimeError("Initial cooperative vehicle registry validation failed")

        print(
            "spawned %d stable cooperative vehicles and %d walkers, press Ctrl+C to exit."
            % (len(connected), len(walkers_list))
        )

        traffic_manager.global_percentage_speed_difference(30.0)

        if args.no_monitor:
            print("CP vehicle monitor: DISABLED")
        else:
            print(
                "CP vehicle monitor: OBSERVATION ONLY "
                "(check=%.1fs, stuck_warning=%.1fs, stopped_speed=%.2fm/s)"
                % (
                    args.monitor_interval,
                    args.stuck_warning_s,
                    args.stopped_speed,
                )
            )
            print("Monitor policy: no destroy, no teleport, no recycle, no respawn.")

        last_monitor_wall = 0.0
        last_status_wall = 0.0
        last_status = None
        previous_missing = set()

        while True:
            if not args.asynch and synchronous_master:
                world.tick()
            else:
                world.wait_for_tick()

            if args.no_monitor:
                continue

            now_wall = time.monotonic()

            if now_wall - last_monitor_wall >= args.monitor_interval:
                last_monitor_wall = now_wall
                last_status = run_vehicle_monitor(world, vehicle_specs, args)

                current_missing = set(last_status["missing"])
                newly_missing = sorted(current_missing - previous_missing)

                for vehicle_id in newly_missing:
                    logging.error(
                        "[MONITOR] %s disappeared from CARLA. "
                        "No automatic respawn will be performed.",
                        vehicle_id,
                    )

                previous_missing = current_missing

            if (
                last_status is not None
                and now_wall - last_status_wall >= args.status_interval
            ):
                last_status_wall = now_wall

                missing_text = (
                    ",".join(last_status["missing"])
                    if last_status["missing"] else "-"
                )
                stuck_text = (
                    ",".join(last_status["stuck"])
                    if last_status["stuck"] else "-"
                )

                logging.info(
                    "[MONITOR] online=%d/%d | moving=%d | stopped=%d | "
                    "transitioning=%d | stuck=%s | missing=%s",
                    last_status["online"],
                    last_status["expected"],
                    last_status["moving"],
                    last_status["stopped"],
                    last_status["transitioning"],
                    stuck_text,
                    missing_text,
                )

    finally:
        if world is not None:
            if not args.asynch and synchronous_master:
                settings = world.get_settings()
                settings.synchronous_mode = False
                settings.no_rendering_mode = False
                settings.fixed_delta_seconds = None
                world.apply_settings(settings)

            for item in walkers_list:
                controller_id = item.get("con")
                if controller_id is None:
                    continue
                controller = world.get_actor(controller_id)
                if controller is not None:
                    try:
                        controller.stop()
                    except RuntimeError:
                        pass

            print("\ndestroying managed cooperative vehicles")
            current, _ = scan_managed_vehicles(world, expected_vehicle_ids)
            current_vehicle_ids = [actor.id for actor in current.values()]

            if current_vehicle_ids:
                client.apply_batch(
                    [DestroyActor(actor_id) for actor_id in current_vehicle_ids]
                )

            print("\ndestroying %d walkers" % len(walkers_list))
            if all_id:
                client.apply_batch([DestroyActor(actor_id) for actor_id in all_id])

            time.sleep(0.5)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print("\ndone.")
