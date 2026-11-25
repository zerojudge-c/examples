#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CARLA 0.9.16 - controlled rear-end chase V4.

Uses one existing Ego and one existing NPC. Both are staged on a verified
straight driving-lane segment. During the approach, autopilot is disabled,
steering is held at zero, and both vehicles receive velocity commands along
one shared fixed world-space heading.

Unlike the previous V3 lock scripts, the chase loop does NOT repeatedly
project the moving vehicles back onto OpenDRIVE waypoints. This avoids false
lane-change aborts caused by waypoint direction/road-segment projection.
A collision is reported only when CARLA's collision sensor confirms contact.

This is a controlled simulation scenario, not a spontaneous traffic accident.
"""

import argparse
import csv
import json
import math
import random
import time
import threading
from datetime import datetime
from pathlib import Path

import carla

STATE = Path(__file__).with_name("carla_rear_chase_v4_state.json")
OLD_STATES = (
    Path(__file__).with_name("carla_forced_rear_demo_state.json"),
    Path(__file__).with_name("carla_natural_rear_state.json"),
)


def planar(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def kmh(actor):
    v = actor.get_velocity()
    return 3.6 * math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)


def yaw_diff(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def freeze(actor):
    if actor and actor.is_alive:
        actor.apply_control(carla.VehicleControl(
            throttle=0.0, steer=0.0, brake=1.0, hand_brake=True
        ))
        actor.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
        actor.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
        actor.set_simulate_physics(False)


def restore_autopilot(actor, tm_port):
    if actor and actor.is_alive:
        actor.set_simulate_physics(True)
        actor.apply_control(carla.VehicleControl(
            throttle=0.0, steer=0.0, brake=0.0, hand_brake=False
        ))
        actor.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
        actor.set_autopilot(True, tm_port)


def resume_saved(world, ego_id, tm_port):
    state_file = STATE if STATE.exists() else next(
        (p for p in OLD_STATES if p.exists()), None
    )
    if state_file is None:
        print("No saved V4 crash state. Nothing to resume.")
        return 0

    data = json.loads(state_file.read_text(encoding="utf-8"))
    if data.get("map") != world.get_map().name:
        print("State belongs to another map; refusing to resume.")
        return 2
    if data.get("ego") != ego_id:
        print("Saved Ego ID does not match --ego-id; refusing to resume.")
        return 2

    for key in ("ego", "npc"):
        actor_id = data.get(key)
        actor = world.get_actor(actor_id) if isinstance(actor_id, int) else None
        if actor and actor.is_alive:
            restore_autopilot(actor, tm_port)
            print("Restored", key, "actor", actor_id)

    state_file.unlink(missing_ok=True)
    print("Resume complete.")
    return 0


def straight_section(mp, entry, length_m, step=2.0):
    """Return a connected, nearly straight driving-lane waypoint sequence."""
    if (entry is None or entry.is_junction or
            entry.lane_type != carla.LaneType.Driving):
        return None

    result = [entry]
    current = entry
    base_yaw = entry.transform.rotation.yaw

    for _ in range(int(math.ceil(length_m / step))):
        nxt = current.next(step)
        if len(nxt) != 1:
            return None

        current = nxt[0]
        if (current.is_junction or
                current.lane_type != carla.LaneType.Driving or
                yaw_diff(current.transform.rotation.yaw, base_yaw) > 3.0 or
                yaw_diff(current.transform.rotation.yaw,
                         result[-1].transform.rotation.yaw) > 2.0 or
                planar(current.transform.location,
                       result[-1].transform.location) > step + 1.0):
            return None

        result.append(current)

    return result


def choose_staging(world, ego, npc, gap_m, search_length=55.0):
    """
    Find an empty straight lane segment and return two poses on that same
    connected lane. Other actors are not moved or modified.
    """
    mp = world.get_map()
    waypoints = list(mp.generate_waypoints(5.0))
    random.shuffle(waypoints)

    other_actors = []
    for actor in world.get_actors():
        if actor.id in (ego.id, npc.id) or not actor.is_alive:
            continue
        if not (actor.type_id.startswith("vehicle.") or
                actor.type_id.startswith("walker.pedestrian.")):
            continue
        try:
            loc = actor.get_location()
            extent = (actor.bounding_box.extent.x
                      if actor.type_id.startswith("vehicle.") else 0.6)
            other_actors.append((loc, extent))
        except RuntimeError:
            continue

    step = 2.0
    target_index = max(1, int(round(gap_m / step)))

    for entry in waypoints:
        section = straight_section(
            mp, entry, max(search_length, gap_m + 30.0), step
        )
        if section is None or target_index >= len(section):
            continue

        forward = entry.transform.get_forward_vector()
        origin = entry.transform.location
        lane_width = float(entry.lane_width)
        corridor_half_width = max(1.9, min(2.4, lane_width * 0.58))
        clear = True

        for other_pos, extent in other_actors:
            if abs(other_pos.z - origin.z) >= 3.0:
                continue
            dx = other_pos.x - origin.x
            dy = other_pos.y - origin.y
            along = dx * forward.x + dy * forward.y
            lateral = abs(-dx * forward.y + dy * forward.x)
            if (-5.0 <= along <= gap_m + 30.0 and
                    lateral <= corridor_half_width + min(extent, 0.5)):
                clear = False
                break

        if not clear:
            continue

        ego_wp = section[0]
        npc_wp = section[target_index]
        actual_gap = planar(ego_wp.transform.location,
                            npc_wp.transform.location)
        if actual_gap < gap_m - 2.5:
            continue

        ego_tf = ego_wp.transform
        npc_loc = npc_wp.transform.location
        # Give both vehicles exactly the same yaw. The NPC is ahead in the
        # connected waypoint direction, not placed using an unrelated spawn.
        rotation = carla.Rotation(
            pitch=ego_tf.rotation.pitch,
            yaw=ego_tf.rotation.yaw,
            roll=ego_tf.rotation.roll
        )
        ego_pose = carla.Transform(
            carla.Location(ego_tf.location.x, ego_tf.location.y,
                           ego_tf.location.z + 0.45),
            rotation
        )
        npc_pose = carla.Transform(
            carla.Location(npc_loc.x, npc_loc.y, npc_loc.z + 0.45),
            rotation
        )
        return ego_pose, npc_pose, ego_wp.road_id, ego_wp.lane_id, actual_gap

    return None


def set_fixed_motion(actor, forward, speed_kmh, fixed_yaw,
                     allow_yaw_correction=True):
    """
    Keep steering at zero and command velocity along the shared fixed heading.
    Small orientation drift is corrected before the cars reach contact range.
    """
    actor.apply_control(carla.VehicleControl(
        throttle=0.0, steer=0.0, brake=0.0, hand_brake=False
    ))

    tf = actor.get_transform()
    error = yaw_diff(tf.rotation.yaw, fixed_yaw)

    if allow_yaw_correction and error > 0.8:
        # Correct orientation only; preserve the current world position.
        actor.set_transform(carla.Transform(
            tf.location,
            carla.Rotation(pitch=tf.rotation.pitch,
                           yaw=fixed_yaw, roll=0.0)
        ))

    actor.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
    speed = speed_kmh / 3.6
    actor.set_target_velocity(carla.Vector3D(
        forward.x * speed, forward.y * speed, 0.0
    ))
    return error


def relative_geometry(ego_loc, npc_loc, forward):
    dx = npc_loc.x - ego_loc.x
    dy = npc_loc.y - ego_loc.y
    along = dx * forward.x + dy * forward.y
    lateral = abs(-dx * forward.y + dy * forward.x)
    return along, lateral


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ego-id", required=True, type=int)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=2000, type=int)
    parser.add_argument("--tm-port", default=8000, type=int)
    parser.add_argument("--delay", default=10, type=int)
    parser.add_argument("--gap", default=20.0, type=float,
                        help="Initial center-to-center gap in meters")
    parser.add_argument("--ego-kmh", default=28.0, type=float)
    parser.add_argument("--npc-kmh", default=8.0, type=float)
    parser.add_argument("--lead-in", default=1.0, type=float,
                        help="Seconds both cars move at the same speed first")
    parser.add_argument("--cruise-kmh", default=8.0, type=float)
    parser.add_argument("--timeout", default=20.0, type=float)
    parser.add_argument("--max-lateral", default=2.4, type=float)
    parser.add_argument("--diagnose", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not (0.0 <= args.delay <= 120 and
            10.0 <= args.gap <= 35.0 and
            5.0 <= args.npc_kmh < args.ego_kmh <= 50.0 and
            0.0 <= args.lead_in <= 8.0 and
            5.0 <= args.cruise_kmh <= 20.0 and
            4.0 <= args.timeout <= 60.0 and
            1.5 <= args.max_lateral <= 4.0):
        parser.error("Invalid settings: check gap, speeds, delay, lead-in, timeout.")

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()
    print("MAP:", world.get_map().name,
          "| asynchronous world is not ticked or reconfigured", flush=True)

    if args.resume:
        return resume_saved(world, args.ego_id, args.tm_port)

    if STATE.exists():
        print("A previous V4 crash is frozen. Run this script with --resume first.")
        return 2
    old_state = next((p for p in OLD_STATES if p.exists()), None)
    if old_state is not None:
        print("A previous V3/older crash state exists:", old_state.name)
        print("Run this script with --resume once before starting a new test.")
        return 2

    ego = world.get_actor(args.ego_id)
    if ego is None or not ego.is_alive or not ego.type_id.startswith("vehicle."):
        print("ERROR: Ego ID not found. Check the current Ego actor ID.")
        return 2

    if args.delay:
        print("STARTING IN %d SECONDS. No vehicles are staged yet." % args.delay,
              flush=True)
        for remaining in range(args.delay, 0, -1):
            print("  %d..." % remaining, flush=True)
            time.sleep(1.0)

    vehicles = list(world.get_actors().filter("vehicle.*"))
    candidates = [
        actor for actor in vehicles
        if actor.id != ego.id and actor.is_alive
        and actor.attributes.get("role_name", "") not in
        ("hero", "ego_moving_crash_target")
        and actor.bounding_box.extent.x >= 1.3
    ]
    if not candidates:
        print("ERROR: No existing NPC vehicle found. Keep traffic generation running.")
        return 2

    random.shuffle(candidates)
    npc = candidates[0]

    staging = choose_staging(world, ego, npc, args.gap)
    if staging is None:
        print("ABORT: Could not find a clear straight driving-lane segment.")
        print("Other traffic was not moved. Try again after traffic changes.")
        return 1

    ego_pose, npc_pose, road_id, lane_id, actual_gap = staging
    yaw = ego_pose.rotation.yaw
    radians = math.radians(yaw)
    forward = carla.Vector3D(math.cos(radians), math.sin(radians), 0.0)

    print("EXISTING EGO:", ego.id, "| EXISTING NPC:", npc.id, flush=True)
    print("STAGING: road=%s lane=%s initial_gap=%.1fm" %
          (road_id, lane_id, actual_gap), flush=True)
    print("V4 FIXED-HEADING CHASE: Ego %.1f km/h, NPC %.1f km/h" %
          (args.ego_kmh, args.npc_kmh), flush=True)
    print("No autopilot, no lane-waypoint checks during chase, no world.tick().",
          flush=True)

    sensor = None
    collided = threading.Event()
    armed = threading.Event()
    event_info = {}
    rows = []
    frozen = False
    start = time.monotonic()
    sim_start = None
    csv_path = Path(__file__).with_name(
        "rear_chase_v4_%s.csv" % datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    try:
        # Take manual control and stage both existing actors.
        ego.set_autopilot(False, args.tm_port)
        npc.set_autopilot(False, args.tm_port)
        ego.set_simulate_physics(True)
        npc.set_simulate_physics(True)

        for actor, pose in ((ego, ego_pose), (npc, npc_pose)):
            actor.apply_control(carla.VehicleControl(
                throttle=0.0, steer=0.0, brake=1.0, hand_brake=False
            ))
            actor.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
            actor.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
            actor.set_transform(pose)
            actor.apply_control(carla.VehicleControl(
                throttle=0.0, steer=0.0, brake=0.0, hand_brake=False
            ))

        collision_bp = world.get_blueprint_library().find(
            "sensor.other.collision"
        )
        sensor = world.spawn_actor(
            collision_bp, carla.Transform(), attach_to=ego
        )

        def on_collision(event):
            other = event.other_actor
            if (armed.is_set() and other is not None and
                    other.id == npc.id and not collided.is_set()):
                event_info.update(
                    frame=int(event.frame),
                    sim_time=float(event.timestamp),
                    wall_time=time.monotonic() - start
                )
                collided.set()

        sensor.listen(on_collision)
        armed.set()
        sim_start = world.get_snapshot().timestamp.elapsed_seconds
        print("CHASE LOOP STARTED. Waiting for verified sensor contact.", flush=True)

        last_print = -1
        first_near = None
        contact_guard = (
            float(ego.bounding_box.extent.x) +
            float(npc.bounding_box.extent.x) + 0.35
        )

        while time.monotonic() - start < args.timeout and not collided.is_set():
            if not ego.is_alive or not npc.is_alive:
                print("ABORT: Ego or NPC disappeared.", flush=True)
                break

            ego_loc = ego.get_location()
            npc_loc = npc.get_location()
            along, lateral = relative_geometry(ego_loc, npc_loc, forward)

            if abs(ego_loc.z - npc_loc.z) > 3.0:
                print("ABORT: Vehicles are no longer at the same elevation.",
                      flush=True)
                break
            if along <= 0.0:
                print("ABORT: NPC is no longer ahead of Ego.", flush=True)
                break
            if lateral > args.max_lateral:
                print("ABORT: Relative lateral separation exceeded %.2fm." %
                      args.max_lateral, flush=True)
                break

            elapsed = time.monotonic() - start
            if elapsed < args.lead_in:
                ego_speed = npc_speed = args.cruise_kmh
                phase = "LEAD-IN"
            else:
                ego_speed = args.ego_kmh
                npc_speed = args.npc_kmh
                phase = "CHASE"

            # Stop orientation corrections once the vehicle bodies approach
            # contact range, so the actual collision physics can resolve contact.
            allow_correction = along > contact_guard
            ego_yaw_error = set_fixed_motion(
                ego, forward, ego_speed, yaw, allow_correction
            )
            npc_yaw_error = set_fixed_motion(
                npc, forward, npc_speed, yaw, allow_correction
            )

            sim_elapsed = (
                world.get_snapshot().timestamp.elapsed_seconds - sim_start
            )
            center_distance = planar(ego_loc, npc_loc)

            if along < 9.0 and first_near is None:
                first_near = elapsed
                print("APPROACH: longitudinal_gap=%.2fm lateral=%.2fm" %
                      (along, lateral), flush=True)

            if int(elapsed) != last_print:
                print(
                    "wall=%.1fs sim=%.1fs %s | gap=%.2fm lateral=%.2fm "
                    "| Ego=%.1f NPC=%.1f km/h" %
                    (elapsed, sim_elapsed, phase, along, lateral,
                     kmh(ego), kmh(npc)),
                    flush=True
                )
                if args.diagnose:
                    print(
                        "  fixed_yaw=%.1f | yaw_error E/N=%.2f/%.2f deg "
                        "| contact_guard=%.2fm" %
                        (yaw, ego_yaw_error, npc_yaw_error, contact_guard),
                        flush=True
                    )
                last_print = int(elapsed)

            rows.append([
                round(elapsed, 3), round(sim_elapsed, 3), phase,
                round(along, 3), round(lateral, 3),
                round(kmh(ego), 2), round(kmh(npc), 2), 0
            ])
            time.sleep(0.02)

        if collided.is_set():
            collision_sim = event_info["sim_time"] - sim_start
            print(
                "CONFIRMED CONTACT: Ego %d -> NPC %d | wall=%.2fs sim=%.2fs" %
                (ego.id, npc.id, event_info["wall_time"], collision_sim),
                flush=True
            )
            rows.append([
                round(event_info["wall_time"], 3), round(collision_sim, 3),
                "CONTACT", 0.0, 0.0, round(kmh(ego), 2),
                round(kmh(npc), 2), 1
            ])
            freeze(ego)
            freeze(npc)
            STATE.write_text(json.dumps({
                "map": world.get_map().name,
                "ego": ego.id,
                "npc": npc.id
            }), encoding="utf-8")
            frozen = True
            print("Both test vehicles frozen. Other traffic remains active.",
                  flush=True)
            if first_near is not None:
                print("Time from near-approach to contact: %.2fs" %
                      (event_info["wall_time"] - first_near), flush=True)
            return 0

        print("NO VERIFIED CONTACT within %.1fs. Restoring autopilot." %
              args.timeout, flush=True)
        return 1

    except (RuntimeError, KeyboardInterrupt) as exc:
        print("STOP:", exc, flush=True)
        return 3

    finally:
        armed.clear()
        if sensor is not None:
            try:
                sensor.stop()
                sensor.destroy()
            except RuntimeError:
                pass

        if not frozen:
            for actor in (ego, npc):
                try:
                    restore_autopilot(actor, args.tm_port)
                except RuntimeError:
                    pass

        with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "elapsed_wall_s", "elapsed_sim_s", "phase",
                "longitudinal_gap_m", "lateral_gap_m",
                "ego_kmh", "npc_kmh", "confirmed_contact"
            ])
            writer.writerows(rows)
        print("CSV SAVED:", csv_path, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
