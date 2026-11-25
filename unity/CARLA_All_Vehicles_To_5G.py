"""Send nearby CARLA traffic to the existing 5G -> MEC -> Unity UDP path.

Run on the Windows PC running CARLA 0.9.16 with Python 3.10:
    py -3.10 CARLA_All_Vehicles_To_5G.py --list
    py -3.10 CARLA_All_Vehicles_To_5G.py <EGO_ACTOR_ID>

Read-only: does NOT spawn, destroy, tick, or control CARLA actors.
Keep ego_vehicle.py and generate_traffic.py as they are.

Every datagram uses the same JSON schema as the existing Unity receiver:
    {"vehicles":[{"id":"car_01", "x":..., "y":..., "yaw":..., "speed_kmh":...}, ...]}
Nearby cars are given stable ids 'traffic_<CARLA actor id>'.
The reserved car_02 maps to one nearby car for backward compatibility with Audi_Other.
To avoid UDP/IP fragmentation through the 5G path, large snapshots are split
into several independent JSON datagrams, each <= MAX_UDP_BYTES.
"""

import json
import math
import socket
import sys
import time

import carla

CARLA_HOST = "127.0.0.1"
CARLA_PORT = 2000
DEST_IP = "26.129.236.197"     # User's Windows Radmin VPN address
DEST_PORT = 50100             # Existing carla_5g_router.py input
SEND_HZ = 10.0
NEARBY_RADIUS_M = 120.0       # All vehicles within 120m of Ego (2D distance)
MAX_UDP_BYTES = 1100          # Stay below typical tunnel MTU after headers
REQUIRED_MAP = "Town10HD_Opt" # Imported Unity road map


def car_data(actor, actor_id_label, transform=None):
    t = transform if transform is not None else actor.get_transform()
    v = actor.get_velocity()
    return {
        "id": actor_id_label,
        "x": round(t.location.x, 3),
        "y": round(t.location.y, 3),
        "yaw": round(t.rotation.yaw, 2),
        "speed_kmh": round(math.sqrt(v.x*v.x + v.y*v.y + v.z*v.z) * 3.6, 1),
    }


def make_packets(records):
    """Produce fully valid packets <= MAX_UDP_BYTES; never truncate vehicles."""
    out = []
    batch = []
    for record in records:
        candidate = batch + [record]
        payload = json.dumps({"vehicles": candidate}, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        if len(payload) <= MAX_UDP_BYTES:
            batch = candidate
            continue
        if not batch:
            raise ValueError("Single vehicle record exceeds UDP packet limit")
        out.append(json.dumps({"vehicles": batch}, separators=(",", ":")).encode("utf-8"))
        batch = [record]
        size = len(json.dumps({"vehicles": batch}, separators=(",", ":")).encode("utf-8"))
        if size > MAX_UDP_BYTES:
            raise ValueError("Single vehicle record exceeds UDP packet limit")
    if batch:
        out.append(json.dumps({"vehicles": batch}, separators=(",", ":")).encode("utf-8"))
    return out


def list_vehicles(world):
    cars = world.get_actors().filter("vehicle.*")
    print("MAP:", world.get_map().name, "TOTAL VEHICLES:", len(cars))
    print("ACTOR_ID  ROLE        MODEL")
    for actor in cars:
        role = actor.attributes.get("role_name", "")
        print("%-9s %-11s %s" % (actor.id, role, actor.type_id))
    print("Find the Ego actor ID using ego_vehicle.py output (role_name=hero is NOT always unique).")


def main():
    if len(sys.argv) != 2:
        print("Usage: py -3.10 CARLA_All_Vehicles_To_5G.py --list")
        print("       py -3.10 CARLA_All_Vehicles_To_5G.py <EGO_ACTOR_ID>")
        return 2
    list_only = sys.argv[1] == "--list"
    if not list_only:
        try:
            ego_id = int(sys.argv[1])
        except ValueError:
            print("ERROR: Ego Actor ID must be a number, or use --list.")
            return 2

    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(8.0)
    try:
        world = client.get_world()
        map_name = world.get_map().name
    except Exception as exc:
        print("ERROR: Cannot connect to CARLA %s:%s: %s" % (CARLA_HOST, CARLA_PORT, exc))
        return 1
    if list_only:
        list_vehicles(world)
        return 0
    if not map_name.endswith(REQUIRED_MAP):
        print("ERROR: Current CARLA map is %s, but Unity roads are %s." % (map_name, REQUIRED_MAP))
        print("Do not mix maps; ask the CARLA operator to verify the scenario.")
        return 1

    ego = world.get_actor(ego_id)
    if ego is None or not ego.type_id.startswith("vehicle."):
        print("ERROR: Ego Actor ID %s was not found. Run with --list." % ego_id)
        return 1
    if ego.attributes.get("role_name") != "hero":
        print("WARNING: Ego ID %s has role_name=%r (verify ID)." %
              (ego_id, ego.attributes.get("role_name", "")))

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    period = 1.0 / SEND_HZ
    report_at = 0.0
    preferred_other_id = None
    sent_frames = 0
    print("CARLA MULTI-VEHICLE -> 5G STARTED (READ-ONLY)")
    print("MAP:", map_name, "EGO:", ego_id, ego.type_id)
    print("ALL VEHICLES WITHIN %.0fm, %.0f Hz, UDP packets <= %d bytes" %
          (NEARBY_RADIUS_M, SEND_HZ, MAX_UDP_BYTES))
    print("DESTINATION: %s:%d" % (DEST_IP, DEST_PORT))
    print("Keep your existing Windows router / UE bridge / MEC bridge / Unity running.")
    print("Ctrl+C stops ONLY this sender; CARLA vehicles continue moving.")

    try:
        while True:
            began = time.monotonic()
            ego = world.get_actor(ego_id)
            if ego is None:
                print("ERROR: Ego actor %s disappeared. Run --list for the new actor ID." % ego_id)
                break

            ego_transform = ego.get_transform()
            ep = ego_transform.location
            radius_sq = NEARBY_RADIUS_M * NEARBY_RADIUS_M
            near = []
            for actor in world.get_actors().filter("vehicle.*"):
                if actor.id == ego_id:
                    continue
                t = actor.get_transform()
                dx = t.location.x - ep.x
                dy = t.location.y - ep.y
                d2 = dx * dx + dy * dy
                if d2 <= radius_sq:
                    near.append((d2, actor, t))
            near.sort(key=lambda item: item[0])

            # Preserve 'car_02' for the existing Audi_Other Unity transform.
            # Prefer an Audi TT, then the nearest non-hero vehicle.
            other = next((item for item in near if item[1].id == preferred_other_id), None)
            if other is None:
                other = next((item for item in near if item[1].type_id == "vehicle.audi.tt"), None)
                if other is None:
                    other = next((item for item in near if item[1].attributes.get("role_name", "") != "hero"), None)
                if other is None and near:
                    other = near[0]
                preferred_other_id = other[1].id if other is not None else None

            records = [car_data(ego, "car_01", ego_transform)]
            if other is not None:
                records.append(car_data(other[1], "car_02", other[2]))
            for _, actor, transform in near:
                if other is not None and actor.id == other[1].id:
                    continue
                records.append(car_data(actor, "traffic_%d" % actor.id, transform))

            datagrams = make_packets(records)
            for payload in datagrams:
                sock.sendto(payload, (DEST_IP, DEST_PORT))
            sent_frames += 1
            if began - report_at >= 1.0:
                report_at = began
                print("SENT frame=%d  EGO=%d  nearby=%d  total_visible=%d  UDP_parts=%d  max_bytes=%d  speed=%.1f km/h" %
                      (sent_frames, ego_id, len(near), len(records), len(datagrams),
                       max(map(len, datagrams)), records[0]["speed_kmh"]), flush=True)
            time.sleep(max(0.0, period - (time.monotonic() - began)))
    except KeyboardInterrupt:
        print("\nSender stopped; CARLA traffic is unchanged.")
    except Exception as exc:
        print("ERROR: %s: %s" % (type(exc).__name__, exc))
        return 1
    finally:
        sock.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
