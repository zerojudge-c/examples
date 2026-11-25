"""Export CARLA driving-lane geometry for a read-only Unity BEV road map.

CARLA 0.9.16, Python 3.10. Run on the Windows host that runs CARLA:
    py -3.10 CARLA_Export_Roads.py

This script never creates, moves, or destroys CARLA actors, and never ticks the world.
Output: CARLA_Roads.json in the same directory as this script.
"""

import json
import math
from collections import defaultdict
from pathlib import Path

import carla

HOST = '127.0.0.1'
PORT = 2000
WAYPOINT_SPACING_M = 2.0
MAX_GAP_M = 6.5
OUTPUT_PATH = Path(__file__).with_name('CARLA_Roads.json')


def mark_type(marking):
    if marking is None:
        return 'None'
    return str(marking.type).split('.')[-1]


def point(w):
    t = w.transform
    p = t.location
    return {
        'x': round(float(p.x), 3),
        'y': round(float(p.y), 3),
        'z': round(float(p.z), 3),
        'yaw': round(float(t.rotation.yaw), 2),
        'width': round(float(w.lane_width), 3),
        'junction': bool(w.is_junction),
        'left_mark': mark_type(w.left_lane_marking),
        'right_mark': mark_type(w.right_lane_marking),
    }


def main():
    print(f'Connecting to CARLA {HOST}:{PORT} (read-only)...', flush=True)
    client = carla.Client(HOST, PORT)
    client.set_timeout(15.0)
    carla_map = client.get_world().get_map()
    print('Map:', carla_map.name, flush=True)

    groups = defaultdict(list)
    all_points = carla_map.generate_waypoints(WAYPOINT_SPACING_M)
    for w in all_points:
        if w.lane_type != carla.LaneType.Driving:
            continue
        groups[(int(w.road_id), int(w.section_id), int(w.lane_id))].append(w)

    lanes = []
    for (road_id, section_id, lane_id), waypoints in sorted(groups.items()):
        waypoints.sort(key=lambda w: float(w.s))
        fragments = []
        current = []
        previous = None
        for w in waypoints:
            if previous is not None:
                a = previous.transform.location
                b = w.transform.location
                gap = math.hypot(float(b.x - a.x), float(b.y - a.y))
                if gap > MAX_GAP_M:
                    if len(current) >= 2:
                        fragments.append(current)
                    current = []
            current.append(point(w))
            previous = w
        if len(current) >= 2:
            fragments.append(current)
        for fragment_index, fragment in enumerate(fragments):
            lanes.append({
                'road_id': road_id,
                'section_id': section_id,
                'lane_id': lane_id,
                'fragment': fragment_index,
                'points': fragment,
            })

    payload = {
        'format': 'carla_unity_bev_lanes_v1',
        'map': carla_map.name,
        'carla_coordinates': 'x_forward_y_right_z_up; world meters; raw absolute CARLA coordinates',
        'spacing_m': WAYPOINT_SPACING_M,
        'lane_count': len(lanes),
        'lanes': lanes,
    }
    with OUTPUT_PATH.open('w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, separators=(',', ':'))

    print(f'Export complete: {OUTPUT_PATH}', flush=True)
    print(f'Driving lane fragments: {len(lanes)}', flush=True)
    print(f'File size: {OUTPUT_PATH.stat().st_size / 1024 / 1024:.2f} MB', flush=True)
    print('Only road geometry was read; all CARLA vehicles are unchanged.', flush=True)


if __name__ == '__main__':
    main()
