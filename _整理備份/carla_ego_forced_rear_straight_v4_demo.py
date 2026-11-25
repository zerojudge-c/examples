#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CARLA 0.9.16: straight-lane, moving-vs-moving Ego rear-crash demo V4 lane-route tracking (moving lead-in and dual clocks).

Uses existing Ego + one existing NPC in the active city. BOTH actors are moved
onto an EMPTY, verified DRIVING lane and begin moving immediately; there is no
10-second stationary countdown. Other traffic, map and CARLA clock are intact.

This is a CONTROLLED scenario, not a naturally occurring traffic crash.

Run:
 py -3.10 carla_ego_forced_rear_straight_v4_demo.py --ego-id 240
 py -3.10 carla_ego_forced_rear_straight_v4_demo.py --ego-id 240 --resume
"""
import argparse
import csv
import json
import math
import random
import threading
import time
from datetime import datetime
from pathlib import Path

import carla

STATE = Path(__file__).with_name('carla_forced_rear_demo_state.json')
OLD_STATE = Path(__file__).with_name('carla_natural_rear_state.json')


def planar(a, b):
    return math.hypot(a.x-b.x, a.y-b.y)


def kmh(car):
    v=car.get_velocity()
    return 3.6*math.sqrt(v.x*v.x+v.y*v.y+v.z*v.z)


def yaw_difference(a,b):
    return abs((a-b+180)%360-180)


def freeze(car):
    car.apply_control(carla.VehicleControl(throttle=0, brake=1, hand_brake=True))
    car.set_target_velocity(carla.Vector3D(0,0,0))
    car.set_target_angular_velocity(carla.Vector3D(0,0,0))
    car.set_simulate_physics(False)


def return_to_autopilot(car, port):
    if car and car.is_alive:
        car.set_simulate_physics(True)
        car.apply_control(carla.VehicleControl(throttle=0,brake=0,hand_brake=False))
        car.set_target_angular_velocity(carla.Vector3D(0,0,0))
        car.set_autopilot(True,port)


def resume(world, port):
    # Accept state left by either forced or the older natural-rear program.
    state_file = STATE if STATE.exists() else OLD_STATE if OLD_STATE.exists() else None
    if state_file is None:
        print('No saved crash state. Nothing to resume.')
        return 0
    data=json.loads(state_file.read_text(encoding='utf-8'))
    if data.get('map') != world.get_map().name:
        print('State is for a different map: refusing to change any vehicle.')
        return 2
    if data.get('ego') != getattr(resume, 'expected_ego_id', None):
        print('State Ego ID differs from requested Ego; refusing to resume old actors.')
        return 2
    for identity in ('ego','npc'):
        actor_id=data.get(identity)
        actor=world.get_actor(actor_id) if isinstance(actor_id,int) else None
        if actor:
            return_to_autopilot(actor,port)
            print('Resumed',identity,'actor',actor_id)
    state_file.unlink(missing_ok=True)
    return 0


def safe_lane_section(mp, entry, length=40, step=2.0):
    """Verify a single nearly straight driving lane with no intersections."""
    if entry is None or entry.is_junction or entry.lane_type != carla.LaneType.Driving:
        return None
    result=[entry]
    cur=entry
    for _ in range(int(length/step)):
        nxt=cur.next(step)
        if len(nxt)!=1:
            return None
        cur=nxt[0]
        # OpenDRIVE road_id/lane_id can change at a segment boundary even when
        # the actual driving lane continues straight.  Follow wp.next() (the
        # connected lane topology) rather than demanding constant numeric IDs.
        if (cur.is_junction or cur.lane_type != carla.LaneType.Driving or
                yaw_difference(cur.transform.rotation.yaw,
                               entry.transform.rotation.yaw) > 8 or
                yaw_difference(cur.transform.rotation.yaw,
                               result[-1].transform.rotation.yaw) > 4 or
                planar(cur.transform.location, result[-1].transform.location) > step + 1.3):
            return None
        result.append(cur)
    return result


def choose_site(world, exclude_ids, ego_length, npc_length, gap, tries=500):
    mp=world.get_map()
    origins=list(mp.get_spawn_points())
    random.shuffle(origins)
    # Traffic on current map may occupy many spawn points; also try mapped waypoints.
    waypoints=[]
    for tf in origins:
        wp=mp.get_waypoint(tf.location,project_to_road=True,lane_type=carla.LaneType.Driving)
        if wp is not None:
            waypoints.append(wp)
    # Add extra true driving waypoints even when plenty of spawn points exist:
    # otherwise a city with 50 NPCs can leave almost every spawn point occupied.
    extras=[wp for wp in mp.generate_waypoints(10.0)
            if wp.lane_type == carla.LaneType.Driving and not wp.is_junction]
    random.shuffle(extras)
    waypoints += extras
    random.shuffle(waypoints)  # mix route waypoints with the spawn points
    actors=[]
    for a in world.get_actors():
        if a.id in exclude_ids or not a.is_alive:
            continue
        if not (a.type_id.startswith('vehicle.') or a.type_id.startswith('walker.pedestrian.')):
            continue
        try:
            p=a.get_location()
            actors.append((p, a.bounding_box.extent.x if a.type_id.startswith('vehicle.') else 0.6))
        except RuntimeError:
            continue

    for entry in waypoints[:tries]:
        # Give the cars enough straight road to close the gap before a junction.
        section=safe_lane_section(mp,entry, length=gap+55)
        if section is None:
            continue
        # Use more than just two points: keep the entire driving corridor clear.
        clear=True
        for wp in section:
            loc=wp.transform.location
            for other_pos,extent in actors:
                if abs(other_pos.z-loc.z)<3.0 and planar(other_pos,loc)<max(6.0,extent+4.0):
                    clear=False
                    break
            if not clear:
                break
        if not clear:
            continue
        ego_wp=section[0]
        target_wp=section[round(gap/2)]
        if planar(ego_wp.transform.location,target_wp.transform.location)<gap-2:
            continue
        et=ego_wp.transform
        nt=target_wp.transform
        e_pose=carla.Transform(carla.Location(et.location.x,et.location.y,et.location.z+0.45),et.rotation)
        n_pose=carla.Transform(carla.Location(nt.location.x,nt.location.y,nt.location.z+0.45),nt.rotation)
        return e_pose,n_pose,ego_wp.road_id,ego_wp.lane_id,section
    return None



def route_waypoint(section, position, last_index):
    """Find the nearest waypoint on the *staged forward route*, not a nearby
    opposite-direction lane that map.get_waypoint(project_to_road=True) might
    select at a segment boundary or median. Search forward with some tolerance.
    """
    lo=max(0, last_index-2)
    hi=min(len(section), last_index+18)
    candidates=[]
    for idx in range(lo, hi):
        loc=section[idx].transform.location
        if abs(position.z-loc.z)>2.6:
            continue
        candidates.append((planar(position,loc),idx))
    if not candidates:
        return None,last_index,float('inf')
    dist,idx=min(candidates)
    return section[idx],max(last_index,idx),dist


def drive_along_waypoint(car, wp, target_speed):
    """Keep a car aligned with lane center, not merely pushed forward by velocity.

    CARLA's old steering angle can survive a teleport. Explicit steering correction
    toward the road's heading + centerline avoids the diagonal approach.
    """
    pose=car.get_transform()
    car_yaw=pose.rotation.yaw
    lane_yaw=wp.transform.rotation.yaw
    error_yaw=((lane_yaw-car_yaw+180.0)%360.0)-180.0
    here=pose.location
    center=wp.transform.location
    rad=math.radians(car_yaw)
    lateral_error=-(center.x-here.x)*math.sin(rad)+(center.y-here.y)*math.cos(rad)
    # Both terms are right-positive in CARLA local steering convention.
    steer=max(-0.35,min(0.35, 0.025*error_yaw+0.09*lateral_error))
    car.apply_control(carla.VehicleControl(throttle=0.0,steer=steer,
                                           brake=0.0,hand_brake=False))
    # Accelerate along the vehicle's actual heading, not the map tangent.
    # Sending map-aligned velocity every frame can force sideways sliding.
    direction=pose.get_forward_vector()
    car.set_target_velocity(carla.Vector3D(direction.x*target_speed/3.6,
                                           direction.y*target_speed/3.6, 0))
    return error_yaw,lateral_error


def same_straight_approach(ego_wp, npc_wp, ego_pos, npc_pos):
    """Check geometry across OpenDRIVE road segment ID changes.

    True when the NPC remains ahead, both waypoints run in nearly the same
    direction and their lateral separation is lane-sized.  Unlike equality
    of (road_id, lane_id), this works across connected straight road segments.
    """
    forward=ego_wp.transform.get_forward_vector()
    dx=npc_pos.x-ego_pos.x
    dy=npc_pos.y-ego_pos.y
    along=dx*forward.x+dy*forward.y
    lateral=abs(-dx*forward.y+dy*forward.x)
    heading=yaw_difference(ego_wp.transform.rotation.yaw,
                           npc_wp.transform.rotation.yaw)
    width=min(float(ego_wp.lane_width),float(npc_wp.lane_width))
    lateral_limit=max(1.7,min(2.6,width*0.62))
    return (-0.5 < along < 50.0 and lateral <= lateral_limit
            and heading <= 13.0), along, lateral, heading


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ego-id',required=True,type=int)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--host',default='127.0.0.1')
    p.add_argument('--port',default=2000,type=int)
    p.add_argument('--tm-port',default=8000,type=int)
    p.add_argument('--gap',default=30.,type=float,help='Center-to-center setup distance in meters')
    p.add_argument('--ego-kmh',default=28.,type=float)
    p.add_argument('--npc-kmh',default=12.,type=float)
    p.add_argument('--lead-in',default=3.5,type=float,help='Wall-clock seconds of moving lead-in at matched speed before chase')
    p.add_argument('--cruise-kmh',default=8.,type=float,help='Both vehicles drive at this speed during lead-in')
    p.add_argument('--timeout',default=22.,type=float,help='Max collision attempt after setup (seconds)')
    p.add_argument('--diagnose',action='store_true',help='Print detailed road/lane and alignment information each second')
    args=p.parse_args()
    if not (18<=args.gap<=30 and 15<=args.ego_kmh<=50 and 5<=args.npc_kmh<args.ego_kmh and 4<=args.timeout<=40 and 0<=args.lead_in<=8 and 5<=args.cruise_kmh<=20):
        p.error('Require gap 18-30, Ego 15-50 km/h, moving NPC 5+ km/h, and timeout 4-40 seconds')
    client=carla.Client(args.host,args.port)
    client.set_timeout(10)
    world=client.get_world()
    print('MAP:',world.get_map().name,'| no world.tick(), no map reload',flush=True)
    if args.resume:
        resume.expected_ego_id=args.ego_id
        return resume(world,args.tm_port)
    if STATE.exists():
        print('Previous FORCED demo was frozen. First use --resume.')
        return 2
    if OLD_STATE.exists():
        print('Old natural-rear state found. Run THIS script with --resume once before new test.')
        return 2
    ego=world.get_actor(args.ego_id)
    if ego is None or not ego.is_alive or not ego.type_id.startswith('vehicle.'):
        print('ERROR: Ego ID not found. Check ego_vehicle.py latest actor ID.')
        return 2
    vehicles=[]
    for attempt in range(6):
        vehicles=list(world.get_actors().filter('vehicle.*'))
        if len(vehicles)>1:
            break
        time.sleep(0.3)
    npc_candidates=[a for a in vehicles if a.id!=ego.id and a.is_alive and
                    a.attributes.get('role_name','') not in ('hero','ego_moving_crash_target') and
                    a.bounding_box.extent.x>=1.3]
    random.shuffle(npc_candidates)
    if not npc_candidates:
        print('ERROR: no available existing NPC. Keep generate_traffic.py running.')
        return 2
    # Randomly reuse ONE existing NPC, and reposition it in front of Ego.
    # Do not spawn extra cars or alter other traffic participants.
    npc=npc_candidates[0]
    site=choose_site(world,{ego.id,npc.id},ego.bounding_box.extent.x,
                     npc.bounding_box.extent.x,args.gap)
    if site is None:
        print('ABORT: no sufficiently clear straight DRIVING lane found. Other traffic untouched.')
        print('Try again after traffic moves, or restart with fewer crowded cars.')
        return 1
    e_pose,n_pose,road,lane,section=site
    print('EXISTING EGO:',ego.id,'| EXISTING NPC:',npc.id)
    print('STAGING on STRAIGHT ROAD %s LANE %s: place existing NPC ahead of Ego.' % (road,lane))
    print('MOVING LEAD-IN: BOTH cars %.0f km/h for %.1f wall seconds, then Ego %.0f vs NPC %.0f km/h.' %
          (args.cruise_kmh,args.lead_in,args.ego_kmh,args.npc_kmh),flush=True)
    print('NO stationary countdown. No world.tick() / map or simulation setting changes.',flush=True)
    print('NOTE: controlled test, NOT spontaneous traffic accident. Other actors unchanged.',flush=True)
    print('V4: follow staged forward lane waypoints; never re-project onto opposite traffic lane.',flush=True)
    collided=threading.Event()
    armed=threading.Event()
    event_info={}
    sensor=None
    rows=[]
    frozen=False
    start=time.monotonic()
    csvfile=Path(__file__).with_name('forced_rear_demo_v4_%s.csv' % datetime.now().strftime('%Y%m%d_%H%M%S'))
    try:
        ego.set_autopilot(False,args.tm_port)
        npc.set_autopilot(False,args.tm_port)
        ego.set_simulate_physics(True)
        npc.set_simulate_physics(True)
        for car, pose in ((ego,e_pose),(npc,n_pose)):
            # Clear steering, physics impulses, and angular motion before staging.
            car.apply_control(carla.VehicleControl(throttle=0,steer=0,brake=1,hand_brake=False))
            car.set_target_velocity(carla.Vector3D(0,0,0))
            car.set_target_angular_velocity(carla.Vector3D(0,0,0))
            car.set_transform(pose)
            car.apply_control(carla.VehicleControl(throttle=0,steer=0,brake=0,hand_brake=False))
            # Start the moving lead-in right after staging, before sensor setup.
            # This matters when the CARLA simulation runs faster than wall time.
            forward=pose.get_forward_vector()
            speed=args.cruise_kmh/3.6
            car.set_target_velocity(carla.Vector3D(forward.x*speed,forward.y*speed,0))
        sensor=world.spawn_actor(world.get_blueprint_library().find('sensor.other.collision'),
                                 carla.Transform(),attach_to=ego)
        def collision_callback(ev):
            if armed.is_set() and ev.other_actor.id==npc.id and not collided.is_set():
                event_info.update(frame=int(ev.frame),sim_time=float(ev.timestamp),
                                  seconds=time.monotonic()-start)
                collided.set()
        sensor.listen(collision_callback)
        armed.set()
        first_near=None
        last_print=-1
        sim_start = world.get_snapshot().timestamp.elapsed_seconds
        chase_started = False
        chase_wall = None
        ego_route_index = 0
        npc_route_index = round(args.gap / 2.0)
        print('CLOCK: recording wall time and CARLA simulation time separately.', flush=True)
        while time.monotonic()-start<args.timeout and not collided.is_set():
            if not ego.is_alive or not npc.is_alive:
                print('ABORT: Ego or NPC disappeared.')
                break
            e_loc=ego.get_location()
            n_loc=npc.get_location()
            # Follow the SAME preverified connected straight route for both cars.
            # CARLA's nearest-road projection can snap to an opposite-direction
            # lane near a road boundary, producing a false 180-degree heading.
            ew, ego_route_index, e_offset = route_waypoint(section,e_loc,ego_route_index)
            nw, npc_route_index, n_offset = route_waypoint(section,n_loc,npc_route_index)
            if ew is None or nw is None or e_offset>2.4 or n_offset>2.4:
                print('ABORT: off staged straight route (E=%.2fm N=%.2fm).' %
                      (e_offset,n_offset),flush=True)
                break
            same_lane, longitudinal, lateral, lane_heading = same_straight_approach(
                ew, nw, e_loc, n_loc)
            if not same_lane:
                print('ABORT: vehicles no longer geometrically share one straight lane '
                      '(forward=%.1fm lateral=%.2fm heading=%.1fdeg).' %
                      (longitudinal,lateral,lane_heading),flush=True)
                break
            elapsed=time.monotonic()-start
            if elapsed < args.lead_in:
                ego_goal=npc_goal=args.cruise_kmh
                phase='LEAD-IN'
            else:
                ego_goal=args.ego_kmh
                npc_goal=args.npc_kmh
                phase='CHASE'
                if not chase_started:
                    chase_started=True
                    chase_wall=elapsed
                    print('CHASE STARTED: both cars already moving. gap %.1fm' % planar(e_loc,n_loc), flush=True)
            e_heading,e_lateral=drive_along_waypoint(ego,ew,ego_goal)
            n_heading,n_lateral=drive_along_waypoint(npc,nw,npc_goal)
            if max(abs(e_heading),abs(n_heading))>22 or max(abs(e_lateral),abs(n_lateral))>2.0:
                print('ABORT: vehicle alignment lost; returning to autopilot.')
                break
            d=planar(e_loc,n_loc)
            if d<11 and first_near is None:
                first_near=time.monotonic()-start
                print('GEOMETRY NEAR: %.1fm (not LiDAR)' % d,flush=True)
            elapsed=time.monotonic()-start
            sim_elapsed=world.get_snapshot().timestamp.elapsed_seconds-sim_start
            if int(elapsed)!=last_print:
                print('wall=%.1fs | sim=%.1fs | %s | center=%.1fm | Ego=%.1f | NPC=%.1f km/h' %
                      (elapsed,sim_elapsed,phase,d,kmh(ego),kmh(npc)),flush=True)
                if args.diagnose:
                    print('  ROUTE EGO road/lane=%s/%s NPC=%s/%s | lateral=%.2fm '
                          'heading=%.1fdeg | steering offset E/N=%.2f/%.2fm' %
                          (ew.road_id,ew.lane_id,nw.road_id,nw.lane_id,
                           lateral,lane_heading,e_lateral,n_lateral),flush=True)
                last_print=int(elapsed)
            rows.append([round(elapsed,3),round(sim_elapsed,3),phase,round(d,3),round(kmh(ego),2),round(kmh(npc),2),0])
            time.sleep(0.04)
        if collided.is_set():
            collision_sim_elapsed=event_info['sim_time']-sim_start
            print('CONFIRMED CONTACT: Ego %d -> moving NPC %d | wall %.2fs | sim %.2fs since start' %
                  (ego.id,npc.id,event_info['seconds'],collision_sim_elapsed),flush=True)
            rows.append([round(event_info['seconds'],3),round(collision_sim_elapsed,3),'CONTACT',0,round(kmh(ego),2),round(kmh(npc),2),1])
            for a in (ego,npc):
                freeze(a)
            STATE.write_text(json.dumps({'map':world.get_map().name,'ego':ego.id,'npc':npc.id}),encoding='utf-8')
            frozen=True
            print('BOTH CRASH CARS FROZEN. Other city traffic remains active. Use --resume for next demo.',flush=True)
            if first_near is not None:
                print('GEOMETRY warning lead: %.3fs (NOT LiDAR TTC)' %
                      (event_info['seconds']-first_near),flush=True)
            return 0
        print('NO VERIFIED CONTACT within %.1fs; restoring autopilot.' % args.timeout)
        return 1
    except (RuntimeError,KeyboardInterrupt) as e:
        print('STOP:',e)
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
            for a in (ego,npc):
                try:
                    return_to_autopilot(a,args.tm_port)
                except RuntimeError:
                    pass
        with csvfile.open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.writer(f)
            writer.writerow(['elapsed_wall_s','elapsed_sim_s','phase','center_distance_m','ego_kmh','npc_kmh','confirmed_contact'])
            writer.writerows(rows)
        print('CSV SAVED:',csvfile,flush=True)

if __name__=='__main__':
    raise SystemExit(main())
