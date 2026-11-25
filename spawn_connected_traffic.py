#!/usr/bin/env python3

import argparse
import random
import time

import carla


# ============================================================
# Arguments
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            'Spawn CARLA vehicles as cooperative-perception '
            'participants with role_name car_01 ~ car_N.'
        )
    )

    parser.add_argument(
        '--host',
        default='140.130.35.3',
        help='CARLA server IP'
    )

    parser.add_argument(
        '--port',
        type=int,
        default=2000,
        help='CARLA RPC port'
    )

    parser.add_argument(
        '--tm-port',
        type=int,
        default=8000,
        help='Traffic Manager port'
    )

    parser.add_argument(
        '-n',
        '--number',
        type=int,
        default=50,
        help='Number of connected vehicles'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )

    parser.add_argument(
        '--speed-difference',
        type=float,
        default=0.0,
        help=(
            'Traffic Manager global speed difference percentage. '
            'Positive means slower.'
        )
    )

    parser.add_argument(
        '--clean',
        action='store_true',
        help='Destroy existing CARLA vehicle actors before spawning'
    )

    return parser.parse_args()


# ============================================================
# Cleanup existing vehicles
# ============================================================

def destroy_existing_vehicles(
    client,
    world
):

    vehicles = list(
        world.get_actors().filter(
            'vehicle.*'
        )
    )

    if not vehicles:

        print(
            'No existing vehicles to remove.'
        )

        return

    print(
        f'Removing {len(vehicles)} '
        f'existing vehicle actors...'
    )

    commands = [
        carla.command.DestroyActor(
            actor.id
        )
        for actor in vehicles
    ]

    client.apply_batch_sync(
        commands,
        False
    )

    # Allow CARLA one moment to update actor registry.
    time.sleep(1.0)


# ============================================================
# Vehicle blueprint selection
# ============================================================

def get_vehicle_blueprints(
    world
):

    library = (
        world.get_blueprint_library()
    )

    blueprints = list(
        library.filter(
            'vehicle.*'
        )
    )

    if not blueprints:

        raise RuntimeError(
            'No vehicle blueprints found.'
        )

    # Filter out unusual static-looking entries if any.
    usable = []

    for blueprint in blueprints:

        if not blueprint.has_attribute(
            'role_name'
        ):
            continue

        usable.append(
            blueprint
        )

    if not usable:

        raise RuntimeError(
            'No usable vehicle blueprints '
            'with role_name attribute.'
        )

    return usable


# ============================================================
# Spawn
# ============================================================

def spawn_connected_vehicles(
    world,
    traffic_manager,
    number,
    seed
):

    random.seed(
        seed
    )

    blueprints = (
        get_vehicle_blueprints(
            world
        )
    )

    spawn_points = list(
        world.get_map()
        .get_spawn_points()
    )

    random.shuffle(
        spawn_points
    )

    if number > len(
        spawn_points
    ):

        raise RuntimeError(
            f'Requested {number} vehicles, '
            f'but map only has '
            f'{len(spawn_points)} spawn points.'
        )

    spawned = []

    next_vehicle_index = 1

    # Try every spawn point until N vehicles exist.
    for spawn_point in spawn_points:

        if len(spawned) >= number:
            break

        blueprint_template = (
            random.choice(
                blueprints
            )
        )

        # Get a fresh blueprint reference by ID.
        blueprint = (
            world.get_blueprint_library()
            .find(
                blueprint_template.id
            )
        )

        vehicle_id = (
            f'car_{next_vehicle_index:02d}'
        )

        blueprint.set_attribute(
            'role_name',
            vehicle_id
        )

        # Random color where supported.
        if blueprint.has_attribute(
            'color'
        ):

            recommended = (
                blueprint
                .get_attribute(
                    'color'
                )
                .recommended_values
            )

            if recommended:

                blueprint.set_attribute(
                    'color',
                    random.choice(
                        recommended
                    )
                )

        # Random driver where supported.
        if blueprint.has_attribute(
            'driver_id'
        ):

            recommended = (
                blueprint
                .get_attribute(
                    'driver_id'
                )
                .recommended_values
            )

            if recommended:

                blueprint.set_attribute(
                    'driver_id',
                    random.choice(
                        recommended
                    )
                )

        actor = (
            world.try_spawn_actor(
                blueprint,
                spawn_point
            )
        )

        if actor is None:
            continue

        actor.set_autopilot(
            True,
            traffic_manager.get_port()
        )

        spawned.append(
            actor
        )

        print(
            f'[{vehicle_id}] '
            f'actor_id={actor.id:<4} '
            f'type={actor.type_id}'
        )

        next_vehicle_index += 1

    if len(spawned) != number:

        print()
        print(
            f'WARNING: requested {number}, '
            f'but only spawned '
            f'{len(spawned)} vehicles.'
        )

    return spawned


# ============================================================
# Verification
# ============================================================

def verify_roles(
    world,
    expected_number
):

    connected = []

    for actor in (
        world.get_actors()
        .filter(
            'vehicle.*'
        )
    ):

        role_name = (
            actor.attributes.get(
                'role_name',
                ''
            )
        )

        if role_name.startswith(
            'car_'
        ):

            connected.append(
                (
                    role_name,
                    actor.id,
                    actor.type_id
                )
            )

    connected.sort(
        key=lambda item: item[0]
    )

    print()
    print(
        '===== Connected Vehicle Registry ====='
    )

    for (
        role_name,
        actor_id,
        type_id
    ) in connected:

        print(
            f'{role_name:<8} '
            f'actor_id={actor_id:<4} '
            f'{type_id}'
        )

    print()
    print(
        f'Connected vehicles: '
        f'{len(connected)} / '
        f'{expected_number}'
    )

    roles = [
        item[0]
        for item in connected
    ]

    duplicate_count = (
        len(roles)
        -
        len(set(roles))
    )

    print(
        f'Duplicate role_names: '
        f'{duplicate_count}'
    )

    return (
        len(connected)
        ==
        expected_number
        and
        duplicate_count
        ==
        0
    )


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    client = carla.Client(
        args.host,
        args.port
    )

    client.set_timeout(
        10.0
    )

    world = (
        client.get_world()
    )

    print(
        f'Connected to CARLA '
        f'{args.host}:{args.port}'
    )

    print(
        f'Map: '
        f'{world.get_map().name}'
    )

    # --------------------------------------------------------
    # Traffic Manager
    # --------------------------------------------------------

    traffic_manager = (
        client.get_trafficmanager(
            args.tm_port
        )
    )

    traffic_manager.set_random_device_seed(
        args.seed
    )

    traffic_manager.global_percentage_speed_difference(
        args.speed_difference
    )

    # --------------------------------------------------------
    # Optional cleanup
    # --------------------------------------------------------

    if args.clean:

        destroy_existing_vehicles(
            client,
            world
        )

    # --------------------------------------------------------
    # Spawn
    # --------------------------------------------------------

    vehicles = (
        spawn_connected_vehicles(
            world=world,
            traffic_manager=traffic_manager,
            number=args.number,
            seed=args.seed
        )
    )

    if not vehicles:

        raise RuntimeError(
            'No connected vehicles were spawned.'
        )

    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    valid = verify_roles(
        world,
        len(vehicles)
    )

    if not valid:

        print(
            'WARNING: Connected vehicle '
            'registry verification failed.'
        )

    else:

        print()
        print(
            'Connected vehicle registry OK.'
        )

    print()
    print(
        'Traffic is running.'
    )

    print(
        'Press Ctrl+C to destroy these '
        'connected vehicles and exit.'
    )

    # --------------------------------------------------------
    # Keep this client / TM session alive
    # --------------------------------------------------------

    try:

        while True:

            time.sleep(
                1.0
            )

    except KeyboardInterrupt:

        print()
        print(
            'Stopping connected traffic...'
        )

    finally:

        commands = [
            carla.command.DestroyActor(
                actor.id
            )
            for actor in vehicles
            if actor.is_alive
        ]

        if commands:

            client.apply_batch_sync(
                commands,
                False
            )

        print(
            'Connected vehicles destroyed.'
        )


if __name__ == '__main__':

    main()