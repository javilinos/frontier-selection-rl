"""
Random world generator
"""
from __future__ import annotations
import argparse
import math
from pathlib import Path
import random
from jinja2 import Environment, FileSystemLoader

__authors__ = "David Ho, Pedro Arias Perez"
__license__ = "BSD-3-Clause"


def distance(point1: tuple[float, float], point2: tuple[float, float]) -> float:
    """
    Calculate the euclidean distance between 2 points
    """
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def randomize_drone_pose(upper: int, lower: int) -> dict[str, list]:
    """
    Generate a random drone position within a [lower, upper] range
    """
    if upper < lower:
        raise ValueError
    x = round(random.uniform(lower, upper), 2)
    y = round(random.uniform(lower, upper), 2)
    yaw = round(random.uniform(0, 2 * math.pi), 2)

    drone = {
        "xyz": [x, y, 0],
        "rpy": [0, 0, yaw],
    }
    return drone


def generate_obstacles(num_objects: int, drone_coords: tuple[float, float],
                       min_distance: float, upper: int, lower: int) -> dict[str, list[float]]:
    """
    Given drone coordinates, generate random obstacles within the same grid

    drone_coords = (x, y)
    min_distance = minimum distance away from drone
    """
    if upper < lower:
        raise ValueError

    objects = {}
    drones = [drone_coords]
    for i in range(num_objects):
        while True:
            x = round(random.uniform(lower, upper), 2)
            y = round(random.uniform(lower, upper), 2)

            too_close = any(distance((x, y), drone) <
                            min_distance for drone in drones)

            if not too_close:
                break

        objects[f"pole{i + 1}"] = [x, y, 0, 0, 0, 0]
    return objects


def generate_world(world_name: str, num_world: int, num_object: int,
                   safety_margin: int) -> None:
    """Generate world"""
    assets_path = Path(__file__).parents[1].joinpath('assets')
    environment = Environment(loader=FileSystemLoader(
        assets_path.joinpath("templates")))

    json_template = environment.get_template("drone.json.jinja")
    sdf_template = environment.get_template("world.sdf.jinja")
    world_size = 10.0

    for i in range(num_world):
        world_name = f"{world_name}"
        drone = randomize_drone_pose(upper=world_size, lower=-world_size)
        drone_xy = drone['xyz'][:-1]
        drone['model_name'] = "drone0"
        obstacles = generate_obstacles(
            num_object, drone_xy, safety_margin, world_size, -world_size)

        json_output = json_template.render(
            world_name=world_name, drones=[drone])
        sdf_output = sdf_template.render(
            world_name=world_name, models=obstacles)

        json_path = assets_path.joinpath(f"worlds/{world_name}.json")
        sdf_path = assets_path.joinpath(f"worlds/{world_name}.sdf")

        # Write the JSON string to the file
        with open(json_path, "w", encoding='utf-8') as json_file:
            json_file.write(json_output)

        # Write the XML string to the file
        with open(sdf_path, "w", encoding='utf-8') as sdf_file:
            sdf_file.write(sdf_output)

        print(f"{world_name} has been saved")


def main():
    """
    entrypoint
    """
    parser = argparse.ArgumentParser(
        description='Generate a randomize world as a json file')
    parser.add_argument('number_of_objects', metavar='objects', type=int,
                        help='number of objects to generate')
    parser.add_argument('number_of_worlds', metavar='worlds', type=int, nargs='?',
                        default=1, help='number of worlds to generate')
    parser.add_argument('margin_of_safety', metavar='safety', type=int, nargs='?',
                        default=2, help='margin of safety for the drone (m)')
    parser.add_argument('-name', '--world_name', metavar='name', type=str, nargs='?',
                        default='world_density_enormous', help='generic name of generated worlds')
    args = parser.parse_args()

    num_obj = args.number_of_objects    # Number of Objects
    mos = args.margin_of_safety         # Margin of Safety for the drone
    num_world = args.number_of_worlds   # Number of worlds to generate
    world_name = args.world_name        # World name

    generate_world(world_name, num_world, num_obj, mos)


if __name__ == '__main__':
    main()
