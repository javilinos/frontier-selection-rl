#!/bin/python3

"""
get_drones.py
"""

import argparse
import json
from pathlib import Path


def get_drones_yml(filename: Path) -> list[str]:
    """Get drone names listed in swarm config file"""
    with open(filename, "r", encoding='utf-8') as f:
        lines = f.readlines()
        drones_ = []
        for line in lines:
            if line.strip().startswith("#"):
                continue
            if line.strip().startswith("uri"):
                continue

            drones_.append(line.strip().split(":")[0])
    return drones_


def get_drones_json(filename: Path) -> list[str]:
    """Get drone names listed in swarm config file"""
    with open(f"{filename}", "r", encoding='utf-8') as f:
        world_json = json.load(f)
    drones_ = []
    for drone in world_json['drones']:
        drones_.append(drone['model_name'])
    return drones_


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="Path to swarm config file")
    parser.add_argument("--sep", help="Separator", default=":")
    args = parser.parse_args()

    config_file = Path(args.config_file)
    if not config_file.exists():
        raise FileNotFoundError(f"File {config_file} not found")
    if config_file.suffix in (".yml", ".yaml"):
        drones = get_drones_yml(config_file)
    elif config_file.suffix == ".json":
        drones = get_drones_json(config_file)
    else:
        raise ValueError(
            f"File extension '{config_file.suffix}' not supported")

    # Return for use in bash scripts
    print(args.sep.join(drones))
