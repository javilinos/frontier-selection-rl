#!/usr/bin/env python3
import argparse
import matplotlib.pyplot as plt
from pathlib import Path


#!/usr/bin/env python3
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.collections import LineCollection
import numpy as np


def plot_like_plot_path(obstacles, paths, grid_size: int = 20):
    GRID_SIZE = int(grid_size)
    HALF = GRID_SIZE / 2

    fig, ax = plt.subplots(figsize=(6, 6))

    # Major ticks
    major = [-GRID_SIZE // 2, 0, GRID_SIZE // 2]
    ax.set_xticks(major)
    ax.set_yticks(major)

    # Minor ticks
    ax.set_xticks(range(-int(HALF), int(HALF) + 1), minor=True)
    ax.set_yticks(range(-int(HALF), int(HALF) + 1), minor=True)

    # Grid
    ax.grid(which='minor', linestyle='--', linewidth=0.5)
    ax.grid(which='major', linestyle='-', linewidth=1)

    # Hide spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # 🔥 Bigger obstacles
    if obstacles:
        xo, yo = zip(*obstacles)
        ax.scatter(
            xo, yo,
            s=450,              # increased size
            color='black',
            zorder=3
        )

    # 🔥 Proper overlapping darkness using LineCollection
    for path in paths:
        if len(path) < 2:
            continue

        pts = np.array(path)
        segments = np.stack([pts[:-1], pts[1:]], axis=1)

        lc = LineCollection(
            segments,
            colors='blue',
            linewidths=6,
            alpha=0.35,
            zorder=1
        )

        ax.add_collection(lc)

    ax.set_aspect('equal', 'box')
    ax.set_xlim(-HALF, HALF)
    ax.set_ylim(-HALF, HALF)

    plt.tight_layout()
    plt.show()


def read_path_from_rosbag(bag_path: str, topic: str):
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    bag_path = str(Path(bag_path).resolve())

    reader = None
    last_err = None
    for storage_id in ("sqlite3", "mcap"):
        try:
            reader = SequentialReader()
            reader.open(
                StorageOptions(uri=bag_path, storage_id=storage_id),
                ConverterOptions("cdr", "cdr"),
            )
            break
        except Exception as e:
            last_err = e
            reader = None

    if reader is None:
        raise RuntimeError(f"Could not open bag: {last_err}")

    topic_types = {
        t.name: t.type
        for t in reader.get_all_topics_and_types() or []
    }

    if topic not in topic_types:
        raise ValueError(f"Topic '{topic}' not found.")

    msg_type = get_message(topic_types[topic])

    try:
        from rosbag2_py import StorageFilter
        reader.set_filter(StorageFilter(topics=[topic]))
        use_filter = True
    except Exception:
        use_filter = False

    path = []
    while reader.has_next():
        t_name, raw, _ = reader.read_next()
        if not use_filter and t_name != topic:
            continue
        try:
            msg = deserialize_message(raw, msg_type)
            path.append((msg.pose.position.x, msg.pose.position.y))
        except Exception:
            continue

    if len(path) < 2:
        raise RuntimeError("Not enough valid poses.")

    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True)
    parser.add_argument("--topic", default="/drone0/self_localization/pose")
    parser.add_argument("--grid-size", type=int, default=5)

    args = parser.parse_args()

    # 🔥 DEFINE OBSTACLES HERE (by code)
    obstacles = [
        (-0.9, 1.0),
        (1.4, 1.4),
        (2.2, 1.2),
        (1.0, -1.0),
        (-1.7, -1.3)
    ]

    path = read_path_from_rosbag(args.bag, args.topic)

    # wrap as list to match plot_path(obstacles, paths)
    plot_like_plot_path(
        obstacles=obstacles,
        paths=[path],
        grid_size=args.grid_size
    )


if __name__ == "__main__":
    main()