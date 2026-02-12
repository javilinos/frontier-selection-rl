import matplotlib.pyplot as plt

def plot_path_from_rosbag(
    bag_path: str,
    topic: str = "/drone0/self_localization/pose",
):
    """
    Load a ROS 2 rosbag2 (sqlite3 folder or mcap file/folder), read PoseStamped messages
    from `topic`, and plot the XY path.
    """
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    # ---- Open bag (try common storage backends) ----
    last_err = None
    reader = None
    for storage_id in ("sqlite3", "mcap"):
        try:
            reader = SequentialReader()
            storage_options = StorageOptions(uri=bag_path, storage_id=storage_id)
            converter_options = ConverterOptions(
                input_serialization_format="cdr",
                output_serialization_format="cdr",
            )
            reader.open(storage_options, converter_options)
            last_err = None
            break
        except Exception as e:
            last_err = e
            reader = None

    if reader is None:
        raise RuntimeError(
            f"Could not open rosbag at '{bag_path}' with storage_id sqlite3 or mcap. "
            f"Last error: {last_err}"
        )

    # ---- Build topic -> type dict robustly ----
    topic_types = {}
    all_topics = reader.get_all_topics_and_types() or []
    for t in all_topics:
        # TopicMetadata has .name and .type
        name = getattr(t, "name", None)
        typ = getattr(t, "type", None)
        if name and typ:
            topic_types[name] = typ

    if not topic_types:
        raise RuntimeError(
            f"No topics found in bag '{bag_path}'. "
            f"(metadata may be missing/corrupt, or bag empty)"
        )

    if topic not in topic_types:
        available = "\n".join(sorted(topic_types.keys()))
        raise ValueError(f"Topic '{topic}' not found in bag. Available topics:\n{available}")

    msg_type_str = topic_types[topic]
    msg_type = get_message(msg_type_str)

    # ---- Optional filter (older distros may not support it) ----
    try:
        from rosbag2_py import StorageFilter
        reader.set_filter(StorageFilter(topics=[topic]))
        use_filter = True
    except Exception:
        use_filter = False

    # ---- Read messages ----
    poses = []
    while reader.has_next():
        t_name, raw, _ = reader.read_next()
        if (not use_filter) and (t_name != topic):
            continue
        try:
            msg = deserialize_message(raw, msg_type)
        except Exception:
            # Skip malformed messages instead of crashing
            continue
        poses.append(msg)

    if not poses:
        raise RuntimeError(
            f"Topic '{topic}' exists but had 0 readable messages in this bag."
        )

    # ---- Convert PoseStamped list -> polyline path ----
    path_xy = []
    for m in poses:
        # Defensive: in case message lacks fields (shouldn't happen for PoseStamped)
        try:
            x = m.pose.position.x
            y = m.pose.position.y
            path_xy.append((x, y))
        except Exception:
            continue

    if len(path_xy) < 2:
        raise RuntimeError(
            f"Not enough valid poses to plot (got {len(path_xy)} points)."
        )

    # ---- Plot (same style as your original) ----
    GRID_SIZE = 5
    HALF = GRID_SIZE / 2

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.set_xticks([-2, 0, 2])
    ax.set_yticks([-2, 0, 2])

    ax.set_xticks(range(-int(HALF), int(HALF) + 1), minor=True)
    ax.set_yticks(range(-int(HALF), int(HALF) + 1), minor=True)

    ax.grid(which="minor", linestyle="--", linewidth=0.5)
    ax.grid(which="major", linestyle="-", linewidth=1)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    xs, ys = zip(*path_xy)
    ax.plot(xs, ys, color="blue", linewidth=5, alpha=0.3, zorder=1)

    ax.set_aspect("equal", "box")
    ax.set_xlim(-HALF, HALF)
    ax.set_ylim(-HALF, HALF)

    plt.tight_layout()
    plt.show()



# Example:
plot_path_from_rosbag("rosbag2_2025_12_17-16_49_50")
