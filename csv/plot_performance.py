import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def load_csv_files(directory='.', file_name=""):
    """
    Find and return a list of all CSV files in the specified directory.
    By default, the current directory is used.
    """
    # Construct a file pattern
    pattern = os.path.join(directory, f"{file_name}.csv")
    # Use glob to retrieve matching files
    csv_files = glob.glob(pattern)
    return csv_files


def load_dataframe(csv_files, dataframes=None, file_name=""):
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            # Verify that the necessary columns are present.
            # if 'distance' in df.columns and 'mean_area_explored' in df.columns:
            # Use the file name as the label for later plotting.
            dataframes[file_name] = df
            # else:
            #     print(f"Warning: {file} does not contain required columns.")
        except Exception as e:
            print(f"Error reading {file}: {e}")
    return dataframes


def plot_performance(dataframes):
    """
    Create a single plot with each DataFrame's accumulated path length data.
    Each CSV file is plotted as a separate line, with the last one highlighted.
    Uses manually assigned colors for consistency.
    """
    plt.figure(figsize=(10, 6))

    # Manually define a list of colors (same as before)
    manual_colors = ['#8c564b', '#ff7f0e', '#2ca02c', '#d62728',
                     '#9467bd', '#1f77b4', '#e377c2', '#1f77b4']

    items = list(dataframes.items())
    for i, (label, df) in enumerate(items):
        color = manual_colors[i % len(manual_colors)]
        is_last = (i == len(items) - 1)

        plt.plot(
            df['episode'],
            df['accumulated_path_length'],
            label=label,
            color=color,
            linewidth=4 if is_last else 1.5
        )

    plt.xlabel('Episode')
    plt.ylabel('Accumulated Path Length')
    plt.title('Accumulated Path Length per Episode Across 100 Episodes')
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()


def plot_area_performance(dataframes):
    """
    Create a single plot with each DataFrame's accumulated path length data.
    Each CSV file is plotted as a separate line, with the last one highlighted.
    Uses manually assigned colors for consistency.
    """
    # plt.figure(figsize=(10, 6))

    # Manually define a list of colors (same as before)
    manual_colors = [
        '#8c564b', '#ff7f0e', '#2ca02c', '#d62728',
        '#9467bd', '#bcbd22', '#e377c2', '#1f77b4'
    ]

    items = list(dataframes.items())
    fig, ax = plt.subplots(figsize=(14, 10))
    for i, (label, df) in enumerate(items):
        color = manual_colors[i % len(manual_colors)]
        is_last = (i == len(items) - 1)

        # Plot mean (darkest color)
        ax.plot(df['distance'], df['mean_area_explored'] * 100,
                color=color, alpha=1.0, label=label)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        # Plot standard deviation as transparent band
        ax.fill_between(
            df['distance'],
            df['mean_area_explored'] * 100 - df['std_area_explored'] * 100,
            df['mean_area_explored'] * 100 + df['std_area_explored'] * 100,
            color=color, alpha=0.3
        )
        # imax = df['mean_area_explored'].idxmax()
        # x_max = df.loc[imax, 'distance']
        # ax.axvline(
        #     x=x_max,
        #     color=color,
        #     linestyle='--',
        #     linewidth=2,
        #     alpha=0.7
        # )
    # ax.legend(
    #     loc='upper left',
    #     frameon=False,
    #     labelspacing=1.2,
    #     handlelength=3
    # )
    ax.set_xlabel('Distance (m)', fontsize=16)
    ax.set_ylabel('Area Explored (% of total area)', fontsize=16)
    ax.set_title('Mean Area Explored with Standard Deviation', fontsize=18)
    plt.grid(True, axis='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    #     plt.plot(
    #         df['time'],
    #         df['area_explored'] * 100,
    #         label=label,
    #         color=color,
    #         linewidth=4 if is_last else 1.5
    #     )

    # plt.xlabel('Time (s)')
    # plt.ylabel('Area Explored (% of total area)')
    # plt.title('% Area Explored per time at 1 m/s')
    # plt.grid(True)
    # plt.tight_layout()
    # plt.legend()
    # plt.show()


def plot_performance_summary(dataframes):
    """
    Create a bar plot showing the mean accumulated path length
      across all episodes
    for each method (DataFrame), with standard deviation as error bars.
    Each bar uses a manually specified color.
    """
    labels = []
    means = []
    stds = []

    for label, df in dataframes.items():
        mean_val = df['path_length'].mean()
        std_val = df['path_length'].std()

        labels.append(label)
        means.append(mean_val)
        stds.append(std_val)

    x = np.arange(len(labels))

    # Manually define a list of colors (extend if you have more methods)
    manual_colors = [
        '#8c564b', '#ff7f0e', '#2ca02c', '#d62728',
        '#9467bd', '#bcbd22', '#e377c2', '#1f77b4'
    ]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(x, means, yerr=stds, capsize=10, alpha=0.8, color=manual_colors[:len(labels)])

    # Add shaded std dev areas
    for i in range(len(x)):
        plt.fill_between([x[i] - 0.2, x[i] + 0.2],
                         [means[i] - stds[i]] * 2,
                         [means[i] + stds[i]] * 2,
                         color=manual_colors[i],
                         alpha=0.3)

    plt.xticks(x, labels)
    plt.ylabel('Mean Path Length Across 100 Episodes')
    plt.title('Comparison of Methods: Mean Path Length with Std Dev')
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_performance_summary_group(dataframes_groups, sep_factor=5, intra_sep_factor=0.3):
    """
    Plot bars grouped by density (low→mid→high),
    colored by method, with:
      • larger gap between density-blocks (sep_factor),
      • small gap between bars within each block (intra_sep_factor),
      • legend inside the axes,
      • extra right margin so it doesn’t overlap.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    densities = list(dataframes_groups.keys())
    method_labels = list(next(iter(dataframes_groups.values())).keys())
    n_den, n_met = len(densities), len(method_labels)

    # 1) compute means & stds
    means = np.zeros((n_den, n_met))
    stds = np.zeros((n_den, n_met))
    for i, den in enumerate(densities):
        for j, m in enumerate(method_labels):
            df = dataframes_groups[den][m]
            means[i, j] = df['path_length'].mean()
            stds[i, j] = df['path_length'].std()

    # 2) geometry
    base_block_width = 0.8             # total “usable” width per density (before adding gaps)
    bar_w = base_block_width / n_met
    intra_sep = intra_sep_factor * bar_w
    group_width = n_met * bar_w + (n_met - 1) * intra_sep
    group_sep = sep_factor * bar_w

    # 3) colors per method
    manual_colors = [
        '#8c564b', '#ff7f0e', '#2ca02c', '#d62728',
        '#9467bd', '#bcbd22', '#e377c2', '#1f77b4'
    ]
    method_colors = {
        method_labels[j]: manual_colors[j % len(manual_colors)]
        for j in range(n_met)
    }

    # 4) make figure + axes
    fig, ax = plt.subplots(figsize=(14, 6))

    # 5) plot each bar with intra-group spacing
    for i, den in enumerate(densities):
        base = i * (group_width + group_sep)
        for j, m in enumerate(method_labels):
            x = base + j * (bar_w + intra_sep)
            ax.bar(
                x,
                means[i, j],
                yerr=stds[i, j],
                capsize=5,
                width=bar_w,
                alpha=0.8,
                color=method_colors[m]
            )

    # 6) xticks at block centers
    centers = [
        i * (group_width + group_sep) + group_width / 2
        for i in range(n_den)
    ]
    ax.set_xticks(centers)
    ax.set_xticklabels([d.replace('_', ' ').title() for d in densities], fontsize=20)

    # 7) labels, grid, title
    ax.set_ylabel('Mean Path Length', fontsize=20)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)

    # 8) legend inside
    handles = [
        mpatches.Patch(color=method_colors[m], label=m)
        for m in method_labels
    ]
    ax.legend(
        handles=handles,
        title='Method',
        ncol=2,
        loc='upper right',
        frameon=True,
        framealpha=0.9,
    )

    # 9) leave margin on right
    fig.subplots_adjust(right=0.80)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # You can specify any directory where your CSV files are stored. Here we use the current directory.
    directory = '.'

    densities = ['low_density', 'medium_density', 'high_density']
    methods = [
        ('random', "Random Frontier"),
        ('information_gain', "Information Gain"),
        ('hybrid_075_025', "Hybrid\nIG: 0.75\nNearest: 0.25"),
        ('hybrid_05_05', "Hybrid\nIG: 0.5\nNearest: 0.5"),
        ('hybrid_025_075', "Hybrid\nIG: 0.25\nNearest: 0.75"),
        ('tare', "Tare Local"),
        ('nearest', "Nearest Frontier"),
        ('ours', "Ours")
    ]

    dataframes = {}
    dataframes_groups = {}

    for density in densities:
        for method_dir, label in methods:
            path = f"bars_graphic_cum_mean_path_length/{density}/{method_dir}"
            csv_files = load_csv_files(directory, file_name=path)
            # print(f"Loading CSV files from: {csv_files}")

            # make each key unique by tacking on the density
            key = f"{label} ({density.replace('_', ' ').title()})"
            dataframes = load_dataframe(csv_files, dataframes, file_name=label)
        dataframes_groups[density] = dataframes
        dataframes = {}
        # print(dataframes)
    if dataframes_groups:
        plot_performance_summary_group(dataframes_groups)
    else:
        print("No valid CSV files to plot.")

    # directory = '.'
    # dataframes = {}
    # csv_files = load_csv_files(
    #     directory, file_name="time_graphics_10_episodes/random")
    # dataframes = load_dataframe(csv_files, dataframes, file_name="Random Frontier")
    # csv_files = load_csv_files(
    #     directory, file_name="time_graphics_10_episodes/information_gain")
    # dataframes = load_dataframe(csv_files, dataframes, file_name="Information Gain")
    # csv_files = load_csv_files(
    #     directory, file_name="time_graphics_10_episodes/hybrid_075_025")
    # dataframes = load_dataframe(csv_files, dataframes,
    #                             file_name="Hybrid\nIG: 0.75\nNearest: 0.25")
    # csv_files = load_csv_files(
    #     directory, file_name="time_graphics_10_episodes/hybrid_05_05")
    # dataframes = load_dataframe(csv_files, dataframes, file_name="Hybrid\nIG: 0.5\nNearest: 0.5")
    # csv_files = load_csv_files(
    #     directory, file_name="time_graphics_10_episodes/hybrid_025_075")
    # dataframes = load_dataframe(csv_files, dataframes,
    #                             file_name="Hybrid\nIG: 0.25\nNearest: 0.75")
    # csv_files = load_csv_files(
    #     directory, file_name="time_graphics_10_episodes/tare")
    # dataframes = load_dataframe(csv_files, dataframes, file_name="Tare Local")
    # csv_files = load_csv_files(
    #     directory, file_name="time_graphics_10_episodes/nearest")
    # dataframes = load_dataframe(csv_files, dataframes, file_name="Nearest Frontier")
    # csv_files = load_csv_files(
    #     directory, file_name="time_graphics_10_episodes/ours")
    # dataframes = load_dataframe(csv_files, dataframes, file_name="Ours")

    # if dataframes:  # Only plot if we have valid DataFrames
    #     plot_area_performance(dataframes)
    # else:
    #     print("No valid CSV files to plot.")
