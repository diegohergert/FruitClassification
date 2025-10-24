import os
import csv
from collections import Counter
import matplotlib
matplotlib.use('Agg')  # headless backend
import matplotlib.pyplot as plt
from typing import Optional


TRAINING_DIR = os.path.join('FruitClassification', 'data', 'fruits-360_100x100', 'fruits-360', 'Training')
OUTPUT_CSV = 'class_counts.csv'
OUTPUT_PNG = 'class_distribution.png'
OUTPUT_BAR_PNG = 'class_distribution_bar.png'
OUTPUT_BAR_LOG_PNG = 'class_distribution_bar_log.png'


def is_image_file(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    return ext in {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}


def count_images(training_dir: str) -> Counter:
    counts = Counter()
    for entry in os.scandir(training_dir):
        if entry.is_dir():
            class_name = entry.name
            n = 0
            for root, _, files in os.walk(entry.path):
                for f in files:
                    if is_image_file(f):
                        n += 1
            counts[class_name] = n
    return counts


def save_csv(counter: Counter, path: str) -> None:
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['class', 'count'])
        for cls, cnt in counter.most_common():
            writer.writerow([cls, cnt])


def plot_pie(counter: Counter, path: str, top_n: Optional[int] = None) -> None:
    """Create a pie chart of class distribution.

    If top_n is None, include all classes. When many classes exist the labels
    are shown in a legend on the right for readability. Colors are chosen from
    an HSV colormap so many distinct hues are available.
    """
    total = sum(counter.values())
    if total == 0:
        print('No images found to plot.')
        return

    most = counter.most_common() if top_n is None else counter.most_common(top_n)
    labels = [c for c, _ in most]
    sizes = [s for _, s in most]

    # If top_n was specified and there are remaining classes, aggregate them
    if top_n is not None:
        other = total - sum(sizes)
        if other > 0:
            labels.append('Other')
            sizes.append(other)

    # Use a continuous colormap (HSV) to get many distinct colors
    cmap = plt.get_cmap('hsv')
    n = len(sizes)
    colors = [cmap(i / max(1, n)) for i in range(n)]

    def make_autopct(sizes_list):
        def autopct(pct):
            absolute = int(round(pct / 100.0 * sum(sizes_list)))
            return f"{pct:.1f}%\n({absolute})"
        return autopct

    # Make a wide figure so the legend fits when many classes exist
    fig, ax = plt.subplots(figsize=(16, max(8, n * 0.12 + 4)))
    wedges, _texts, autotexts = ax.pie(
        sizes,
        labels=None,              # use legend for long labels
        colors=colors,
        startangle=140,
        autopct=make_autopct(sizes),
        pctdistance=0.72
    )
    ax.axis('equal')
    title = 'Image count per class (all classes)' if top_n is None else f'Image count per class (top {top_n} + Other)'
    ax.set_title(title)

    # Legend on the right
    #ax.legend(wedges, labels, title="Classes", loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout(rect=[0, 0, 0.78, 1])  # make room for legend
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_bar(counter: Counter, path: str, log_scale: bool = False) -> None:
    """Create a horizontal bar chart of class counts.

    When there are many classes this is more readable than a pie chart. The
    bars are sorted by count descending.
    """
    items = counter.most_common()
    if not items:
        print('No data for bar plot.')
        return

    labels, values = zip(*items)
    # Reverse for horizontal bar (largest on top)
    labels = list(labels)[::-1]
    values = list(values)[::-1]

    fig_height = max(6, len(labels) * 0.15)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    y_positions = range(len(labels))
    # generate a distinct color per bar from a categorical colormap
    cmap = plt.get_cmap('tab20')
    n = len(labels)
    colors = [cmap(i / max(1, n - 1)) for i in range(n)]
    ax.barh(y_positions, values, color=colors)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Number of images')
    ax.set_title('Images per class' + (' (log scale)' if log_scale else ''))
    if log_scale:
        ax.set_xscale('log')

    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    training_dir = TRAINING_DIR
    if not os.path.isdir(training_dir):
        print(f"Training directory not found: {training_dir}")
        return

    counts = count_images(training_dir)
    if not counts:
        print('No class subfolders found in training directory.')
        return

    save_csv(counts, OUTPUT_CSV)
    print(f'Saved class counts to {OUTPUT_CSV} (total classes: {len(counts)})')
    # Plot all classes (no aggregation) so each class has its own slice
    plot_pie(counts, OUTPUT_PNG, top_n=None)
    print(f'Saved pie chart to {OUTPUT_PNG}')

    # Also save readable bar charts (linear and log scale)
    plot_bar(counts, OUTPUT_BAR_PNG, log_scale=False)
    print(f'Saved bar chart to {OUTPUT_BAR_PNG}')
    #plot_bar(counts, OUTPUT_BAR_LOG_PNG, log_scale=True)
    #print(f'Saved log-scale bar chart to {OUTPUT_BAR_LOG_PNG}')


if __name__ == '__main__':
    main()
