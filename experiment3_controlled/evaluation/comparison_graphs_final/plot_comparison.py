
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent

# Experiment 1 and 2: previously reported global pixel-level test metrics.
# Experiment 3: mean per-image metrics from the controlled held-out test set.
# Comparisons are descriptive because the metric aggregation methods differ.

plots = [
    {
        "filename": "lane_iou_comparison.png",
        "title": "Lane Segmentation: IoU",
        "ylabel": "Intersection over Union (IoU)",
        "labels": ["Experiment 1", "Experiment 3"],
        "values": [0.948260, 0.894229],
        "note": "Exp. 1: global pixel-level test | Exp. 3: mean per-image test",
    },
    {
        "filename": "lane_dice_comparison.png",
        "title": "Lane Segmentation: Dice Score",
        "ylabel": "Dice (F1)",
        "labels": ["Experiment 1", "Experiment 3"],
        "values": [0.973443, 0.942478],
        "note": "Exp. 1: global pixel-level test | Exp. 3: mean per-image test",
    },
    {
        "filename": "road_iou_comparison.png",
        "title": "Road Segmentation: IoU",
        "ylabel": "Intersection over Union (IoU)",
        "labels": ["Experiment 2", "Experiment 3"],
        "values": [0.976016, 0.993060],
        "note": "Exp. 2: global pixel-level test | Exp. 3: mean per-image test",
    },
    {
        "filename": "road_dice_comparison.png",
        "title": "Road Segmentation: Dice Score",
        "ylabel": "Dice (F1)",
        "labels": ["Experiment 2", "Experiment 3"],
        "values": [0.987863, 0.996515],
        "note": "Exp. 2: global pixel-level test | Exp. 3: mean per-image test",
    },
]

for item in plots:
    fig, ax = plt.subplots(figsize=(9, 5.5))

    bars = ax.bar(item["labels"], item["values"], width=0.65)

    # Start at zero so the visual difference is not exaggerated.
    ax.set_ylim(0, 1.08)
    ax.set_ylabel(item["ylabel"])
    ax.set_title(item["title"])
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    for bar, value in zip(bars, item["values"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.018,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    fig.text(
        0.5,
        0.025,
        item["note"]
        + "\nDescriptive comparison only: metric aggregation methods differ.",
        ha="center",
        fontsize=8,
        wrap=True,
    )

    fig.tight_layout(rect=[0, 0.09, 1, 1])

    output_path = OUTPUT_DIR / item["filename"]
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path}")

print("\nAll four comparison graphs have been created.")
