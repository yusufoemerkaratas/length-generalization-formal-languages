"""Plot accuracy against input length from the files written by run.py.

Usage: python plot.py results/dyck1

Writes accuracy_vs_length.png (and accuracy_by_kind.png for languages with
"order" negatives) into the same folder. Lines show the mean over seeds; the
shaded band goes from the worst to the best seed.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import yaml  # noqa: E402

# Fixed color and marker per model, so a model looks the same in every plot.
STYLE = {
    "nope_causal": ("#2a78d6", "o", "NoPE, causal"),
    "sinusoidal_causal": ("#eb6834", "s", "sinusoidal, causal"),
    "learned_causal": ("#1baf7a", "^", "learned, causal"),
    "nope_bidirectional": ("#eda100", "D", "NoPE, bidirectional"),
    "lstm": ("#e87ba4", "v", "LSTM"),
}
GRAY = "#6b6a63"
KIND_TITLES = {"all": "all examples", "pos": "positives", "count": "count negatives",
               "order": "order negatives"}


def load(folder: Path):
    """Return {(model, bin, kind): [accuracy per seed]} and the config."""
    cfg = yaml.safe_load((folder / "config_used.yaml").read_text())
    acc = defaultdict(list)
    with open(folder / "metrics.csv") as f:
        for row in csv.DictReader(f):
            acc[(row["model"], row["bin"], row["kind"])].append(float(row["accuracy"]))
    return acc, cfg


def bin_order_and_labels(cfg):
    lo, hi = cfg["data"]["train_range"]
    held = ", ".join(str(n) for n in cfg["data"]["heldout_lengths"])
    bins = ["train", "heldout"] + [f"{a}-{b}" for a, b in cfg["data"]["test_bins"]]
    labels = [f"{lo}-{hi}\n(train)", f"{held}\n(held out)"] + bins[2:]
    return bins, labels


def draw(ax, acc, bins, kind, has_order):
    x = range(len(bins))
    for model, (color, marker, label) in STYLE.items():
        runs = [acc.get((model, b, kind)) for b in bins]
        if not all(runs):
            continue
        mean = [sum(r) / len(r) for r in runs]
        ax.fill_between(x, [min(r) for r in runs], [max(r) for r in runs],
                        color=color, alpha=0.15, linewidth=0)
        ax.plot(x, mean, color=color, marker=marker, markersize=6, linewidth=2, label=label)
    majority = [acc.get(("majority", b, kind)) for b in bins]
    if all(majority):
        ax.plot(x, [sum(r) / len(r) for r in majority], color=GRAY, linestyle="--",
                linewidth=1.5, label="majority class")
    if has_order and kind == "all":
        ax.axhline(0.75, color=GRAY, linestyle=":", linewidth=1.5,
                   label="max. for NoPE, bidirectional")
    # Separate the lengths seen in training from the longer test lengths.
    ax.axvline(1.5, color=GRAY, linewidth=0.8, alpha=0.5)
    ax.set_ylim(-0.03, 1.03)
    ax.grid(axis="y", color="#e4e3dd", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder")
    folder = Path(parser.parse_args().folder)
    acc, cfg = load(folder)
    bins, labels = bin_order_and_labels(cfg)
    has_order = any(kind == "order" for (_, _, kind) in acc)
    name = cfg["language"]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    draw(ax, acc, bins, "all", has_order)
    ax.set_xticks(range(len(bins)), labels)
    ax.set_xlabel("input length")
    ax.set_ylabel("accuracy (mean over seeds)")
    ax.set_title(f"{name}: accuracy vs. input length")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(folder / "accuracy_vs_length.png", dpi=150)
    plt.close(fig)

    if has_order:
        kinds = ["pos", "count", "order"]
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
        for ax, kind in zip(axes, kinds):
            draw(ax, acc, bins, kind, has_order)
            ax.set_xticks(range(len(bins)), labels, fontsize=7)
            ax.set_title(KIND_TITLES[kind])
            ax.set_xlabel("input length")
        axes[0].set_ylabel("accuracy (mean over seeds)")
        handles, names = axes[0].get_legend_handles_labels()
        fig.legend(handles, names, loc="lower center", ncol=len(names), fontsize=8,
                   frameon=False)
        fig.suptitle(f"{name}: accuracy by example kind")
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        fig.savefig(folder / "accuracy_by_kind.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
