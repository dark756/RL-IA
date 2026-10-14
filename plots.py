"""Draws the study's charts from results/test_results.csv and logs/ into plots/, and prints what matters most.

    python plots.py    (or: python driver.py plots)
"""
import glob
import os
import re
from itertools import combinations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from names import lr_tag
from testing import CONDITIONS, SOLVED
from trainer import MAX_STEPS

ROOT = os.path.dirname(os.path.abspath(__file__))
RUN_NAME = re.compile(r"(ppo|sac)_lr([^_]+)_net([0-9x]+)_seed(\d+)$")  # a training log, e.g. sac_lr3e-4_net64x64_seed0

# Colours checked for colour-blind safety; each learning rate also gets its own marker shape
SURFACE, INK, INK2, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
LR_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
LR_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
OUTCOMES = {"landed": "#0ca30c", "timeout": "#eda100", "crashed": "#d03b3b"}
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
SETTINGS = {"algorithm": "algorithm", "lr": "learning rate", "size": "network size",
            "train_episodes": "training episodes"}


def style():
    fonts = {font.name for font in font_manager.fontManager.ttflist}
    family = next((f for f in ["Segoe UI", "Helvetica Neue", "Helvetica", "Arial"] if f in fonts), "DejaVu Sans")
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": family, "font.size": 9, "text.color": INK,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
        "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "legend.frameon": False, "savefig.dpi": 150, "savefig.bbox": "tight",
    })


def header(fig, title, note, handles=None):
    """Title, a short note and the legend, in a band above the charts."""
    height = fig.get_figheight()
    lines = note.count("\n") + 1
    fig.text(0.01, 1 - 0.1 / height, title, ha="left", va="top", fontsize=13, fontweight="semibold")
    fig.text(0.01, 1 - 0.42 / height, note, ha="left", va="top", fontsize=9, color=INK2, linespacing=1.4)
    band = 0.5 + 0.2 * lines
    if handles:
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.003, 1 - band / height),
                   ncol=len(handles), handlelength=2.4, columnspacing=1.8, fontsize=9)
        band += 0.35
    fig.subplots_adjust(top=1 - (band + 0.3) / height)


def short(episodes):
    return f"{episodes // 1000}k" if episodes >= 1000 and episodes % 1000 == 0 else str(episodes)


def width(size):
    return int(str(size).split("x")[0])


def per_model(results):
    """One row per model with its mean return over the given landings, and the % of them that landed."""
    keys = ["model", "algorithm", "lr", "size", "seed", "train_episodes"]
    landed = (results["outcome"] == "landed") * 100.0
    return results.assign(landed=landed).groupby(keys, as_index=False)[["return", "landed"]].mean()


def read_logs(root, window):
    """Every training landing in logs/, plus the mean return and length of the last `window` landings of its run."""
    frames = []
    for path in sorted(glob.glob(os.path.join(root, "logs", "*.csv"))):
        match = RUN_NAME.match(os.path.splitext(os.path.basename(path))[0])
        if not match:
            continue
        algo, lr, size, seed = match.groups()
        log = pd.read_csv(path)
        frames.append(log.assign(algorithm=algo, lr=float(lr), size=size, seed=int(seed),
                                 return_mean=log["return"].rolling(window).mean(),
                                 length_mean=log["length"].rolling(window).mean()))
    return pd.concat(frames, ignore_index=True) if frames else None


def levels(models):
    algos = [algo for algo in ["ppo", "sac"] if algo in set(models["algorithm"])]
    sizes = sorted(models["size"].unique(), key=width)
    return algos, sizes, sorted(models["lr"].unique()), sorted(models["train_episodes"].unique())


def variance_shares(models):
    """How much of the spread in mean test return between models each setting explains (ANOVA eta squared).
    Whatever no setting or combination of settings explains is the luck of the seed."""
    y = models["return"]
    grand = y.mean()
    total = ((y - grand) ** 2).sum() or 1.0
    shares, effects = {}, {}
    for factor, label in SETTINGS.items():
        effects[factor] = models.groupby(factor)["return"].transform("mean") - grand
        shares[label] = (effects[factor] ** 2).sum() / total
    for a, b in combinations(SETTINGS, 2):
        pair = models.groupby([a, b])["return"].transform("mean") - grand - effects[a] - effects[b]
        shares[f"{SETTINGS[a]} x {SETTINGS[b]}"] = (pair ** 2).sum() / total
    cells = models.groupby(list(SETTINGS))["return"].transform("mean")
    shares["3 or 4 settings combined"] = max(0.0, ((cells - grand) ** 2).sum() / total - sum(shares.values()))
    shares["seed (luck)"] = ((y - cells) ** 2).sum() / total
    return pd.Series(shares) * 100


def reference_line(ax, value, text, label=False):
    """A dashed line such as the solved score or the time limit."""
    ax.axhline(value, color=INK2, linewidth=0.8, linestyle=(0, (4, 3)), zorder=1)
    if label:  # in the margin to the right of the panel, where no line can run into it
        ax.text(1.02, value, text, transform=ax.get_yaxis_transform(), ha="left", va="center",
                fontsize=8, color=INK2)


def lr_handles(colors, lrs):
    return [Line2D([], [], color=colors[lr][0], marker=colors[lr][1], linewidth=2, markersize=5,
                   markeredgecolor=SURFACE, label=f"lr {lr_tag(lr)}") for lr in lrs]


def draw_lines(ax, runs, x_column, x_values, colors, lrs, y="return"):
    """One line per learning rate (mean of the seeds) with a faint band from the worst to the best seed."""
    for lr in lrs:
        stats = runs[runs["lr"] == lr].groupby(x_column)[y].agg(["mean", "min", "max"]).reindex(x_values)
        if stats["mean"].isna().all():
            continue
        x = [width(v) for v in x_values] if x_column == "size" else x_values
        color, marker = colors[lr]
        ax.fill_between(x, stats["min"], stats["max"], color=color, alpha=0.1, linewidth=0)
        ax.plot(x, stats["mean"], color=color, marker=marker, linewidth=2, markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1)


def plot_importance(shares, path):
    shares = shares.sort_values()
    fig, ax = plt.subplots(figsize=(8, 0.3 * len(shares) + 1.6))
    colors = [LR_COLORS[0] if name in SETTINGS.values() else AXIS for name in shares.index]
    ax.barh(shares.index, shares.values, height=0.6, color=colors)
    for y, value in enumerate(shares.values):
        ax.text(value + shares.max() * 0.01, y, f"{value:.1f}%" if value < 10 else f"{value:.0f}%",
                va="center", fontsize=8.5, color=INK2)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("share of the spread in mean test return between models (%)")
    fig.subplots_adjust(left=0.34)  # room for the long labels, so the title lines up with them
    header(fig, "Which settings matter most",
           "Blue: a setting on its own. Gray: settings combined, and the luck of the seed.\n"
           "Each model's return is its mean over every test landing.")
    fig.savefig(path)
    plt.close(fig)


def plot_importance_by_condition(table, path):
    fig, ax = plt.subplots(figsize=(1.05 * table.shape[1] + 2.4, 0.42 * table.shape[0] + 1.8))
    ax.imshow(table.values, cmap=BLUES, vmin=0, vmax=100, aspect="auto")
    for (i, j), value in np.ndenumerate(table.values):
        red, green, blue, _ = BLUES(value / 100)
        ink = "white" if 0.2126 * red + 0.7152 * green + 0.0722 * blue < 0.5 else INK
        ax.text(j, i, f"{value:.0f}%", ha="center", va="center", fontsize=9, color=ink)
    ax.set_xticks(range(table.shape[1]), [c.replace("_", " ") for c in table.columns])
    ax.set_yticks(range(table.shape[0]), table.index)
    ax.set_xticks(np.arange(-0.5, table.shape[1]), minor=True)
    ax.set_yticks(np.arange(-0.5, table.shape[0]), minor=True)
    ax.grid(which="major", visible=False)
    ax.grid(which="minor", axis="both", color=SURFACE, linewidth=3)
    ax.tick_params(which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    header(fig, "Which settings matter most, per test condition",
           "Share of the spread in mean test return between models, tested on each condition's planets.")
    fig.savefig(path)
    plt.close(fig)


def plot_curves(results, colors, path, title, note, y="return"):
    """Mean test return (y="return") or % landed (y="landed") against training episodes: one panel per algorithm
    and network size, one line per learning rate."""
    models = per_model(results)
    algos, sizes, lrs, episodes = levels(models)
    fig, axes = plt.subplots(len(algos), len(sizes), figsize=(3.2 * len(sizes) + 0.8, 2.5 * len(algos) + 1.8),
                             sharex=True, sharey=True, squeeze=False)
    for r, algo in enumerate(algos):
        for c, size in enumerate(sizes):
            ax = axes[r][c]
            if y == "return":
                reference_line(ax, SOLVED, "solved", label=(r == 0 and c == len(sizes) - 1))
            runs = models[(models["algorithm"] == algo) & (models["size"] == size)]
            draw_lines(ax, runs, "train_episodes", episodes, colors, lrs, y)
            ax.set_xscale("log", base=2)
            ax.set_xticks(episodes, [short(e) for e in episodes])
            ax.minorticks_off()
            ax.set_title(f"{algo.upper()}, {size} network", loc="left")
            if r == len(algos) - 1:
                ax.set_xlabel("training episodes")
            if c == 0:
                ax.set_ylabel("mean test return" if y == "return" else "% of test landings that landed")
    if y == "landed":
        axes[0][0].set_ylim(0, 100)  # the panels share the y axis
    header(fig, title, note, lr_handles(colors, lrs))
    fig.savefig(path)
    plt.close(fig)


def plot_sizes(results, colors, path):
    """Return against network size: one panel per algorithm and training stage, one line per learning rate."""
    models = per_model(results)
    algos, sizes, lrs, episodes = levels(models)
    fig, axes = plt.subplots(len(algos), len(episodes), figsize=(2.1 * len(episodes) + 0.8, 2.5 * len(algos) + 1.8),
                             sharex=True, sharey=True, squeeze=False)
    for r, algo in enumerate(algos):
        for c, stage in enumerate(episodes):
            ax = axes[r][c]
            reference_line(ax, SOLVED, "solved", label=(r == 0 and c == len(episodes) - 1))
            runs = models[(models["algorithm"] == algo) & (models["train_episodes"] == stage)]
            draw_lines(ax, runs, "size", sizes, colors, lrs)
            ax.set_xscale("log", base=2)
            ax.set_xticks([width(s) for s in sizes], [str(width(s)) for s in sizes])
            ax.minorticks_off()
            ax.set_title(f"{algo.upper()}, {short(stage)} episodes", loc="left")
            if r == len(algos) - 1:
                ax.set_xlabel("neurons per layer")
            if c == 0:
                ax.set_ylabel("mean test return")
    header(fig, "Test return against network size",
           "Mean over every test landing. Line: mean of the seeds. Band: worst to best seed.",
           lr_handles(colors, lrs))
    fig.savefig(path)
    plt.close(fig)


def plot_outcomes(results, path):
    """How the landings ended, per algorithm, condition and training stage (all learning rates, sizes and seeds)."""
    algos, _, _, episodes = levels(results)
    conditions = [c for c in CONDITIONS if c in set(results["condition"])]
    shares = (results.groupby(["algorithm", "condition", "train_episodes"])["outcome"]
              .value_counts(normalize=True).unstack(fill_value=0) * 100)
    fig, axes = plt.subplots(len(algos), len(conditions), figsize=(2.2 * len(conditions) + 0.8, 2.3 * len(algos) + 1.8),
                             sharex=True, sharey=True, squeeze=False)
    x = np.arange(len(episodes))
    for r, algo in enumerate(algos):
        for c, condition in enumerate(conditions):
            ax = axes[r][c]
            bottom = np.zeros(len(episodes))
            for outcome, color in OUTCOMES.items():
                values = np.array([shares[outcome].get((algo, condition, e), 0) if outcome in shares else 0
                                   for e in episodes])
                ax.bar(x, values, bottom=bottom, width=0.55, color=color, edgecolor=SURFACE, linewidth=1)
                bottom += values
            ax.set_ylim(0, 100)
            ax.set_xticks(x, [short(e) for e in episodes], fontsize=8)
            ax.set_title(f"{algo.upper()}, {condition.replace('_', ' ')}", loc="left", fontsize=9)
            if r == len(algos) - 1:
                ax.set_xlabel("training episodes")
            if c == 0:
                ax.set_ylabel("% of test landings")
    handles = [Patch(color=color, label=outcome if outcome != "timeout" else "ran out of time")
               for outcome, color in OUTCOMES.items()]
    header(fig, "How the test landings ended",
           "All learning rates, network sizes and seeds together. Bars stack from the bottom in legend order.",
           handles)
    fig.savefig(path)
    plt.close(fig)


def plot_training(logs, y, colors, episodes, path, title, note):
    """Training return (y="return") or steps per landing (y="length") against training episodes, from logs/: one
    panel per algorithm and network size, one line per learning rate (mean of the seeds) with a faint band from the
    worst to the best seed. Markers sit at the stages, so each learning rate keeps its marker shape."""
    algos = [algo for algo in ["ppo", "sac"] if algo in set(logs["algorithm"])]
    sizes = sorted(logs["size"].unique(), key=width)
    lrs = sorted(logs["lr"].unique())
    fig, axes = plt.subplots(len(algos), len(sizes), figsize=(3.2 * len(sizes) + 0.8, 2.5 * len(algos) + 1.8),
                             sharex=True, sharey=True, squeeze=False)
    for r, algo in enumerate(algos):
        for c, size in enumerate(sizes):
            ax = axes[r][c]
            label = r == 0 and c == len(sizes) - 1
            if y == "return":
                reference_line(ax, SOLVED, "solved", label)
            else:
                reference_line(ax, MAX_STEPS, "time limit", label)
            runs = logs[(logs["algorithm"] == algo) & (logs["size"] == size)]
            for lr in lrs:
                stats = runs[runs["lr"] == lr].groupby("episode")[f"{y}_mean"].agg(["mean", "min", "max"]).dropna()
                if stats.empty:
                    continue
                color, marker = colors[lr]
                ax.fill_between(stats.index, stats["min"], stats["max"], color=color, alpha=0.1, linewidth=0)
                ax.plot(stats.index, stats["mean"], color=color, linewidth=1.6, marker=marker, markersize=5,
                        markevery=[stats.index.get_loc(e) for e in episodes if e in stats.index],
                        markeredgecolor=SURFACE, markeredgewidth=1)
            ax.set_xscale("log", base=2)
            ax.set_xticks(episodes, [short(e) for e in episodes])
            ax.minorticks_off()
            ax.set_title(f"{algo.upper()}, {size} network", loc="left")
            if r == len(algos) - 1:
                ax.set_xlabel("training episodes")
            if c == 0:
                ax.set_ylabel("training return" if y == "return" else "steps per training landing")
    if y == "length":
        axes[0][0].set_ylim(0, MAX_STEPS * 1.05)  # the panels share the y axis
    header(fig, title, note, lr_handles(colors, lrs))
    fig.savefig(path)
    plt.close(fig)


def peaks(models):
    """For each algorithm, learning rate and size: the stage with the best mean test return, and the drop after it."""
    curve = models.groupby(["algorithm", "lr", "size", "train_episodes"], as_index=False)["return"].mean()
    rows = []
    for (algo, lr, size), group in curve.groupby(["algorithm", "lr", "size"]):
        best = group.loc[group["return"].idxmax()]
        last = group.loc[group["train_episodes"].idxmax()]
        rows.append(dict(algorithm=algo, lr=lr_tag(lr), size=size, best_episodes=int(best["train_episodes"]),
                         best_return=round(best["return"], 1), last_episodes=int(last["train_episodes"]),
                         last_return=round(last["return"], 1), drop_after_best=round(best["return"] - last["return"], 1)))
    table = pd.DataFrame(rows)
    table["order"] = table["size"].map(width)
    return table.sort_values(["algorithm", "order", "lr"]).drop(columns="order")


def make_plots(root=ROOT):
    path = os.path.join(root, "results", "test_results.csv")
    if not os.path.exists(path):
        print(f"No results yet: run 'python driver.py test --log' first ({path} is missing)")
        return
    results = pd.read_csv(path).drop_duplicates(["model", "env"], keep="last")
    folder = os.path.join(root, "plots")
    os.makedirs(os.path.join(folder, "conditions"), exist_ok=True)
    style()
    lrs = sorted(results["lr"].unique())
    colors = {lr: (LR_COLORS[i], LR_MARKERS[i]) for i, lr in enumerate(lrs)}
    models = per_model(results)
    landings = results.groupby("model").size().iloc[0]

    shares = variance_shares(models)
    plot_importance(shares, os.path.join(folder, "1_importance.png"))

    conditions = [c for c in CONDITIONS if c in set(results["condition"])]
    by_condition = {c: variance_shares(per_model(results[results["condition"] == c])) for c in conditions}
    by_condition["all"] = shares
    table = pd.DataFrame(by_condition)
    combined = [name for name in table.index if name not in SETTINGS.values() and name != "seed (luck)"]
    table = pd.concat([table.loc[list(SETTINGS.values())],
                       table.loc[combined].sum().to_frame("settings combined").T, table.loc[["seed (luck)"]]])
    plot_importance_by_condition(table, os.path.join(folder, "2_importance_by_condition.png"))

    plot_curves(results, colors, os.path.join(folder, "3_return_vs_episodes.png"),
                "Test return against training episodes",
                f"Mean over all {landings} test landings. Line: mean of the seeds. Band: worst to best seed.")
    plot_sizes(results, colors, os.path.join(folder, "4_return_vs_size.png"))
    plot_outcomes(results, os.path.join(folder, "5_outcomes.png"))
    for condition in conditions:
        subset = results[results["condition"] == condition]
        plot_curves(subset, colors, os.path.join(folder, "conditions", f"{condition}.png"),
                    f"Test return against training episodes: {condition.replace('_', ' ')}",
                    f"Mean over this condition's {subset.groupby('model').size().iloc[0]} test landings. "
                    "Line: mean of the seeds. Band: worst to best seed.")
    plot_curves(results, colors, os.path.join(folder, "6_landed_vs_episodes.png"),
                "Landing success rate against training episodes",
                f"Share of all {landings} test landings that came to rest. Line: mean of the seeds. "
                "Band: worst to best seed.", y="landed")

    episodes = levels(models)[3]
    window = min(100, episodes[0])
    logs = read_logs(root, window)
    if logs is None:
        print(f"No training logs in {os.path.join(root, 'logs')}, so charts 7 and 8 are skipped")
    else:
        logs = logs[logs["lr"].isin(colors)]
        planets = "Training landings: a new random planet each time, with exploration on."
        points = (f"Each point is the mean of the last {window} landings. "
                  "Line: mean of the seeds. Band: worst to best seed.")
        plot_training(logs, "return", colors, episodes, os.path.join(folder, "7_training_return.png"),
                      "Training return against training episodes", f"{planets}\n{points}")
        plot_training(logs, "length", colors, episodes, os.path.join(folder, "8_training_steps.png"),
                      "Steps per training landing against training episodes",
                      f"{planets} {MAX_STEPS} steps means it ran out of time.\n{points}")

    peak_table = peaks(models)
    peak_table.to_csv(os.path.join(root, "results", "peaks.csv"), index=False)
    best = (models.groupby(["algorithm", "lr", "size", "train_episodes"], as_index=False)["return"]
            .agg(["mean", "min", "max"]).sort_values("mean", ascending=False).head(10))
    pd.set_option("display.width", 160)
    print("\nWhat matters most (% of the spread in mean test return between models):")
    print(shares.sort_values(ascending=False).round(1).to_string())
    print("\nBest settings (mean test return over all test landings; mean, worst and best seed):")
    print(best.assign(lr=best["lr"].map(lr_tag)).round(1).to_string(index=False))
    print("\nWhere each setting peaks, and how much it drops by the last stage (saved to results/peaks.csv):")
    print(peak_table.to_string(index=False))
    print(f"\nCharts saved to {folder}")


if __name__ == "__main__":
    make_plots()
