"""Draws the report's two diagrams into plots/: how the agent learns, and how the study was run.

    python diagram.py

The numbers come from the study's own settings and SB3's defaults, so the diagrams match the code.
"""
import inspect
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from gymnasium.envs.box2d.lunar_lander import FPS
from matplotlib.patches import FancyBboxPatch
from stable_baselines3 import PPO, SAC

from driver import ALGOS, LEARNING_RATES, SEEDS, SIZES, STAGES
from plots import AXIS, INK, INK2, ROOT, header, short, style
from testing import CONDITIONS, ENVS_PER_CONDITION
from trainer import GRAVITY_MEAN, GRAVITY_SPREAD, TURBULENCE_SPREAD, WIND_SPREAD

# (edge, fill) of each kind of box
BLUE, GREEN, AMBER, GRAY = ("#2a78d6", "#e8f1fc"), ("#1baf7a", "#e5f6ef"), ("#eda100", "#fdf3df"), (AXIS, "#f3f2ee")
HEADER = 0.8  # inches that header() takes for the title and a one-line note


def default(algo, name):
    """An SB3 default, e.g. default(PPO, "n_steps") -> 2048. The study only changes the learning rate and size."""
    return inspect.signature(algo.__init__).parameters[name].default


def canvas(width, height, title, note):
    """A figure under the usual header, whose drawing area is measured in inches from its bottom left corner."""
    fig = plt.figure(figsize=(width, height + HEADER))
    ax = fig.add_axes((0, 0, 1, height / (height + HEADER)))
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")
    header(fig, title, note)
    return fig, ax


def box(ax, x, y, w, h, colors, title, lines):
    edge, fill = colors
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12",
                                facecolor=fill, edgecolor=edge, linewidth=1.4))
    ax.text(x + w / 2, y + h - 0.28, title, ha="center", va="center", fontsize=11, fontweight="semibold", color=INK)
    ax.text(x + w / 2, y + h - 0.5, "\n".join(lines), ha="center", va="top", fontsize=9.5, color=INK2,
            linespacing=1.45)


def arrow(ax, *points):
    """Straight segments through the points, with the arrowhead on the last one."""
    if len(points) > 2:
        ax.plot([p[0] for p in points[:-1]], [p[1] for p in points[:-1]], color=INK2, linewidth=1.3)
    ax.annotate("", xy=points[-1], xytext=points[-2],
                arrowprops=dict(arrowstyle="-|>", color=INK2, linewidth=1.3, mutation_scale=14, shrinkA=0, shrinkB=0))


def label(ax, x, y, text, ha="center", va="center"):
    ax.text(x, y, text, ha=ha, va=va, fontsize=9.5, color=INK2, linespacing=1.4)


def draw_loop(path):
    """The template's loop: environment, state, agent, action, reward, state update, learning."""
    fig, ax = canvas(9.5, 6.3, "How the agent learns to land",
                     "Every step, the agent reads the state and acts; "
                     "the reward and the next state feed the learning step.")
    sizes = ["x".join(map(str, size)) for size in SIZES]
    box(ax, 6.2, 4.8, 3.0, 1.3, GRAY, "Planet draw",
        [f"gravity ~ N({GRAVITY_MEAN:g}, {GRAVITY_SPREAD:g})", f"wind ~ |N(0, {WIND_SPREAD:g})|",
         f"turbulence ~ |N(0, {TURBULENCE_SPREAD:g})|"])
    box(ax, 6.2, 2.5, 3.0, 1.5, GREEN, "Environment",
        ["LunarLanderContinuous-v3", f"Box2D physics, {FPS} steps a second"])
    box(ax, 0.3, 2.5, 2.7, 1.5, BLUE, "RL agent",
        [r"policy network $\pi(a\,|\,s)$", "PPO or SAC", f"hidden layers {sizes[0]} to {sizes[-1]}"])
    box(ax, 6.2, 0.2, 3.0, 1.5, AMBER, "Learning",
        [f"PPO: every {default(PPO, 'n_steps')} steps,", f"{default(PPO, 'n_epochs')} epochs of clipped updates",
         "SAC: replay buffer,", f"{default(SAC, 'gradient_steps')} gradient step per step"])

    arrow(ax, (7.7, 4.8), (7.7, 4.0))
    label(ax, 7.85, 4.4, "before each landing", ha="left")
    arrow(ax, (6.2, 3.55), (3.0, 3.55))
    label(ax, 4.6, 3.67, "state $s_t$: position, velocity,\nangle, spin, leg contacts", va="bottom")
    arrow(ax, (3.0, 2.95), (6.2, 2.95))
    label(ax, 4.6, 2.83, "action $a_t$: main engine\nand side engines", va="top")
    arrow(ax, (7.7, 2.5), (7.7, 1.7))
    label(ax, 7.85, 2.1, "reward $r_t$,\nnext state $s_{t+1}$", ha="left")
    arrow(ax, (6.2, 0.95), (1.65, 0.95), (1.65, 2.5))
    label(ax, 3.9, 1.07, "updated weights", va="bottom")
    fig.savefig(path)
    plt.close(fig)


def draw_pipeline(path):
    """The study as a chain: settings, training, testing, analysis."""
    runs = len(ALGOS) * len(LEARNING_RATES) * len(SIZES) * len(SEEDS)
    models = runs * len(STAGES)
    planets = len(CONDITIONS) * ENVS_PER_CONDITION
    fig, ax = canvas(10.55, 1.9, "How the study was run",
                     f"Every model is tested on the same {planets} planets, so all {models} can be compared fairly.")
    boxes = [
        (GRAY, "Study design", [f"{len(ALGOS)} algorithms", f"x {len(LEARNING_RATES)} learning rates",
                                f"x {len(SIZES)} network sizes", f"x {len(SEEDS)} seeds"]),
        (AMBER, "Training", ["random planet per landing;", "weights saved at",
                             ", ".join(short(s) for s in STAGES[:3]) + ",",
                             ", ".join(short(s) for s in STAGES[3:-1]) + f" and {short(STAGES[-1])} episodes"]),
        (GREEN, "Testing", [f"{planets} fixed planets", f"({len(CONDITIONS)} conditions x {ENVS_PER_CONDITION}),",
                            "1 deterministic landing", "on each"]),
        (BLUE, "Analysis", ["return and % landed", "importance (ANOVA)", "drop-offs (peaks.csv)", "charts in plots/"]),
    ]
    flows = [f"{runs}\nruns", f"{models}\nmodels", f"{models * planets:,}\nlandings"]
    w, gap = 2.0, 0.75
    for i, (colors, title, lines) in enumerate(boxes):
        x = 0.15 + i * (w + gap)
        box(ax, x, 0.2, w, 1.5, colors, title, lines)
        if i < len(flows):
            arrow(ax, (x + w, 0.95), (x + w + gap, 0.95))
            label(ax, x + w + gap / 2, 1.05, flows[i], va="bottom")
    fig.savefig(path)
    plt.close(fig)


def make_diagrams(root=ROOT):
    folder = os.path.join(root, "plots")
    os.makedirs(folder, exist_ok=True)
    style()
    draw_loop(os.path.join(folder, "diagram_rl_loop.png"))
    draw_pipeline(os.path.join(folder, "diagram_pipeline.png"))
    print(f"Diagrams saved to {folder}")


if __name__ == "__main__":
    make_diagrams()
