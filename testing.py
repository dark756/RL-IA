"""Tests saved landers on the same fixed test planets, and prints or logs the results.

    test(model_path)               prints a table per condition
    test(model_path, log=True)     also adds every landing to results/test_results.csv
                                   and refreshes results/summary.csv
    test(model_path, render=True)  lets you watch the landings
"""
import argparse
import glob
import json
import os
import re
import time

import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO, SAC

ROOT = os.path.dirname(os.path.abspath(__file__))
ALGOS = {"ppo": PPO, "sac": SAC}
ENVS_PER_CONDITION = 50
TEST_SEED = 2026  # fixes the test planets: the same seed always writes the same files
SOLVED = 200  # gymnasium counts a landing scoring 200 or more as solved

# condition: (gravity, wind, turbulence) ranges; the test planets are spread evenly inside them
CONDITIONS = {
    "ideal": ((-10.0, -10.0), (0.0, 0.0), (0.0, 0.0)),
    "moderate_wind": ((-10.0, -10.0), (5.0, 10.0), (0.5, 1.0)),
    "heavy_wind": ((-10.0, -10.0), (15.0, 20.0), (1.5, 2.0)),
    "heavy_gravity": ((-11.9, -11.0), (0.0, 0.0), (0.0, 0.0)),
    "low_gravity": ((-4.0, -2.0), (0.0, 0.0), (0.0, 0.0)),
    "random_mix": ((-11.9, -2.0), (0.0, 20.0), (0.0, 2.0)),
}
MODEL_NAME = re.compile(r"(ppo|sac)_lr([^_]+)_net([0-9x]+)_seed(\d+)_ep(\d+)")


def make_envs(per_condition=ENVS_PER_CONDITION, root=ROOT):
    """Writes the test planets to resources/envs/, one JSON file each. The seed in each name
    fixes that landing's terrain, starting push and wind pattern."""
    folder = os.path.join(root, "resources", "envs")
    os.makedirs(folder, exist_ok=True)
    for old in glob.glob(os.path.join(folder, "*.json")):
        os.remove(old)
    rng = np.random.default_rng(TEST_SEED)
    seeds = rng.choice(1_000_000, size=len(CONDITIONS) * per_condition, replace=False)
    for c, (condition, (gravity, wind, turbulence)) in enumerate(CONDITIONS.items()):
        for i in range(per_condition):
            env = dict(condition=condition,
                       gravity=round(float(rng.uniform(*gravity)), 2),
                       wind=round(float(rng.uniform(*wind)), 1),
                       turbulence=round(float(rng.uniform(*turbulence)), 2),
                       seed=int(seeds[c * per_condition + i]))
            env["name"] = (f"{condition}_{i + 1:02d}_g{env['gravity']:.2f}_wind{env['wind']:.1f}"
                           f"_turb{env['turbulence']:.2f}_seed{env['seed']}")
            with open(os.path.join(folder, env["name"] + ".json"), "w", encoding="utf-8") as f:
                json.dump(env, f, indent=1)
    print(f"wrote {len(seeds)} test planets to {folder}", flush=True)


def load_envs(root=ROOT, conditions=None):
    """Reads the test planets (writing them first if they don't exist yet)."""
    files = sorted(glob.glob(os.path.join(root, "resources", "envs", "*.json")))
    if not files:
        make_envs(root=root)
        files = sorted(glob.glob(os.path.join(root, "resources", "envs", "*.json")))
    envs = []
    for file in files:
        with open(file, encoding="utf-8") as f:
            envs.append(json.load(f))
    return [env for env in envs if not conditions or env["condition"] in conditions]


def model_info(model_path):
    """Reads the settings from a model's file name, e.g. sac_lr3e-4_net256x256_seed0_ep1000.zip"""
    name = os.path.basename(model_path).removesuffix(".zip")
    algo, lr, size, seed, episodes = MODEL_NAME.fullmatch(name).groups()
    return dict(model=name, algorithm=algo, lr=float(lr), size=size, seed=int(seed), train_episodes=int(episodes))


def land(model, planets, render=False):
    """Gives the model one deterministic landing on each planet. All landers fly side by side,
    so the model picks every lander's action in one call."""
    envs = [gym.make("LunarLanderContinuous-v3", render_mode="human" if render else None,
                     gravity=p["gravity"], enable_wind=True, wind_power=p["wind"],
                     turbulence_power=p["turbulence"]) for p in planets]
    obs = np.array([env.reset(seed=p["seed"])[0] for env, p in zip(envs, planets)])
    returns, steps, outcomes = np.zeros(len(envs)), np.zeros(len(envs), dtype=int), [""] * len(envs)
    flying = list(range(len(envs)))
    while flying:
        actions, _states = model.predict(obs[flying], deterministic=True)
        still_flying = []
        for i, action in zip(flying, actions):
            obs[i], reward, te, tr, info = envs[i].step(action)
            returns[i] += reward
            steps[i] += 1
            if te:  # the last reward is +100 for coming to rest and -100 for a crash or flying off screen
                outcomes[i] = "landed" if reward > 0 else "crashed"
            elif tr:
                outcomes[i] = "timeout"
            if te or tr:
                envs[i].close()
            else:
                still_flying.append(i)
        flying = still_flying
    return [{"env": p["name"], "condition": p["condition"], "gravity": p["gravity"], "wind": p["wind"],
             "turbulence": p["turbulence"], "return": round(float(r), 2), "steps": int(s), "outcome": o}
            for p, r, s, o in zip(planets, returns, steps, outcomes)]


def test(model_path, log=False, render=False, conditions=None, root=ROOT):
    """Tests one saved model on every test planet (or only the given conditions), prints a table
    per condition and returns the landings. log=True also saves them to results/."""
    info = model_info(model_path)
    model = ALGOS[info["algorithm"]].load(model_path, device="cpu")
    planets = load_envs(root, conditions)
    if render:  # one at a time, so you can watch each landing
        landings = [row for planet in planets for row in land(model, [planet], render=True)]
    else:
        landings = land(model, planets)
    results = pd.DataFrame([{**info, "train_steps": model.num_timesteps, **row} for row in landings])
    results["solved"] = results["return"] >= SOLVED
    print(table(results), flush=True)
    if log:
        save(results, root)
        summarize(root)
    return results


def table(results):
    """One model's results per condition, as text."""
    first = results.iloc[0]
    lines = [f"{first['model']}  (trained {first['train_episodes']} episodes, {first['train_steps']:,} steps)",
             f"  {'condition':<15}{'landings':>9}{'mean return':>13}{'solved':>8}{'landed':>8}{'crashed':>9}{'timeout':>9}"]
    present = [c for c in CONDITIONS if c in set(results["condition"])]
    for condition in present + ["all"]:
        group = results if condition == "all" else results[results["condition"] == condition]
        share = lambda outcome: f"{(group['outcome'] == outcome).mean():.0%}"
        lines.append(f"  {condition:<15}{len(group):>9}{group['return'].mean():>13.1f}{group['solved'].mean():>8.0%}"
                     f"{share('landed'):>8}{share('crashed'):>9}{share('timeout'):>9}")
    return "\n".join(lines) + "\n"


def write_csv(frame, path, **kwargs):
    """Writes a CSV, waiting while another program (e.g. Excel) has the file open."""
    while True:
        try:
            frame.to_csv(path, **kwargs)
            return
        except PermissionError:
            print(f"{path} is open in another program (Excel?) - close it and I'll carry on", flush=True)
            time.sleep(15)


def save(results, root=ROOT):
    """Adds one model's landings to results/test_results.csv."""
    path = os.path.join(root, "results", "test_results.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_csv(results, path, mode="a", header=not os.path.exists(path), index=False)


def summarize(root=ROOT):
    """Rewrites results/summary.csv: one row per model and condition, plus 'all'."""
    path = os.path.join(root, "results", "test_results.csv")
    results = pd.read_csv(path)
    latest = results.drop_duplicates(["model", "env"], keep="last")  # a retested model keeps its newest landings
    if len(latest) < len(results):
        write_csv(latest, path, index=False)
    both = pd.concat([latest, latest.assign(condition="all")])
    both = both.assign(landed=both["outcome"] == "landed", crashed=both["outcome"] == "crashed",
                       timeout=both["outcome"] == "timeout")
    keys = ["model", "algorithm", "lr", "size", "seed", "train_episodes", "train_steps", "condition"]
    summary = both.groupby(keys, sort=False).agg(
        landings=("return", "size"), mean_return=("return", "mean"), std_return=("return", "std"),
        solved_pct=("solved", "mean"), landed_pct=("landed", "mean"), crashed_pct=("crashed", "mean"),
        timeout_pct=("timeout", "mean"), mean_steps=("steps", "mean")).reset_index()
    for column in ["solved_pct", "landed_pct", "crashed_pct", "timeout_pct"]:
        summary[column] = (summary[column] * 100).round(1)
    for column in ["mean_return", "std_return", "mean_steps"]:
        summary[column] = summary[column].round(2)
    summary["condition"] = pd.Categorical(summary["condition"], list(CONDITIONS) + ["all"], ordered=True)
    summary = summary.sort_values(["algorithm", "lr", "size", "seed", "train_episodes", "condition"],
                                  key=lambda c: c.str.split("x").str[0].astype(int) if c.name == "size" else c)
    write_csv(summary, os.path.join(root, "results", "summary.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test one saved model on the test planets.")
    parser.add_argument("model", help="e.g. resources/sac/sac_lr3e-4_net256x256_seed0_ep4000.zip")
    parser.add_argument("--log", action="store_true", help="also save the results to results/")
    parser.add_argument("--render", action="store_true", help="watch the landings")
    parser.add_argument("--conditions", nargs="+", choices=list(CONDITIONS), help="only these conditions")
    args = parser.parse_args()
    test(args.model, log=args.log, render=args.render, conditions=args.conditions)
