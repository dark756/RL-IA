"""Trains one PPO or SAC lander and saves its weights at every training-episode stage.

Every training landing happens on a new planet drawn from bell curves around the ideal
planet (gravity -10, no wind). The planets come from a generator seeded with the run's
seed, so every model with the same seed meets the same planets in the same order.
"""
import csv
import json
import os
import time
from collections import deque

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback

from names import ROOT, finished, model_name, stage_path

ALGOS = {"ppo": PPO, "sac": SAC}
MAX_STEPS = 1000  # LunarLander ends a landing after 1000 steps

# Training planets: bell curves centred on the ideal planet
GRAVITY_MEAN, GRAVITY_SPREAD = -10.0, 1.0
WIND_SPREAD = 7.0  # half bell curve starting at 0 (no wind)
TURBULENCE_SPREAD = 0.7  # half bell curve starting at 0 (no turbulence)
# A draw outside the ranges the test planets use is drawn again
GRAVITY_LIMITS = (-11.9, -2.0)
WIND_LIMITS = (0.0, 20.0)
TURBULENCE_LIMITS = (0.0, 2.0)


class GaussPlanetWrapper(gym.Wrapper):
    """Like ChaosPlanetWrapper in lunar_random.py, but the planets come from bell curves and a seeded
    generator. It only sets gravity, wind and turbulence before each landing; nothing pushes the lander."""

    def __init__(self, env, seed):
        super().__init__(env)
        self.rng = np.random.default_rng(seed)
        self.rng_state = self.rng.bit_generator.state

    def reset(self, **kwargs):
        self.rng_state = self.rng.bit_generator.state  # saved at each stage, so a run carries on from this planet
        self.gravity = draw(lambda: self.rng.normal(GRAVITY_MEAN, GRAVITY_SPREAD), GRAVITY_LIMITS)
        self.wind = draw(lambda: abs(self.rng.normal(0.0, WIND_SPREAD)), WIND_LIMITS)
        self.turbulence = draw(lambda: abs(self.rng.normal(0.0, TURBULENCE_SPREAD)), TURBULENCE_LIMITS)
        self.env.unwrapped.gravity = self.gravity
        self.env.unwrapped.wind_power = self.wind
        self.env.unwrapped.turbulence_power = self.turbulence
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, te, tr, info = self.env.step(action)
        if te or tr:  # tells the training log which planet this landing was on
            info["planet"] = (self.gravity, self.wind, self.turbulence)
        return obs, reward, te, tr, info


def draw(sample, limits):
    value = sample()
    while not limits[0] <= value <= limits[1]:
        value = sample()
    return float(value)


class StageSaver(BaseCallback):
    """Counts finished landings, logs each one, saves weights and progress at every stage, stops after the last."""

    def __init__(self, name, folder, stages, episodes, log, save_progress):
        super().__init__()
        self.name, self.folder, self.stages, self.log = name, folder, stages, log
        self.save_progress = save_progress
        self.episodes = episodes  # landings finished so far
        self.recent = deque(maxlen=100)
        self.start = time.time()

    def _on_step(self):
        for info in self.locals["infos"]:
            if "episode" not in info:
                continue
            self.episodes += 1
            ret, length = info["episode"]["r"], info["episode"]["l"]
            gravity, wind, turbulence = info["planet"]
            self.log.writerow([self.episodes, self.num_timesteps, round(ret, 2), length,
                               round(gravity, 3), round(wind, 3), round(turbulence, 3)])
            self.recent.append(ret)
            if self.episodes in self.stages:
                self.model.save(os.path.join(self.folder, f"{self.name}_ep{self.episodes}"))
                self.save_progress(self.episodes)
                print(f"{self.name}  episode {self.episodes:>5}  steps {self.num_timesteps:>9,}  "
                      f"last 100 landings {np.mean(self.recent):7.1f}  {(time.time() - self.start) / 60:6.1f} min",
                      flush=True)
        return self.episodes < self.stages[-1]


def pick_device(algo, size):
    """The RTX 3050 only speeds up SAC's big networks; PPO and small networks are faster on the CPU."""
    return "cuda" if algo == "sac" and max(size) >= 256 and torch.cuda.is_available() else "cpu"


def train(algo, lr, size, seed, stages, device="cpu", root=ROOT):
    """Trains one model and saves resources/<algo>/<name>_ep<stage>.zip at every stage.

    A run whose stage files all exist is skipped. A run that already reached a stage carries on from
    it instead of starting over: after a restart cut it short, or after 8000 is added to the stages."""
    stages = sorted(stages)
    name = model_name(algo, lr, size, seed)
    if finished(algo, lr, size, seed, stages, root):
        print(f"{name}  already trained, skipping", flush=True)
        return None
    folder = os.path.join(root, "resources", algo)
    extras = os.path.join(folder, "continue")  # what a run needs to carry on past its last stage later
    os.makedirs(extras, exist_ok=True)
    os.makedirs(os.path.join(root, "logs"), exist_ok=True)
    state_file = os.path.join(extras, f"{name}_state.json")
    buffer_file = os.path.join(extras, f"{name}_buffer.pkl")
    log_file = os.path.join(root, "logs", f"{name}.csv")

    state = None
    if os.path.exists(state_file):
        with open(state_file, encoding="utf-8") as f:
            state = json.load(f)
    start = state["episodes"] if state else 0
    can_continue = (0 < start < stages[-1]
                    and os.path.exists(stage_path(algo, lr, size, seed, start, root))
                    and all(os.path.exists(stage_path(algo, lr, size, seed, s, root)) for s in stages if s <= start)
                    and (algo == "ppo" or os.path.exists(buffer_file)))
    if not can_continue:
        start, state = 0, None
        if os.path.exists(state_file):
            os.remove(state_file)

    env = GaussPlanetWrapper(gym.make("LunarLanderContinuous-v3", enable_wind=True), seed)
    if start:
        model = ALGOS[algo].load(stage_path(algo, lr, size, seed, start, root), env=env, device=device)
        if algo == "sac":
            model.load_replay_buffer(buffer_file)
        env.rng.bit_generator.state = state["rng"]
        with open(log_file, newline="", encoding="utf-8") as f:  # drop rows from an earlier attempt that was cut short
            rows = list(csv.reader(f))[:start + 1]  # header + episodes 1..start (a crash can leave half a row)
        with open(log_file, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(rows)
        print(f"{name}  carrying on from episode {start}", flush=True)
    else:
        policy_kwargs = dict(net_arch=list(size))
        model = ALGOS[algo]("MlpPolicy", env, learning_rate=lr, policy_kwargs=policy_kwargs, seed=seed,
                            device=device, verbose=0)

    started = time.time()
    with open(log_file, "a" if start else "w", newline="", encoding="utf-8") as f:
        log = csv.writer(f)
        if not start:
            log.writerow(["episode", "steps", "return", "length", "gravity", "wind", "turbulence"])

        def save_progress(episodes):
            """What the run needs to carry on from this stage, so a restart only loses the work since it."""
            f.flush()
            if os.path.exists(state_file):
                os.remove(state_file)  # a crash mid-save then means starting over, never a mismatched buffer
            if algo == "sac":
                model.save_replay_buffer(buffer_file + ".tmp")
                os.replace(buffer_file + ".tmp", buffer_file)
            seconds = time.time() - started + (state["seconds"] if state else 0)
            with open(state_file + ".tmp", "w", encoding="utf-8") as g:
                json.dump({"episodes": episodes, "steps": model.num_timesteps, "seconds": round(seconds),
                           "device": device, "rng": env.rng_state}, g, indent=1)
            os.replace(state_file + ".tmp", state_file)

        saver = StageSaver(name, folder, stages, start, log, save_progress)
        model.learn(total_timesteps=MAX_STEPS * (stages[-1] - start + 1), callback=saver,
                    reset_num_timesteps=not start)
    seconds = time.time() - started + (state["seconds"] if state else 0)
    env.close()
    print(f"{name}  done: {saver.episodes} episodes, {model.num_timesteps:,} steps, {seconds / 60:.1f} min "
          f"on {device} ({model.num_timesteps / seconds:.0f} steps/s)", flush=True)
    return dict(name=name, episodes=saver.episodes, steps=model.num_timesteps, seconds=round(seconds), device=device)


if __name__ == "__main__":
    # One run on its own; driver.py trains the whole study
    train("sac", 3e-4, [256, 256], 0, [125, 250, 500], device=pick_device("sac", [256, 256]))
