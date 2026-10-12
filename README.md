# PPO vs SAC on LunarLander: which settings matter most

Trains PPO and SAC landers with different learning rates, network sizes and amounts of
training, then tests every model on the same 300 test planets and charts which setting
matters most, and where performance drops off.

## The study

| Setting | Values |
|---|---|
| Algorithm | PPO, SAC on `LunarLanderContinuous-v3` |
| Learning rate | 1e-4, 3e-4 (default), 1e-3, 3e-3 |
| Network size | 16x16, 64x64, 256x256 |
| Seeds | 0, 1, 2 |
| Training episodes (stages) | 125, 250, 500, 1000, 2000, 4000 |
| Everything else | Stable-Baselines3 defaults |

That's 72 training runs. Each run trains once and saves its weights at every stage, which
gives 432 models. The lists are at the top of `driver.py`.

## Training planets

Every model trains on planets drawn from bell curves around the ideal planet. Each
training landing gets a new planet:

| | Bell curve | Result |
|---|---|---|
| Gravity | centred on -10, spread 1 | 70% of landings between -9 and -11; 14% at -11 to -11.9 |
| Wind | half bell curve from 0, spread 7 | half under 5, 32% at 5-10, 3% at 15-20 |
| Turbulence | half bell curve from 0, spread 0.7 | half under 0.5, 3% at 1.5-2 |

- Wind is on in every landing, and the three values are drawn separately.
- A draw outside the test ranges (gravity -2 to -11.9, wind 0-20, turbulence 0-2) is drawn again.
- The planets come from a generator seeded with the run's seed. Every model with the same seed
  meets the same planets in the same order, and a run that carries on past its last stage
  continues the same sequence.
- The bell curves are set at the top of `trainer.py`.
- **No keyboard control.** `GaussPlanetWrapper` works like your `ChaosPlanetWrapper`: it only sets
  gravity, wind and turbulence before each landing. Nothing in this folder reads the keyboard
  or pushes the lander, and nothing imports `lunar_random.py`, which starts its WASD
  `run()` when imported.

## Test planets

There are 6 conditions with 50 planets each. They are written once to `resources/envs/`,
one JSON file per planet:

| Condition | Gravity | Wind | Turbulence |
|---|---|---|---|
| ideal | -10 | 0 | 0 |
| moderate wind | -10 | 5-10 | 0.5-1 |
| heavy wind | -10 | 15-20 | 1.5-2 |
| heavy gravity | -11 to -11.9 | 0 | 0 |
| low gravity | -2 to -4 | 0 | 0 |
| random mix | -2 to -11.9 | 0-20 | 0-2 |

- `TEST_SEED` in `testing.py` fixes all 300 planets.
- The seed in each file name fixes that landing's terrain, starting push and wind pattern.
- Training never uses these seeds.

## Setup on the PC (once)

Make a Python 3.13 venv in this folder. Install torch first, from the CUDA index, so SB3 doesn't
pull in the CPU-only build:

```
py -3.13 -m venv .venv
.\.venv\Scripts\activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install stable-baselines3==2.8.0 "gymnasium[box2d]==1.2.3" "numpy~=2.4.0" pandas==3.0.2 matplotlib==3.10.8
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Activate the venv (`.\.venv\Scripts\activate`) in every new terminal before running the commands below.

The last line should print `True NVIDIA GeForce RTX 3050`. If it prints `False`, everything
still works, but it all runs on the CPU and takes longer.

## 1. Quick check (about 5 minutes)

```
python driver.py train --quick
python driver.py test --quick --log
python driver.py plots --quick
```

This runs a tiny version in `quick/`: 2 learning rates, 2 sizes, 1 seed, stages 10 and 30,
and 2 test planets per condition.

- Each run's `done` line shows its device and its speed in steps per second. The SAC 256x256 runs
  should say `on cuda`.
- A full 4000-episode SAC run is roughly 1.5-2.4 million steps: early landers hover until the
  1000-step limit, and even good ones take 300-400 steps to land in the wind.
- Delete the `quick` folder afterwards.

## 2. Train (about 40 hours)

```
python driver.py train --gpu-workers 0
```

- 11 runs train at once, one per CPU thread, longest runs first.
- `--gpu-workers 0` keeps everything on the CPU. On this laptop the GPU is no faster than a CPU thread
  once the CPU is busy. Without the flag, the 12 SAC 256x256 runs queue for 2 GPU slots, which
  takes much longer.
- To change how many runs train at once, use `--cpu-workers N` and `--gpu-workers N`.
  - More than 12, the CPU's thread count, doesn't make anything faster.
  - Each SAC run uses about 0.6-0.7 GB of RAM, so 11 runs at once nearly fill the 16 GB.
  - Use 12 only with other apps closed.
- A line prints when a run reaches each stage: episode, total steps, mean return of the last 100
  landings and minutes so far.
- To see how far the whole study has got, open a second PowerShell window and run:
  ```
  cd C:\Users\gurka\Desktop\RL-IA
  .\.venv\Scripts\python driver.py progress
  ```
  It redraws one line every 30 seconds, for example
  `[###########-------------------]  37.4%  27 of 72 runs done  about 25 h left`.
  - It only reads `logs/` and checks which models exist. Ctrl+C there closes it without touching
    training.
  - The % counts each run by how long it takes, so it moves at about the same speed all the way
    through. A SAC 256x256 run counts 12 times as much as a PPO run.
  - The time left comes from the same rough speeds, so treat it as an estimate.
  - It shows 0% until the first stage, about 45 minutes in. After that it lags about 1% behind,
    because training writes its logs in chunks of about 200 landings.
  - It doesn't load torch, so it needs only a few tens of MB of RAM.
  - If you train with `--cpu-workers N`, pass the same `--cpu-workers N` so the time left is right.
- Before you start:
  - keep the laptop plugged in, with sleep set to "Never" on mains power;
  - pause Windows Update for a week (Settings > Windows Update > Pause updates), because it restarts the
    PC at night.
- Run it in its own PowerShell window, not VS Code's terminal, so closing VS Code doesn't stop it.
- **Ctrl+C**, a crash or a restart stops everything. Run the same command again to carry on:
  finished runs are skipped, and unfinished ones pick up from their last saved stage. Only the work
  since that stage is lost.

## 3. Test (about 10-20 minutes)

```
python driver.py test          # prints a table per model
python driver.py test --log    # also saves results/test_results.csv and results/summary.csv
```

- With `--log`, models already in `test_results.csv` are skipped, so testing can be stopped and
  restarted. To test everything again, delete `results/`.
- If a results file is open in Excel, testing waits until you close it.

To test one model:

```
python testing.py resources/sac/sac_lr3e-4_net256x256_seed0_ep4000.zip
python testing.py resources/sac/sac_lr3e-4_net256x256_seed0_ep4000.zip --log
python testing.py resources/sac/sac_lr3e-4_net256x256_seed0_ep4000.zip --render --conditions heavy_wind
```

The same from Python: `from testing import test; test("resources/...zip", log=True)`.

When testing, all landers fly side by side so the model handles them in one go. With
`--render` they fly one at a time so you can watch. Because of rounding, a watched landing can
come out a hair different from the logged one, e.g. coming to rest a few steps later.

## 4. Charts

```
python driver.py plots
```

Charts go to `plots/`, and the main findings are printed:

| File | Shows |
|---|---|
| `1_importance.png` | which settings matter most: each setting's share of the differences between models (ANOVA), including combinations of settings and seed-to-seed luck |
| `2_importance_by_condition.png` | the same shares, worked out separately for each test condition |
| `3_return_vs_episodes.png` | test return against training episodes for each algorithm and size, one line per learning rate, with a band from the worst to the best seed. This is where declines show up |
| `4_return_vs_size.png` | test return against network size at each stage |
| `5_outcomes.png` | how the landings ended (landed, ran out of time, crashed) for each condition and stage |
| `conditions/*.png` | chart 3 for each test condition on its own |

`results/peaks.csv` lists, for every algorithm, learning rate and size, the stage with the best
test return and how much is lost by the last stage.

## Training past 4000 episodes

Add `8000` to `STAGES` in `driver.py` and run `python driver.py train` again. Each run carries
on from its 4000 file instead of starting over:
- SAC reloads its replay buffer, like your `lunar.py`;
- the planet sequence carries on where it stopped.

## Files

```
RL-IA/
  driver.py      study settings, train_all(), test_all(), show_progress()
  names.py       model names and file paths (no torch, so progress stays small)
  trainer.py     GaussPlanetWrapper, train()
  testing.py     make_envs(), test()
  plots.py       charts
  resources/
    ppo/, sac/   sac_lr3e-4_net256x256_seed0_ep1000.zip  (algorithm, lr, size, seed, training episodes)
      continue/  SAC replay buffers and planet-generator states, saved at every stage so a run can carry on
    envs/        heavy_wind_07_g-10.00_wind17.3_turb1.84_seed735120.json
  logs/          one CSV per run: every training landing with its return and planet
  results/       test_results.csv, summary.csv, peaks.csv
  plots/
```

Disk use is about 0.5 GB of models, plus about 3 GB of SAC replay buffers. Once training has
finished, and if you won't train past 4000 episodes, you can delete `resources/sac/continue/*.pkl`.

### `results/test_results.csv` (one row per test landing)

| Column | Meaning |
|---|---|
| model, algorithm, lr, size, seed | which model |
| train_episodes, train_steps | how much it trained |
| env, condition, gravity, wind, turbulence | which test planet |
| return | total reward of the landing |
| steps | how long the landing took (1000 = ran out of time) |
| outcome | landed, crashed (including flying off screen) or timeout |
| solved | return of 200 or more |

`results/summary.csv` has one row per model and condition, plus `all`. Each row gives the mean
and spread of the return, % solved, % landed, % crashed, % timed out and the mean steps.
