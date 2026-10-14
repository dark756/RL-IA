# PPO vs SAC on LunarLander: which settings matter most

Trains PPO and SAC landers with different learning rates, network sizes and amounts of
training, then tests every model on the same 300 test planets and charts which setting
matters most, and where performance drops off.

The study is finished: all 432 models are trained, tested and charted, and the write-up is
`submit/LAB CA-Report.pdf`.

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

## Results

| | Setting | Mean test return | Landed | Training per run |
|---|---|---|---|---|
| Best SAC (SAC 1) | 256x256, lr 3e-3, 4000 episodes | 275 | 96% | 8.1-9.8 h |
| Best PPO (PPO 1) | 256x256, lr 1e-4, 4000 episodes | 220 | 85% | 0.8-1.0 h |
| Cheap SAC (SAC 4) | 64x64, lr 3e-3, 2000 episodes | 264 | 93% | about 2.0-2.6 h |

- SAC beats PPO at the same number of episodes. PPO runs 7-13 times faster.
- PPO with lr 1e-3 or higher on a 64x64 or 256x256 network collapses. 256x256 at lr 3e-3 never learns
  and ends at -968.
- Low gravity, which training never sees, is the hardest condition for every model.
- Which setting matters most depends on the measure. By mean return, settings combined and seed luck
  dominate. By landed %, training episodes explain 40% of the spread.

The tables per setting and seed are in `submit/best performers.md`.

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
still works, but it all runs on the CPU.

## 1. Quick check (about 5 minutes)

```
python driver.py train --quick
python driver.py test --quick --log
python driver.py plots --quick
```

This runs a tiny version in `quick/`: 2 learning rates, 2 sizes, 1 seed, stages 10 and 30,
and 2 test planets per condition.

- Each run's `done` line shows its device and its speed in steps per second.
- A full 4000-episode SAC run takes 0.8-2.4 million steps. SAC 256x256 needs about 1 million.
  SAC 16x16 needs about 2.4 million, because 38% of its training landings hover until the
  1000-step limit.
- Delete the `quick` folder afterwards.

## 2. Train (about 28 hours)

```
python driver.py train --cpu-workers 11 --gpu-workers 0
```

- 11 runs train at once, one per CPU thread, longest runs first.
- `--gpu-workers 0` keeps everything on the CPU. On this laptop the GPU is no faster than a CPU thread
  once the CPU is busy, because each kernel launch costs 30-80 us under Windows. Without the flag, the
  12 SAC 256x256 runs queue for 2 GPU slots, which takes much longer.
- To change how many runs train at once, use `--cpu-workers N` and `--gpu-workers N`.
  - More than 12, the CPU's thread count, doesn't make anything faster.
  - Each SAC run uses about 0.6-0.7 GB of RAM, so 11 runs at once nearly fill the 16 GB.
  - Use 12 only with other apps closed.
- The study trained from 2026-10-12 10:18 to 2026-10-13 14:43: 271 run-hours in all. Measured time per
  run, with one CPU thread each and up to 11 runs at once:

  | Runs | Hours per run | Steps per run |
  |---|---|---|
  | PPO, each size | 0.7 (0.2-1.1) | 1.3-1.8 million |
  | SAC 16x16 | 6.6 (5.1-8.5) | 2.4 million |
  | SAC 64x64 | 4.7 (3.7-5.7) | 1.4 million |
  | SAC 256x256 | 9.1 (8.1-10.1) | 1.0 million |

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
    through.
  - The time left comes from rough speeds guessed before training (`SAC_HOURS` and `PPO_HOURS` in
    `driver.py`), so treat it as an estimate.
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

## 3. Test (about 10 minutes)

```
python driver.py test          # prints a table per model
python driver.py test --log    # also saves results/test_results.csv and results/summary.csv
```

- 432 models x 300 planets is 129,600 landings. With 11 workers that took about 10 minutes.
- With `--log`, models already in `test_results.csv` are skipped, so testing can be stopped and
  restarted. To test everything again, delete `results/`.
- If a results file is open in Excel, testing waits until you close it.

To test one model:

```
python testing.py resources/sac/sac_lr3e-3_net256x256_seed0_ep4000.zip
python testing.py resources/sac/sac_lr3e-3_net256x256_seed0_ep4000.zip --log
python testing.py resources/sac/sac_lr3e-3_net256x256_seed0_ep4000.zip --render --conditions heavy_wind
```

The same from Python: `from testing import test; test("resources/...zip", log=True)`.

When testing, all landers fly side by side so the model handles them in one go. With
`--render` they fly one at a time so you can watch. Because of rounding, a watched landing can
come out a hair different from the logged one, e.g. coming to rest a few steps later. A watched
landing plays in real time (50 steps a second); press Ctrl+C in the terminal to stop, since each
condition has 50 planets.

## 4. Charts

```
python driver.py plots
```

Charts go to `plots/`, and the main findings are printed:

| File | Shows | In the report |
|---|---|---|
| `1_importance.png` | which settings matter most: each setting's share of the differences between models (ANOVA), including combinations of settings and seed-to-seed luck | Figure A2 |
| `2_importance_by_condition.png` | the same shares, worked out separately for each test condition | Figure A3 |
| `3_return_vs_episodes.png` | test return against training episodes for each algorithm and size, one line per learning rate, with a band from the worst to the best seed. This is where declines show up | Figure 5 |
| `4_return_vs_size.png` | test return against network size at each stage | Figure A5 |
| `5_outcomes.png` | how the landings ended (landed, ran out of time, crashed) for each condition and stage | Figure A4 |
| `6_landed_vs_episodes.png` | chart 3 for the landing success rate: the % of test landings that came to rest | Figure 6 |
| `7_training_return.png` | the return of the training landings against training episodes, from `logs/` (mean of the last 100) | Figure 4 |
| `8_training_steps.png` | the steps per training landing against training episodes; 1000 means it ran out of time | not used |
| `conditions/*.png` | chart 3 for each test condition on its own | low gravity: Figure A6, heavy wind: Figure A7 |

Charts 7 and 8 show training: each landing is on a new random planet, with exploration on, so they're noisier and
lower than the test charts. They're drawn from `logs/`, which isn't in git, so only this PC can redraw them.

The report's two diagrams don't depend on the results. Draw them with:

```
python diagram.py
```

| File | Shows | In the report |
|---|---|---|
| `diagram_rl_loop.png` | the learning loop: planet draw, environment, state, agent, action, reward, learning | Figure 1 |
| `diagram_pipeline.png` | how the study was run: 72 runs, 432 models, 300 test planets, 129,600 test landings | Figure 2 |

`results/peaks.csv` lists, for every algorithm, learning rate and size, the stage with the best
test return and how much is lost by the last stage.

## Training past 4000 episodes

The SAC replay buffers (`resources/sac/continue/*_buffer.pkl`, 3 GB) were deleted on 2026-10-14, so
only PPO can still carry on exactly.

- **PPO:** add 8000 to the stages for PPO only, from this folder:
  ```
  python -c "import driver; driver.train_all(algos=['ppo'], stages=driver.STAGES + [8000], cpu_workers=11, gpu_workers=0)"
  ```
  Each PPO run reloads its 4000 file and its planet sequence carries on where it stopped.
- **Don't add 8000 to `STAGES` in `driver.py` and run `train`.** Without its buffer, `train()` treats a SAC
  run as unable to carry on and starts it again from episode 0. That overwrites the run's saved models, its
  log and its state file, which the report's training times come from. Change the code first.

## The report

| File in `submit/` | What it is |
|---|---|
| `LAB CA-Report.docx`, `.pdf` | the report, version 2 (2026-10-14) |
| `LAB CA-Report v1.docx`, `.pdf` | the first version |
| `LAB CA-Report Format.docx` | the course's template, which the report keeps the fonts and layout of |
| `LAB CA-Rubrics.docx.pdf` | the marking rubric |
| `report plan.md` | what goes in each section, the demo commands for presenting it, and viva questions |
| `best performers.md` | the top settings in detail, with every seed's model file |
| `fig3_landings.png`, `figA1_ppo_progress.png` | the report's landing pictures (Figures 3 and A1), drawn from the saved models |

The report was generated by a script outside this repository, so make later changes in Word.
Your name, roll number, batch and date of performance on page 1 are still blank.

## Files

```
RL-IA/
  driver.py      study settings; train_all(), test_all(), show_progress()
  names.py       model names and file paths (no torch, so progress stays small)
  trainer.py     GaussPlanetWrapper, StageSaver, train()
  testing.py     make_envs(), land(), test(), summarize()
  plots.py       the charts, variance_shares(), peaks()
  diagram.py     the report's two diagrams
  resources/     (not in git)
    ppo/, sac/   sac_lr3e-4_net256x256_seed0_ep1000.zip  (algorithm, lr, size, seed, training episodes)
      continue/  one state file per run: episodes, steps, training seconds, planet-generator state
    envs/        heavy_wind_07_g-10.00_wind17.3_turb1.84_seed735120.json
  logs/          (not in git) one CSV per run: every training landing with its return and planet
  results/       test_results.csv, summary.csv, peaks.csv
  plots/         the charts, conditions/ and the two diagrams
  observations/  my observations.txt (your notes on the charts) and claudes observations.txt (a review of them)
  submit/        the report, its template and rubric, and notes for presenting it
```

Disk use is about 0.4 GB for `resources/` (432 models and the test planets) and 13 MB for `logs/`.

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

## Code reference

Every function and class in the six code files, how to use it, and what it does. All six files
can be imported without side effects: each only runs its command line under
`if __name__ == "__main__"`. So you can call these functions from Python in this folder, with the venv
active. Functions marked "internal" are called by the others and rarely need calling yourself.

### `driver.py`: runs the whole study

Command line: `python driver.py train | test [--log] | plots | progress`, plus `--quick`,
`--cpu-workers N` and `--gpu-workers N`.

The settings at the top are:
- `ALGOS`, `LEARNING_RATES`, `SIZES`, `SEEDS` and `STAGES`: the study grid;
- `QUICK` and `QUICK_ENVS_PER_CONDITION`: the `--quick` version;
- `GPU_WORKERS`: big SAC runs at once on the GPU;
- `SAC_HOURS` and `PPO_HOURS`: the progress bar's weights.

| Function | How to use it | What it does |
|---|---|---|
| `train_all(algos, learning_rates, sizes, seeds, stages, root, cpu_workers, gpu_workers)` | `python driver.py train --cpu-workers 11 --gpu-workers 0`, or `train_all(cpu_workers=11, gpu_workers=0)` | Trains every run that isn't finished, several at once: big SAC runs in a GPU pool, the rest in a CPU pool. Finished runs are skipped and unfinished ones carry on from their last stage. |
| `test_all(log, root, envs_per_condition, workers)` | `python driver.py test --log`, or `test_all(log=True)` | Tests every saved model on the test planets in parallel and prints each table. With `log=True` it saves the landings, skips models already saved and rewrites `summary.csv`. |
| `show_progress(algos, learning_rates, sizes, seeds, stages, root, workers, every)` | `python driver.py progress` | Redraws one progress line every `every` seconds (30) from the logs and stage files, until every run is done or Ctrl+C. |
| `episodes_done(run, stages, root)` | internal: called by `show_progress` | How many training landings a run has finished, read from the end of its log. |
| `train_job(run, stages, root, device)` | internal: run in a worker by `train_all` | Sets one torch thread and trains one run with `trainer.train`. |
| `test_job(model_path, root)` | internal: run in a worker by `test_all` | Sets one torch thread and tests one model with `testing.test`. |
| `stop_pool(pool)` | internal: used on Ctrl+C | Stops a pool's worker processes at once instead of waiting for their runs to end. |

### `names.py`: model names and file paths

It holds `ROOT`, the project folder. It doesn't import torch, so `driver.py progress` stays small.

| Function | How to use it | What it does |
|---|---|---|
| `lr_tag(lr)` | `lr_tag(0.0003)` gives `'3e-4'` | The short learning-rate tag used in file names. |
| `model_name(algo, lr, size, seed)` | `model_name("sac", 3e-4, [256, 256], 0)` gives `'sac_lr3e-4_net256x256_seed0'` | A run's name. Its stage files add `_ep<episodes>`. |
| `stage_path(algo, lr, size, seed, episodes, root)` | `stage_path("sac", 3e-4, [256, 256], 0, 1000)` | The path of one stage's model zip in `resources/<algo>/`. |
| `finished(algo, lr, size, seed, stages, root)` | `finished("sac", 3e-4, [256, 256], 0, [125, 250])` | True if every stage zip of the run exists. |

### `trainer.py`: trains one run

Command line: `python trainer.py` trains one example run (SAC, lr 3e-4, 256x256, seed 0, stages
125-500). Here it just says the run is already trained.

The settings at the top are:
- `MAX_STEPS`: 1000 steps per landing;
- `GRAVITY_MEAN`, `GRAVITY_SPREAD`, `WIND_SPREAD` and `TURBULENCE_SPREAD`: the bell curves;
- `GRAVITY_LIMITS`, `WIND_LIMITS` and `TURBULENCE_LIMITS`: the ranges outside which a draw is redrawn.

| Function | How to use it | What it does |
|---|---|---|
| `train(algo, lr, size, seed, stages, device="cpu", root=ROOT)` | `train("ppo", 1e-4, [256, 256], 1, [125, 250, 500, 1000, 2000, 4000])` | Trains one run on random planets. It saves `resources/<algo>/<name>_ep<stage>.zip` at every stage and logs every landing to `logs/<name>.csv`. A finished run is skipped, and an interrupted one carries on from its last stage. |
| `GaussPlanetWrapper(env, seed)` | `GaussPlanetWrapper(gym.make("LunarLanderContinuous-v3", enable_wind=True), seed=0)` | A Gymnasium wrapper. `reset()` draws a new gravity, wind and turbulence from the seeded bell curves before each landing, and `step()` adds the planet to `info` when a landing ends. It never pushes the lander. |
| `StageSaver(name, folder, stages, episodes, log, save_progress)` | internal: the callback `train` gives `model.learn` | SB3 calls its `_on_step()` after every step. It counts finished landings, writes each one to the log, saves the weights and progress at each stage and stops after the last stage. |
| `save_progress(episodes)`, inside `train` | internal: called by `StageSaver` at each stage | Flushes the log, then writes the SAC replay buffer and the state file atomically: episodes, steps, seconds and the planet-generator state. |
| `draw(sample, limits)` | `draw(lambda: rng.normal(-10, 1), (-11.9, -2.0))` | Draws again until the value is inside the limits. |
| `pick_device(algo, size)` | `pick_device("sac", [256, 256])` | Gives `"cuda"` only for SAC with 256-wide layers when a GPU is available, otherwise `"cpu"`. |

### `testing.py`: tests saved models on the test planets

Command line: `python testing.py <model.zip> [--log] [--render] [--conditions heavy_wind low_gravity ...]`.

The settings at the top are:
- `CONDITIONS`: each condition's gravity, wind and turbulence ranges;
- `ENVS_PER_CONDITION`: 50;
- `TEST_SEED`: 2026;
- `SOLVED`: 200.

| Function | How to use it | What it does |
|---|---|---|
| `test(model_path, log=False, render=False, conditions=None, root=ROOT)` | `python testing.py resources/sac/sac_lr3e-3_net256x256_seed0_ep4000.zip`, or `test(path, conditions=["low_gravity"])` | Tests one model on every test planet, or only the given conditions. It prints a table per condition and returns the landings as a DataFrame. `log=True` also saves them; `render=True` shows each landing in a window. |
| `land(model, planets, render=False)` | `land(SAC.load(path), load_envs(conditions=["ideal"])[:5])` | Flies one deterministic landing per planet, all side by side with one batched `predict` per step. It labels each one landed, crashed or timeout. |
| `make_envs(per_condition=50, root=ROOT)` | `make_envs()` | Writes the test planets to `resources/envs/`, replacing the old files. `TEST_SEED` makes them identical every time. |
| `load_envs(root=ROOT, conditions=None)` | `load_envs(conditions=["heavy_wind"])` | Reads the test planets, writing them first if they don't exist. |
| `model_info(model_path)` | `model_info("sac_lr3e-4_net256x256_seed0_ep1000.zip")` | Reads the algorithm, lr, size, seed and training episodes from a model's file name. |
| `table(results)` | `print(table(results))` | One model's results per condition as text. This is what `test` prints. |
| `save(results, root=ROOT)` | internal: called by `test(log=True)` and `driver.test_all` | Appends one model's landings to `results/test_results.csv`. |
| `summarize(root=ROOT)` | `summarize()` | Rewrites `results/summary.csv` from `test_results.csv`: one row per model and condition, plus `all`. A retested model keeps only its newest landings. |
| `write_csv(frame, path, **kwargs)` | internal: used by `save` and `summarize` | Writes a CSV, waiting and retrying while another program (Excel) has the file open. |

### `plots.py`: charts and the analysis

Command line: `python plots.py`, or `python driver.py plots`.

The settings at the top are:
- the colours (`LR_COLORS`, `LR_MARKERS`, `OUTCOMES`, `BLUES` and the greys), checked for colour-blind
  safety;
- `SETTINGS`: the four settings the analysis uses.

| Function | How to use it | What it does |
|---|---|---|
| `make_plots(root=ROOT)` | `python driver.py plots` | Draws every chart into `plots/` and writes `results/peaks.csv`. It prints the importance shares, the ten best settings and where each setting peaks. |
| `variance_shares(models)` | `variance_shares(per_model(results))` | The analysis of variance (eta squared): each setting's share of the spread in mean test return between models, and each combination's. The rest is seed luck. |
| `per_model(results)` | `per_model(pd.read_csv("results/test_results.csv"))` | One row per model, with its mean return and % landed over the given landings. |
| `peaks(models)` | `peaks(per_model(results))` | For each algorithm, lr and size: the stage with the best mean return and the drop after it (the content of `peaks.csv`). |
| `read_logs(root, window)` | `read_logs(".", 100)` | Every training landing in `logs/` in one table, with each run's rolling mean return and length. |
| `plot_importance(shares, path)` | internal: chart 1 | Bar chart of the importance shares. |
| `plot_importance_by_condition(table, path)` | internal: chart 2 | Heat map of the shares in each test condition. |
| `plot_curves(results, colors, path, title, note, y="return")` | internal: charts 3, 6 and `conditions/` | Test return, or % landed with `y="landed"`, against training episodes. One panel per algorithm and size, one line per learning rate. |
| `plot_sizes(results, colors, path)` | internal: chart 4 | Test return against network size at each stage. |
| `plot_outcomes(results, path)` | internal: chart 5 | Stacked bars of landed, ran out of time and crashed, per algorithm, condition and stage. |
| `plot_training(logs, y, colors, episodes, path, title, note)` | internal: charts 7 and 8 | Training return (`y="return"`) or steps per landing (`y="length"`) against training episodes, from `logs/`. |
| `draw_lines(ax, runs, x_column, x_values, colors, lrs, y="return")` | internal | One line per learning rate (the mean of the seeds), with a band from the worst to the best seed. |
| `reference_line(ax, value, text, label=False)` | internal | A dashed line for the solved score or the time limit. |
| `lr_handles(colors, lrs)` | internal | The legend entries for the learning rates. |
| `levels(models)` | internal | The algorithms, sizes, learning rates and stages present in the results. |
| `header(fig, title, note, handles=None)` | internal, also used by `diagram.py` | The title, note and legend band above a chart. |
| `style()` | internal, also used by `diagram.py` | The shared matplotlib style. |
| `short(episodes)` | `short(4000)` gives `'4k'` | A short stage label. |
| `width(size)` | `width("64x64")` gives `64` | Neurons per layer from a size like `"64x64"`. |

### `diagram.py`: the report's two diagrams

Command line: `python diagram.py`. It writes `plots/diagram_rl_loop.png` and `plots/diagram_pipeline.png`.

| Function | How to use it | What it does |
|---|---|---|
| `make_diagrams(root=ROOT)` | `python diagram.py` | Draws both diagrams into `plots/`. |
| `draw_loop(path)` | internal | Figure 1: the learning loop (planet draw, environment, agent, learning). |
| `draw_pipeline(path)` | internal | Figure 2: study design, training, testing and analysis, with the counts between them. |
| `default(algo, name)` | `default(PPO, "n_steps")` gives `2048` | Reads a Stable-Baselines3 default, so the diagrams' numbers match SB3. |
| `canvas(width, height, title, note)` | internal | A figure with the usual header and a drawing area measured in inches. |
| `box(ax, x, y, w, h, colors, title, lines)` | internal | A rounded box with a title and lines of text. |
| `arrow(ax, *points)` | internal | An arrow through the given points. |
| `label(ax, x, y, text, ha="center", va="center")` | internal | Grey text next to an arrow. |
