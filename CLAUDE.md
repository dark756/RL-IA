# PPO vs SAC on LunarLander

A study comparing PPO and SAC on LunarLander, built by Claude on 2026-10-12. `README.md` here is the full guide for
the user. Its "Code reference" lists every function in the six code files, how to call it and what it does; keep it
in step whenever the code changes.

The study was built as `RL-main/research/`. On the Windows PC it is its own repo, `Desktop/RL-IA/`, whose parent is
the Desktop. The user's own work lives in `RL-main/`, which isn't next to it here:
- the scripts `lunar.py`, `lunar_random.py`, `lunar_non_continuous.py`, `polecart*.py`, `frozenLake.py` and `game.py`;
- the model folders `lunar/` and `lunar_nc/`.

## Working with the user

- **Plan first. Don't write code or start training until the user says "go".** Questions,
  plan changes and a rejected plan approval are not a go-ahead. Code written early has had to be deleted twice.
- When asked for a plan or summary, keep it short and give the actual numbers: settings, training-episode
  stages and file names. Then wait.
- After "go", build the whole plan, test it end to end, and delete the test outputs.
- Don't edit the user's scripts in `RL-main/`. Never import them either: they start training or a pygame
  loop as soon as they load. `lunar_random.py`'s `run()` pushes the lander with the WASD keys.
- **No keyboard input may reach the lander in this folder.** A planet wrapper is fine if it only sets gravity, wind
  and turbulence, like `ChaosPlanetWrapper`.

## Machines

**Windows PC (main machine):** Windows 11 laptop with an i5-12450H (8 cores, 12 threads), 16 GB of RAM and an
RTX 3050 6GB Laptop GPU.
- Use the venv `.venv/` (Python 3.13.5), e.g. `.\.venv\Scripts\python driver.py ...`. The system Python has older
  numpy, pandas and matplotlib, and no torch or SB3. Don't reuse `../RL-3D/.venv`: it has other versions and no Box2D.
- Libraries in `.venv`: torch 2.11.0+cu128 (`torch.cuda.is_available()` is True), SB3 2.8.0, gymnasium[box2d] 1.2.3,
  numpy 2.4.6, pandas 3.0.2 and matplotlib 3.10.8. To rebuild it, install torch first so SB3 doesn't pull in the CPU build:
  1. `pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128`
  2. `pip install stable-baselines3==2.8.0 "gymnasium[box2d]==1.2.3" "numpy~=2.4.0" pandas==3.0.2 matplotlib==3.10.8`
- Workers start with the `spawn` method, so keep the `if __name__ == "__main__"` guards.
- Print ASCII only, because the console may be cp1252. Write files with `encoding="utf-8"`.
- The PC must not sleep during long runs. Sleep on mains power is set to Never; on battery it's 10 minutes, so keep it
  plugged in.
- Pause Windows Update before long runs. Outside the active hours (09:00-03:00) it restarts the PC to install
  updates, as it did at 05:17 on 2026-10-12.

**Intel Mac (small tests only):** it can't install torch >= 2.3, which SB3 2.8 needs.
- Test there in a throwaway venv with:
  - torch 2.2.2 and `numpy<2`;
  - `pip install --no-deps stable-baselines3==2.8.0`;
  - gymnasium[box2d] 1.2.3, pandas 3.0.2 and matplotlib 3.10.8.
- Loading the PC's numpy-2 zips there needs `custom_objects`: new observation and action spaces, and None for
  `_last_obs`, `_last_episode_starts`, `_last_original_obs`, `ep_info_buffer` and `ep_success_buffer`.

## The study

**Question:** which setting matters most for landing, and where does performance drop off?

| Setting | Values |
|---|---|
| Algorithm | PPO, SAC on `LunarLanderContinuous-v3`, with SB3 defaults and a constant learning rate |
| Learning rate | 1e-4, 3e-4, 1e-3, 3e-3 |
| Network size | [16,16], [64,64], [256,256] (the user capped it at 256) |
| Seeds | 0, 1, 2 |
| Stages (training episodes) | 125, 250, 500, 1000, 2000, 4000 |

That's 72 runs. Each trains once and saves its weights at every stage, which gives 432 models.

**Training planets.** Every run trains on "gauss" planets. The user chose these over ideal-only or uniform-random
training.
- `GaussPlanetWrapper` in `trainer.py` draws a new planet before each landing:
  - gravity N(-10, 1);
  - wind |N(0, 7)|;
  - turbulence |N(0, 0.7)|.
- A draw outside the test ranges is drawn again. The ranges are gravity -11.9 to -2, wind 0-20 and turbulence 0-2.
- The generator is seeded with the run's seed, so the same seed always gives the same sequence of planets.

**Test planets.** 300 fixed planets: 6 conditions, 50 each.
- The conditions are ideal, moderate wind, heavy wind, heavy gravity, low gravity and random mix.
- Training never sees low gravity.
- `TEST_SEED = 2026` in `testing.py` fixes the planets, and they're written to `resources/envs/*.json`.
- Each model makes one deterministic landing per planet.

**Names.** A model file looks like `resources/sac/sac_lr3e-4_net256x256_seed0_ep1000.zip`. The training condition
isn't in the name because the user asked to drop it.

**Commands** (run from this folder):
- `python driver.py train | test [--log] | plots | progress`.
  - `--quick` runs a tiny version in `quick/`.
  - `--cpu-workers N` and `--gpu-workers N` set how many runs train at once.
  - `progress` redraws one line showing how far training has got, for a second window. It only reads the logs and
    stage files and never loads torch. That's why names and paths live in `names.py`, and `driver.py` imports
    torch, pandas, `trainer` and `testing` inside the tasks that use them.
- `python diagram.py` draws the report's two diagrams, the learning loop and the study pipeline, into `plots/`. Their
  numbers come from the settings in `driver.py`, `testing.py` and `trainer.py`, and from SB3's defaults.
- `python testing.py <model.zip> [--log] [--render] [--conditions heavy_wind ...]` tests one model.
- Without `--log`, testing only prints a table. With `--log` it also:
  - appends to `results/test_results.csv`;
  - rewrites `summary.csv`;
  - skips models that are already logged.

**Outputs:**
- `logs/<name>.csv`: every training landing with its planet.
- `resources/<algo>/continue/`: the state needed to train past the last stage. It holds one state file per run:
  episodes, steps, training seconds and the planet-generator state. The SAC replay buffers that used to sit here
  were deleted on 2026-10-14.
- `results/`: the test results, plus `peaks.csv`, which shows where each setting peaks and how much it loses after.
- `plots/`: the charts and the two diagrams. Charts 7 and 8 are drawn from `logs/`, which isn't committed, so only
  the PC can redraw them.
- `observations/`: the user's notes on the charts (`my observations.txt`) and Claude's review of them.
- `submit/`: the report, as `LAB CA-Report.docx` and `.pdf` (version 2) plus `LAB CA-Report v1.*`. Next to it:
  - the course's template and rubric;
  - `report plan.md`, which also holds the demo commands and viva questions;
  - `best performers.md`;
  - the report's two landing pictures.

`.gitignore` here skips `.venv/`, `__pycache__/`, `quick/`, `resources/`, `logs/`, `*.zip` and `*.pkl`, so models,
replay buffers and training logs aren't committed. `results/` and `plots/` can be committed.

## Not obvious from the code

- **Saving the planet sequence.** SB3's DummyVecEnv resets the environment itself before the callback sees the
  episode end, so by then the wrapper has already drawn the next planet. That's why the wrapper saves `rng_state`
  before each draw and `train()` stores it in `continue/<name>_state.json`.
- **Training past 4000 episodes.** The carry-on path:
  - reloads the last stage's zip, plus the replay buffer for SAC;
  - calls `learn(reset_num_timesteps=False)`;
  - continues the planet sequence exactly. This was checked against a run that was never stopped.

  The SAC replay buffers were deleted on 2026-10-14 at the user's request, so:
  - PPO can still carry on: `driver.train_all(algos=["ppo"], stages=driver.STAGES + [8000])`.
  - **Don't add 8000 to `STAGES` for SAC.** Without its buffer, `can_continue` in `train()` is False, so the run
    starts again from episode 0. That overwrites its stage zips, its log and its state file, and the report's
    training hours come from the state files. Change the code first.
- **Stopping and carrying on.** `train()` saves its progress at every stage, not only at the end, through
  `save_progress`: the log flush, the SAC buffer and the state json.
  - After Ctrl+C, a crash or a restart, the same command carries on each unfinished run from its last stage. Only
    the work since that stage is lost.
  - `save_progress` removes the state file first and writes the buffer and state through `.tmp` plus `os.replace`.
    A crash mid-save therefore means a fresh start, never a mismatched buffer.
  - On carry-on, the log is trimmed to `rows[:start + 1]`, because a crash can leave half a row.
  - A carried-on run meets the same planets as one that never stopped, but its training noise differs.
- **Scheduling.**
  - SAC runs with a 256-wide network go to a GPU pool of 2 workers (`GPU_WORKERS = 2`).
  - Everything else goes to a CPU pool, with one torch thread per worker.
  - When the CPU runs out of its own work, it takes big SAC runs, but only while
    `len(gpu_queue) > 2 * gpu_workers`.
  - Train with `--gpu-workers 0`, which gives 11 CPU workers. Under full load the GPU is no faster than a CPU thread,
    because each kernel launch costs 30-80 us under Windows.
- **Outcomes.**
  - landed: the landing ended and the last reward was above 0, meaning the lander came to rest (+100);
  - crashed: any other ending, including flying off screen;
  - timeout: 1000 steps.
- **Batched testing.** Testing flies all the landers side by side with one batched `predict`, which is 3.4x faster.
  `--render` flies them one at a time, which can differ slightly because of rounding. In one check, 1 landing in 48
  came to rest 4 steps later.
- **Gravity precision.** Box2D stores gravity as float32, so compare it with a tolerance.
- **pandas 3:** avoid chained assignment. `summary.csv` rounds only the metric columns; rounding everything turned
  lr 3e-4 into 0.0.
- **Chart colours.** The colours in `plots.py` passed a colour-blind validation. Keep them if you edit the charts.

## Status (2026-10-14)

The study is done: all 432 models are trained, tested and charted. The results are in `results/` and the charts
in `plots/`, and the report is in `submit/`. The user commits them.

**Training.**
- It ran with `--cpu-workers 11 --gpu-workers 0` from 2026-10-12 10:18 to 2026-10-13 14:43. It was stopped
  part-way and carried on with the same command.
- The first attempt, started at 03:18 on 2026-10-12 with `--gpu-workers 4`, was lost to a Windows Update restart
  at 05:17. That's why runs save their progress at every stage.
- All 72 runs were trained on the PC. The Mac trained none of them, so `mac.md`, the plan for training on the Mac
  at the same time, was deleted.
- The 36 SAC replay buffers (`resources/sac/continue/*_buffer.pkl`, 3 GB) were deleted on 2026-10-14. The report
  never used them. The 72 state files are kept, because the report's training hours come from them.
- Checked on 2026-10-13: all 432 zips pass their CRC check, each zip's `num_timesteps` equals its log's `steps`
  at that episode, every log has episodes 1-4000, and every state file says 4000.

**Measured time per run** (one CPU thread each, up to 11 runs at once, from `seconds` in the state files):

| Runs | Hours per run | Steps per run (mean) |
|---|---|---|
| PPO, each size | 0.7 (0.2-1.1) | 1.3-1.8M |
| SAC 16x16 | 6.6 (5.1-8.5) | 2.4M |
| SAC 64x64 | 4.7 (3.7-5.7) | 1.4M |
| SAC 256x256 | 9.1 (8.1-10.1) | 1.0M |

- That's 271 run-hours in all.
- SAC 16x16 takes the most steps: 38% of its training landings hover until the 1000-step limit, against 2% for
  SAC 256x256.
- `SAC_HOURS` and `PPO_HOURS` in `driver.py`, which weight the progress bar, are still the guesses made before
  training.

**Testing.** `test --log` took about 10 minutes with 11 workers, for 129,600 landings. `plots` takes seconds.

**Findings.**
- SAC beats PPO. The best setting is SAC 256x256 with lr 3e-3 at 4000 episodes: mean return 275, and 96% of its
  landings come to rest.
- Landed at 4000 episodes: SAC 80-90% in every condition but low gravity (61%); PPO 46-68%, and 42% in low
  gravity. Low gravity, which training never sees, is the hardest condition.
- Performance only drops off for PPO with a high learning rate on a bigger network:
  - 256x256 at lr 3e-3 never learns: it ends at -968 and lands none;
  - 256x256 at lr 1e-3 peaks at 2000 episodes (169), then falls to -355;
  - 64x64 peaks at 250 episodes with lr 3e-3, and at 1000 with lr 1e-3.
- SAC 64x64 at lr 1e-3 and 3e-3 dips 4 and 20 points after 2000. The other 18 of the 24 settings are at their
  best at 4000.

**Which setting matters most depends on the measure.** Share of the spread between all 432 models:

| Setting | Mean return (the charts) | Landed % |
|---|---|---|
| training episodes | 12% | 40% |
| algorithm | 12% | 8% |
| learning rate | 7% | 2% |
| network size | 1% | 4% |
| settings combined | 45% | 29% |
| seed (luck) | 23% | 16% |

- The PPO collapses, near -1000 against +275 for the best, dominate the mean-return shares, because the analysis
  squares each model's distance from the average.
- Among the 72 final models alone, the algorithm matters most on both measures. On landed % it explains 21% of
  the spread, against 15% for the learning rate and 1% for network size.

**Report.** The study is the user's RL Lab CA mini project (rubric: 25 marks). The user frames it as a
comparative study of how hyperparameters affect PPO and SAC.
- `submit/LAB CA-Report.docx` and `.pdf` are version 2, from 2026-10-14. `LAB CA-Report v1.*` is the first
  version.
  - Both are built on the course template, `LAB CA-Report Format.docx`, and keep its header, styles and page
    setup.
  - Fonts: Times New Roman, plus Consolas for code.
- Version 2 changed:
  - **References:** only 3 remain: PPO, SAC, and Henderson et al. 2018.
  - **Figure 7 deleted:** chart 8, steps per training landing. The user chose this; Table 7 and the
    training-return chart stay.
  - **Table 5:** gained SAC 4, which is SAC 64x64 at lr 3e-3 after 2000 episodes. It scores 264.3 in about a
    quarter of SAC 3's training time.
  - **Section 3.3:** now walks through the three commands.
  - **F1-F4:** gained the user's observations that were missing.
- The body (sections 1-4) must stay within the template's 15 pages. It ends on page 15, and the report is 24
  pages in all.
- It was generated by a python-docx script outside the repo, so later edits go into the .docx in Word.
- Still the user's to do:
  - fill in their name, roll number, batch and date of performance on page 1;
  - decide on the AI-use statement in Appendix F.

**Open idea, not decided:** training on to 8000 episodes, to see where the 18 settings still at their best at 4000
drop off. Since the SAC replay buffers are gone, only PPO can carry on exactly. SAC needs a code change first (see
"Training past 4000 episodes").

Any change to the design needs a plan and the user's "go" first.
