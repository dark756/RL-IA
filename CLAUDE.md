# PPO vs SAC on LunarLander

A study comparing PPO and SAC on LunarLander, built by Claude on 2026-10-12. `README.md` here is the full guide for
the user.

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
- `python driver.py train | test [--log] | plots`.
  - `--quick` runs a tiny version in `quick/`.
  - `--cpu-workers N` and `--gpu-workers N` set how many runs train at once.
- `python testing.py <model.zip> [--log] [--render] [--conditions heavy_wind ...]` tests one model.
- Without `--log`, testing only prints a table. With `--log` it also:
  - appends to `results/test_results.csv`;
  - rewrites `summary.csv`;
  - skips models that are already logged.

**Outputs:**
- `logs/<name>.csv`: every training landing with its planet.
- `resources/<algo>/continue/`: the state needed to train past the last stage.
- `results/`: the test results, plus `peaks.csv`, which shows where each setting peaks and how much it loses after.
- `plots/`: the charts.

`.gitignore` here skips `.venv/`, `__pycache__/`, `quick/`, `resources/`, `logs/`, `*.zip` and `*.pkl`, so models,
replay buffers and training logs aren't committed. `results/` and `plots/` can be committed.

## Not obvious from the code

- **Saving the planet sequence.** SB3's DummyVecEnv resets the environment itself before the callback sees the
  episode end, so by then the wrapper has already drawn the next planet. That's why the wrapper saves `rng_state`
  before each draw and `train()` stores it in `continue/<name>_state.json`.
- **Training past 4000 episodes.** Add 8000 to `STAGES` in `driver.py`. This uses the same carry-on path. The run then:
  - reloads the 4000 zip, plus the replay buffer for SAC;
  - calls `learn(reset_num_timesteps=False)`;
  - continues the planet sequence exactly. This was checked against a run that was never stopped.
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

## Status (2026-10-12)

The study is built and tested on the Mac. On the PC, the venv is set up and the quick check passed.

**First full attempt.**
- It started at 03:18 with `--gpu-workers 4`.
- At 05:17, Windows Update restarted the PC to install KB5124010. The active hours are 09:00-03:00, so Windows
  restarts between 03:00 and 09:00.
- No run had finished, so nothing was kept, and the stale files were deleted.
- Since then, runs save their progress at every stage (see "Stopping and carrying on"). This was tested by killing a
  SAC run and a PPO run after episode 20. Both carried on with identical planets and logs numbered 1-30.

**Measured speed on the PC** (steps/s per run, one torch thread each):

| Run | Alone | Under full load (11 runs at once) |
|---|---|---|
| SAC 256x256, cuda | 34 | 27 with 4 on the GPU and 7 on the CPU (real run) |
| SAC 256x256, cpu | 39 | 29 (benchmark) |
| SAC 64x64, cpu | - | 59-60 (real run) |

- Quick-run `done` lines gave PPO at 280-326 steps/s and SAC 16x16 at 64-98 steps/s, but these include start-up.
- The GPU is slow per operation: each kernel launch costs 30-80 us under Windows. Under full load it's no faster
  than a CPU thread.

**Steps per run.** A SAC run is about 1.5-2.4M steps, not the 1.15M assumed before:
- SAC takes 270-355k steps for its first 500 episodes;
- landers that have learned still take 330-400 steps a landing in the wind;
- stuck ones hover until the 1000-step limit for hundreds of episodes.

PPO hasn't been measured beyond the quick run.

**Full-run estimate:**
- `--gpu-workers 0`, which gives 11 CPU workers by default: about 40 h (35-52 h). This is the recommended setting.
- `--gpu-workers 4`: about 55 h. The CPU only takes GPU runs while more than `2 * gpu_workers` are waiting, and with
  4 workers that never happens (12 - 4 = 8). So the GPU does all 12 big SAC runs in 3 rounds of about 18 h.

Next, the user pauses Windows Update for a week, then starts `.\.venv\Scripts\python driver.py train --gpu-workers 0`
in their own PowerShell window. After that come `test --log` and then `plots`.

Any change to the design needs a plan and the user's "go" first.
