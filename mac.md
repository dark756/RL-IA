# Training on the Mac alongside the PC

This file is for you and for Claude on the Mac. `CLAUDE.md` covers the study itself. This file covers the Mac's job, and
where they differ it takes priority over `CLAUDE.md`, which says the Mac is for small tests only.

## The idea

The PC has been training the full study since 2026-10-12 10:18 with
`driver.py train --cpu-workers 11 --gpu-workers 0`. It works through the 72 runs in a fixed order that can't be
changed while it runs:
1. SAC 256x256, then SAC 64x64, then SAC 16x16, 12 runs each;
2. PPO 256x256, then PPO 64x64, then PPO 16x16, 12 runs each.

Within each group the runs go by learning rate (1e-4, 3e-4, 1e-3, 3e-3), then seed (0, 1, 2).

**How the work is split:**
- The Mac trains the same list **backwards**, starting with `ppo_lr3e-3_net16x16_seed2`, the run the PC does last.
- Finished Mac runs are copied into the PC's folders.
- When the PC reaches a run, `train()` finds its 6 stage files and skips it.

**Timing:**
- The PC reaches the PPO runs 31-40 h after its start, so not before about 14:00 on Oct 13.
- This saves about 2-3 h of the PC's ~40 h.

**Only the Mac's copy of the code changes.** The PC's code stays as it is.

## The Mac

- MacBook Pro with an i5-7360U: 2 cores, 4 threads and 8 GB of RAM. That allows **3 runs at once**
  (`--cpu-workers 3`).
- **Libraries:** torch 2.2.2 and numpy 1.26, the newest an Intel Mac can run, plus SB3 2.8.0, gymnasium[box2d] 1.2.3,
  pandas 3.0.2 and matplotlib 3.10.8. It needs Python 3.11 or 3.12.
- **Differences from the PC**, which has torch 2.11 and numpy 2.4:
  - Mac runs won't match PC runs bit for bit.
  - They use the same algorithm, settings and planets: numpy's seeded generator gives the same planet sequence on
    both machines.
  - Each Mac run's state file records its torch version (see step 3).

## 1. Get the code there

On the PC, in PowerShell:
```powershell
cd C:\Users\gurka\Desktop\RL-IA
Compress-Archive -Force -Path driver.py, names.py, trainer.py, testing.py, plots.py, README.md, CLAUDE.md, mac.md, .gitignore -DestinationPath $HOME\Desktop\rl-ia-code.zip
```
Move `rl-ia-code.zip` to the Mac. It's small, so emailing it to yourself works. Then, on the Mac:
```bash
mkdir -p ~/RL-IA && cd ~/RL-IA && unzip ~/Downloads/rl-ia-code.zip
```

## 2. Set up the venv

```bash
cd ~/RL-IA
python3 --version          # must be 3.11 or 3.12
python3 -m venv .venv
source .venv/bin/activate
pip install torch==2.2.2 "numpy<2"
pip install --no-deps stable-baselines3==2.8.0
pip install "gymnasium[box2d]==1.2.3" pandas==3.0.2 matplotlib==3.10.8 "numpy<2"
```
- If Box2D fails to build, run `xcode-select --install`, then the last line again.
- Run `cd ~/RL-IA && source .venv/bin/activate` in every new Terminal window.

## 3. Code changes, in the Mac's copy only

Claude on the Mac makes these. Match the style of the code around them, print ASCII only, and write files with
`encoding="utf-8"`.

1. **`train --reverse`** in `driver.py`:
   - Add a `reverse=False` parameter to `train_all`, and the flag `--reverse` (`action="store_true"`, help "train:
     start with the runs the PC does last").
   - Right after `todo.sort(...)  # longest runs first`, add `if reverse: todo.reverse()`.
   - The Mac has no CUDA, so every run goes to the CPU queue in that order.
2. **`pack`** in `driver.py`: `python driver.py pack [--quick]` writes `<root>/handover.zip` with
   `zipfile.ZIP_DEFLATED`. Like `progress`, it must not import torch.
   - It includes every run of the study, using the same run list as `train_all`, that is **finished**:
     - all its stage files exist (`names.finished`);
     - `resources/<algo>/continue/<name>_state.json` has `"episodes"` equal to the last stage.
     - The state file is written last at each stage, so this leaves out a run that is still saving.
   - For each run, it adds these files in this order, as paths relative to the root with `/` separators:
     1. `logs/<name>.csv`;
     2. `resources/<algo>/continue/<name>_state.json`;
     3. `resources/<algo>/continue/<name>_buffer.pkl`, if it exists (SAC only);
     4. `resources/<algo>/<name>_ep<stage>.zip` for each stage, in stage order.
   - It rewrites the zip each time with every finished run, then prints how many runs it packed and their names.
3. **`trainer.py`**: in `save_progress`, add `"torch": torch.__version__` to the state json. Mac runs then show
   2.2.2. PC runs show nothing, because the PC's state files don't have the key.

**Untested PC change:** the last change made on the PC hasn't run yet.
- `names.py` holds the model names and file paths, moved out of `trainer.py`.
- `driver.py` imports torch, pandas, trainer and testing inside the tasks that use them, so `progress` doesn't load
  torch.
- Step 4 is the first run of this code. If it breaks, fix it here and write down the exact fix: the PC needs the
  same fix before it runs `test`.

## 4. Quick check, about 15-20 minutes

On the Mac:
```bash
python driver.py train --quick --cpu-workers 3
python driver.py test --quick --log --cpu-workers 3
python driver.py plots --quick
python driver.py pack --quick
```
- `pack --quick` should report 8 runs in `quick/handover.zip`. Never copy that file to the PC: its runs stop at
  episode 30.

**Check that the PC can load models made on the Mac.**
1. Copy these two files from the Mac's `~/RL-IA/quick/resources/` to the PC's Downloads folder:
   - `ppo/ppo_lr3e-4_net16x16_seed0_ep30.zip`
   - `sac/sac_lr3e-4_net256x256_seed0_ep30.zip`
2. On the PC, run them in PowerShell. `OMP_NUM_THREADS=1` keeps them to one thread, so training barely notices:
   ```powershell
   cd C:\Users\gurka\Desktop\RL-IA
   $env:OMP_NUM_THREADS = "1"
   .\.venv\Scripts\python testing.py $HOME\Downloads\ppo_lr3e-4_net16x16_seed0_ep30.zip --conditions ideal
   .\.venv\Scripts\python testing.py $HOME\Downloads\sac_lr3e-4_net256x256_seed0_ep30.zip --conditions ideal
   ```
3. What to expect:
   - Each command prints a table.
   - The first also writes the 300 test planets to `resources/envs/`. That's fine: `test` would write the same
     ones.
   - **If either model fails to load, stop here**, because the Mac's runs would be no use on the PC.
4. Afterwards, delete `quick/` on the Mac and the two zips on the PC.

## 5. Start training on the Mac

```bash
cd ~/RL-IA && source .venv/bin/activate
caffeinate -i python driver.py train --cpu-workers 3 --reverse
```
- **Before you leave it:**
  - Keep the Mac plugged in with the lid open.
  - Close Chrome and other big apps.
  - Leave `caffeinate -i` in the command: it stops the Mac sleeping while training runs.
- **What to see:** the first stage lines should name `ppo_lr3e-3_net16x16_seed2`, `..._seed1` and `..._seed0`.
- **After Ctrl+C or a crash:** run the same command again and each run carries on from its last stage, as on the PC.
- **Progress:** `progress` doesn't help on the Mac, because it assumes the PC's speed and the whole study.

## 6. Hand finished runs to the PC

1. On the Mac:
   ```bash
   python driver.py pack
   ```
2. Copy `~/RL-IA/handover.zip` to the PC's Downloads folder, by USB stick or cloud drive.
3. On the PC, in PowerShell:
   ```powershell
   cd C:\Users\gurka\Desktop\RL-IA
   $zip = "$HOME\Downloads\handover.zip"
   $tmp = Join-Path $env:TEMP "handover"
   Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
   Expand-Archive $zip $tmp
   $stages = 125, 250, 500, 1000, 2000, 4000
   foreach ($log in Get-ChildItem "$tmp\logs\*.csv") {
       $name = $log.BaseName; $algo = $name.Split("_")[0]
       $zips = foreach ($ep in $stages) { "resources\$algo\${name}_ep$ep.zip" }
       if (@($zips | Where-Object { -not (Test-Path "$tmp\$_") }).Count) { "$name  not finished in the zip, left out"; continue }
       if (-not @($zips | Where-Object { -not (Test-Path $_) }).Count) { "$name  already here"; continue }
       if ((Test-Path "logs\$name.csv") -or (Test-Path "resources\$algo\${name}_*") -or (Test-Path "resources\$algo\continue\${name}_*")) { "$name  SKIPPED: the PC has started it"; continue }
       New-Item -ItemType Directory -Force "resources\$algo\continue" | Out-Null
       Copy-Item $log.FullName logs\
       Copy-Item "$tmp\resources\$algo\continue\${name}_*" "resources\$algo\continue\"
       foreach ($file in $zips) { Copy-Item "$tmp\$file" "resources\$algo\" }
       "$name  added"
   }
   Remove-Item $tmp -Recurse -Force
   ```

**What the PowerShell command does:**
- It copies a run only if all 6 of its stage files are in the zip and the PC has none of its files yet.
- It copies the stage files last, so the PC never sees a half-copied run as finished.
- Repeating it is safe: runs already copied show as `already here`.

**When:** do it at least once **by 14:00 on Oct 13**, before the PC can reach the PPO runs. Repeat whenever more
Mac runs have finished.

## 7. When to stop the Mac

The PC has caught up when either of these happens:
- the hand-over prints `SKIPPED: the PC has started it`;
- the PC's window shows a line for a run the Mac is still training.

Then stop the Mac with Ctrl+C and do one last pack and hand-over. Any runs the Mac hadn't finished are dropped.

## Don't

- Don't run `train` on the Mac without `--reverse`.
- Don't copy `quick/handover.zip`, or single run files, into the PC's folders.
- Don't carry on PC runs on the Mac: their zips use numpy 2 (see `CLAUDE.md`).
- Don't change the PC's code. The user wants changes only in the Mac's copy.
