# Best performers: top 3 SAC and top 3 PPO settings

Source: `results/summary.csv` (the logged test of all 432 models), plus training time from
`resources/<algo>/continue/*_state.json`.

## How they were picked

- A **setting** is one algorithm, learning rate and network size, trained with 3 seeds (0, 1, 2).
- Each setting is scored at its best training stage by its **mean test return**. That's the mean over all
  300 test planets, averaged over the 3 seeds.
- All six winners are at their best at **4000 training episodes**, the last stage.
- Ranking by landing rate instead picks the same three settings for each algorithm. Only PPO's first two
  swap places (85.5% vs 85.0% landed).

| Rank | Setting | Mean return | Landed % | Solved % |
|---|---|---|---|---|
| SAC 1 | SAC, 256x256, lr 3e-3 | 275.1 | 96.2 | 96.1 |
| SAC 2 | SAC, 256x256, lr 1e-3 | 271.4 | 96.0 | 96.1 |
| SAC 3 | SAC, 256x256, lr 1e-4 | 264.8 | 95.3 | 95.7 |
| PPO 1 | PPO, 256x256, lr 1e-4 | 220.4 | 85.0 | 83.7 |
| PPO 2 | PPO, 256x256, lr 3e-4 | 212.3 | 85.5 | 80.8 |
| PPO 3 | PPO, 64x64, lr 1e-4 | 209.3 | 79.5 | 78.2 |

**Runners-up.**
- SAC 4th is SAC 64x64 at lr 3e-3, which peaks at 2000 episodes: return 264.3, 93.0% landed. That's only
  0.5 below SAC 3, a tie within seed noise. It also needs about a quarter of the training time.
- PPO 4th is PPO 64x64 at lr 3e-4: return 175.3, 73.3% landed. That's well behind PPO 3.

## Terms

- **Mean return:** the total reward of a landing, averaged over the landings and then over the 3 seeds.
  200 or more counts as solved.
- **Seed range:** the worst and best of the 3 seeds' mean returns.
- **SD per landing:** how much the return varies from one landing to the next within that condition,
  averaged over the seeds. Lower means more consistent.
- **Outcomes:**
  - landed: the lander came to rest;
  - crashed: it hit the ground hard or flew off screen;
  - timeout: it was still flying after 1000 steps.
- **Solved % and landed % aren't the same.**
  - A lander that sits on its legs without coming fully to rest before the 1000-step limit can still score
    200 or more. It counts as solved but timed out, which is why solved % is sometimes above landed %.
  - A slow or wasteful landing can come to rest with less than 200. That counts as landed but not solved.

## The test conditions (50 fixed planets each)

| Condition | Gravity | Wind | Turbulence | Seen in training? |
|---|---|---|---|---|
| ideal | -10 | 0 | 0 | yes |
| moderate wind | -10 | 5-10 | 0.5-1 | yes |
| heavy wind | -10 | 15-20 | 1.5-2 | yes, rarely (about 3% of training landings have wind 15-20) |
| heavy gravity | -11 to -11.9 | 0 | 0 | yes (14% of training landings have gravity -11 to -11.9) |
| low gravity | -2 to -4 | 0 | 0 | **no** |
| random mix | -2 to -11.9 | 0-20 | 0-2 | partly |

## Side by side

**Landed %**

| Condition | SAC 1 | SAC 2 | SAC 3 | PPO 1 | PPO 2 | PPO 3 |
|---|---|---|---|---|---|---|
| ideal | 100.0 | 100.0 | 100.0 | 92.7 | 94.7 | 86.0 |
| moderate wind | 99.3 | 99.3 | 100.0 | 92.7 | 89.3 | 86.7 |
| heavy wind | 98.0 | 98.7 | 98.0 | 81.3 | 78.0 | 69.3 |
| heavy gravity | 98.7 | 99.3 | 99.3 | 86.0 | 85.3 | 83.3 |
| low gravity | 85.3 | 84.0 | 82.7 | 74.0 | 77.3 | 69.3 |
| random mix | 96.0 | 94.7 | 92.0 | 83.3 | 88.7 | 82.0 |
| **all 300** | **96.2** | **96.0** | **95.3** | **85.0** | **85.5** | **79.5** |

**Mean return**

| Condition | SAC 1 | SAC 2 | SAC 3 | PPO 1 | PPO 2 | PPO 3 |
|---|---|---|---|---|---|---|
| ideal | 280.8 | 280.9 | 274.1 | 240.8 | 235.7 | 224.2 |
| moderate wind | 286.2 | 285.8 | 281.4 | 245.9 | 226.4 | 228.9 |
| heavy wind | 279.7 | 276.1 | 270.4 | 205.3 | 184.1 | 181.6 |
| heavy gravity | 279.0 | 280.1 | 274.0 | 230.3 | 221.1 | 219.8 |
| low gravity | 253.2 | 241.8 | 247.7 | 186.0 | 190.9 | 187.2 |
| random mix | 272.0 | 263.6 | 241.4 | 214.0 | 215.3 | 214.0 |
| **all 300** | **275.1** | **271.4** | **264.8** | **220.4** | **212.3** | **209.3** |

## What stands out

- **The three SAC settings are nearly identical in four conditions.** In ideal, both winds and heavy
  gravity they land 98-100% of the time, with returns of 270-286. They only differ in low gravity and the
  random mix.
- **Low gravity is the weak spot for all six.** It's the one condition training never saw.
  - SAC lands 83-85% there, against 98-100% elsewhere. Its failures are split between hovering until the
    time limit (8-14%) and crashing (3-8%).
  - Landings take much longer there: SAC averages 400-547 steps, against about 190-245 in the other
    single conditions.
  - The return varies much more from landing to landing there. SAC's SD is 74-108 in low gravity,
    against 16-39 in ideal, both winds and heavy gravity.
- **PPO's weak spot is heavy wind.** From ideal to heavy wind:
  - PPO's landing rate falls by 11-17 points (PPO 1: 92.7% to 81.3%; PPO 3: 86.0% to 69.3%);
  - SAC's falls by only 1-2 points.
- **They fail in different ways.**
  - PPO fails mostly by crashing: 5-30% crashed depending on the condition, against 0-9% timeouts.
  - SAC fails mostly in low gravity, and there it hovers more often than it crashes: 48 timeouts against 24
    crashes, out of 450 low-gravity landings across the three settings.
- **The seeds agree more for SAC.** SAC's three seeds end within 15 points of each other overall
  (262.7-277.9 at most). PPO's spread is up to 37 points (PPO 2: 197.0-234.2).
- **Settings in common.**
  - All three SAC winners are 256x256, and any learning rate works for them.
  - All three PPO winners use a low learning rate (1e-4 or 3e-4, SB3's default). Two of them are 256x256.
- **Cost.**
  - SAC: 8.1-10.1 hours of one CPU thread per run, for 0.83-1.24M steps.
  - PPO: 0.8-1.0 hours per run, for 1.53-1.91M steps.
  - So PPO used about 1.7 times as many environment steps but about a tenth of the time.

## Per-setting breakdown (4000 training episodes)

Mean return and SD are averaged over the 3 seeds. Percentages are of 150 landings per condition (50 per
seed), or 900 for "all".

### SAC 1: SAC, 256x256, lr 3e-3

| Condition | Mean return | Seed range | SD per landing | Landed % | Crashed % | Timeout % | Solved % | Mean steps |
|---|---|---|---|---|---|---|---|---|
| ideal | 280.8 | 278.4 to 282.6 | 16.3 | 100.0 | 0.0 | 0.0 | 100.0 | 193 |
| moderate wind | 286.2 | 282.9 to 288.8 | 19.3 | 99.3 | 0.0 | 0.7 | 99.3 | 192 |
| heavy wind | 279.7 | 272.3 to 284.5 | 33.0 | 98.0 | 1.3 | 0.7 | 98.0 | 213 |
| heavy gravity | 279.0 | 274.8 to 282.7 | 22.0 | 98.7 | 0.7 | 0.7 | 98.7 | 189 |
| low gravity | 253.2 | 247.4 to 259.1 | 80.0 | 85.3 | 4.7 | 10.0 | 84.7 | 457 |
| random mix | 272.0 | 266.8 to 275.4 | 53.8 | 96.0 | 1.3 | 2.7 | 96.0 | 286 |
| **all** | **275.1** | **273.8 to 276.6** | 45.9 | **96.2** | 1.3 | 2.5 | 96.1 | 255 |

| Model file (`resources/sac/`) | Mean return | Landed % | Training steps | Training hours |
|---|---|---|---|---|
| `sac_lr3e-3_net256x256_seed0_ep4000.zip` | 276.6 | 97.0 | 825,532 | 8.6 |
| `sac_lr3e-3_net256x256_seed1_ep4000.zip` | 275.1 | 94.7 | 923,221 | 9.8 |
| `sac_lr3e-3_net256x256_seed2_ep4000.zip` | 273.8 | 97.0 | 912,992 | 8.1 |

### SAC 2: SAC, 256x256, lr 1e-3

| Condition | Mean return | Seed range | SD per landing | Landed % | Crashed % | Timeout % | Solved % | Mean steps |
|---|---|---|---|---|---|---|---|---|
| ideal | 280.9 | 278.8 to 283.6 | 16.1 | 100.0 | 0.0 | 0.0 | 100.0 | 187 |
| moderate wind | 285.8 | 283.9 to 289.0 | 20.6 | 99.3 | 0.0 | 0.7 | 99.3 | 200 |
| heavy wind | 276.1 | 268.2 to 283.8 | 35.8 | 98.7 | 1.3 | 0.0 | 98.7 | 215 |
| heavy gravity | 280.1 | 277.0 to 284.7 | 22.6 | 99.3 | 0.7 | 0.0 | 99.3 | 190 |
| low gravity | 241.8 | 215.5 to 257.2 | 108.2 | 84.0 | 8.0 | 8.0 | 84.7 | 400 |
| random mix | 263.6 | 251.1 to 270.8 | 64.4 | 94.7 | 2.0 | 3.3 | 94.7 | 310 |
| **all** | **271.4** | **262.7 to 277.9** | 57.8 | **96.0** | 2.0 | 2.0 | 96.1 | 250 |

| Model file (`resources/sac/`) | Mean return | Landed % | Training steps | Training hours |
|---|---|---|---|---|
| `sac_lr1e-3_net256x256_seed0_ep4000.zip` | 262.7 | 93.0 | 919,594 | 8.8 |
| `sac_lr1e-3_net256x256_seed1_ep4000.zip` | 273.6 | 97.7 | 944,749 | 9.1 |
| `sac_lr1e-3_net256x256_seed2_ep4000.zip` | 277.9 | 97.3 | 901,139 | 9.2 |

### SAC 3: SAC, 256x256, lr 1e-4

| Condition | Mean return | Seed range | SD per landing | Landed % | Crashed % | Timeout % | Solved % | Mean steps |
|---|---|---|---|---|---|---|---|---|
| ideal | 274.1 | 271.6 to 277.8 | 17.6 | 100.0 | 0.0 | 0.0 | 100.0 | 211 |
| moderate wind | 281.4 | 278.4 to 284.6 | 20.6 | 100.0 | 0.0 | 0.0 | 100.0 | 212 |
| heavy wind | 270.4 | 267.0 to 273.4 | 38.5 | 98.0 | 1.3 | 0.7 | 97.3 | 244 |
| heavy gravity | 274.0 | 272.0 to 275.6 | 18.5 | 99.3 | 0.0 | 0.7 | 99.3 | 206 |
| low gravity | 247.7 | 237.0 to 264.7 | 74.1 | 82.7 | 3.3 | 14.0 | 85.3 | 547 |
| random mix | 241.4 | 219.6 to 269.2 | 130.0 | 92.0 | 2.7 | 5.3 | 92.0 | 348 |
| **all** | **264.8** | **258.0 to 270.3** | 68.6 | **95.3** | 1.2 | 3.4 | 95.7 | 295 |

| Model file (`resources/sac/`) | Mean return | Landed % | Training steps | Training hours |
|---|---|---|---|---|
| `sac_lr1e-4_net256x256_seed0_ep4000.zip` | 270.3 | 95.7 | 1,103,194 | 9.1 |
| `sac_lr1e-4_net256x256_seed1_ep4000.zip` | 258.0 | 94.3 | 1,223,568 | 10.0 |
| `sac_lr1e-4_net256x256_seed2_ep4000.zip` | 266.1 | 96.0 | 1,240,829 | 10.1 |

### PPO 1: PPO, 256x256, lr 1e-4

| Condition | Mean return | Seed range | SD per landing | Landed % | Crashed % | Timeout % | Solved % | Mean steps |
|---|---|---|---|---|---|---|---|---|
| ideal | 240.8 | 235.3 to 249.3 | 62.4 | 92.7 | 7.3 | 0.0 | 92.7 | 261 |
| moderate wind | 245.9 | 241.0 to 254.1 | 67.8 | 92.7 | 7.3 | 0.0 | 92.7 | 282 |
| heavy wind | 205.3 | 185.4 to 223.0 | 106.7 | 81.3 | 18.0 | 0.7 | 77.3 | 280 |
| heavy gravity | 230.3 | 212.7 to 243.2 | 73.4 | 86.0 | 14.0 | 0.0 | 86.0 | 240 |
| low gravity | 186.0 | 148.4 to 212.9 | 126.4 | 74.0 | 22.7 | 3.3 | 72.0 | 452 |
| random mix | 214.0 | 209.8 to 221.8 | 111.6 | 83.3 | 16.0 | 0.7 | 81.3 | 320 |
| **all** | **220.4** | **214.9 to 223.7** | 97.8 | **85.0** | 14.2 | 0.8 | 83.7 | 306 |

| Model file (`resources/ppo/`) | Mean return | Landed % | Training steps | Training hours |
|---|---|---|---|---|
| `ppo_lr1e-4_net256x256_seed0_ep4000.zip` | 222.6 | 83.3 | 1,726,512 | 0.8 |
| `ppo_lr1e-4_net256x256_seed1_ep4000.zip` | 223.7 | 85.3 | 1,710,480 | 1.0 |
| `ppo_lr1e-4_net256x256_seed2_ep4000.zip` | 214.9 | 86.3 | 1,879,560 | 0.9 |

### PPO 2: PPO, 256x256, lr 3e-4

| Condition | Mean return | Seed range | SD per landing | Landed % | Crashed % | Timeout % | Solved % | Mean steps |
|---|---|---|---|---|---|---|---|---|
| ideal | 235.7 | 224.4 to 250.2 | 57.0 | 94.7 | 5.3 | 0.0 | 92.7 | 304 |
| moderate wind | 226.4 | 212.3 to 242.7 | 75.3 | 89.3 | 10.7 | 0.0 | 82.0 | 308 |
| heavy wind | 184.1 | 165.4 to 207.8 | 119.7 | 78.0 | 22.0 | 0.0 | 72.0 | 323 |
| heavy gravity | 221.1 | 206.1 to 245.0 | 80.2 | 85.3 | 14.7 | 0.0 | 83.3 | 267 |
| low gravity | 190.9 | 152.8 to 228.7 | 116.4 | 77.3 | 18.0 | 4.7 | 74.0 | 511 |
| random mix | 215.3 | 201.1 to 230.8 | 93.4 | 88.7 | 11.3 | 0.0 | 80.7 | 379 |
| **all** | **212.3** | **197.0 to 234.2** | 94.6 | **85.5** | 13.7 | 0.8 | 80.8 | 349 |

| Model file (`resources/ppo/`) | Mean return | Landed % | Training steps | Training hours |
|---|---|---|---|---|
| `ppo_lr3e-4_net256x256_seed0_ep4000.zip` | 197.0 | 83.0 | 1,563,261 | 0.9 |
| `ppo_lr3e-4_net256x256_seed1_ep4000.zip` | 205.6 | 83.3 | 1,531,998 | 0.9 |
| `ppo_lr3e-4_net256x256_seed2_ep4000.zip` | 234.2 | 90.3 | 1,543,960 | 0.8 |

### PPO 3: PPO, 64x64, lr 1e-4

| Condition | Mean return | Seed range | SD per landing | Landed % | Crashed % | Timeout % | Solved % | Mean steps |
|---|---|---|---|---|---|---|---|---|
| ideal | 224.2 | 199.9 to 250.2 | 77.6 | 86.0 | 12.7 | 1.3 | 85.3 | 289 |
| moderate wind | 228.9 | 206.2 to 254.6 | 80.3 | 86.7 | 12.7 | 0.7 | 86.0 | 310 |
| heavy wind | 181.6 | 159.3 to 203.8 | 112.6 | 69.3 | 30.0 | 0.7 | 66.7 | 289 |
| heavy gravity | 219.8 | 210.6 to 234.3 | 85.4 | 83.3 | 16.7 | 0.0 | 83.3 | 273 |
| low gravity | 187.2 | 161.0 to 211.6 | 116.1 | 69.3 | 21.3 | 9.3 | 67.3 | 545 |
| random mix | 214.0 | 208.8 to 222.4 | 101.1 | 82.0 | 16.7 | 1.3 | 80.7 | 371 |
| **all** | **209.3** | **191.6 to 229.5** | 98.1 | **79.5** | 18.3 | 2.2 | 78.2 | 346 |

| Model file (`resources/ppo/`) | Mean return | Landed % | Training steps | Training hours |
|---|---|---|---|---|
| `ppo_lr1e-4_net64x64_seed0_ep4000.zip` | 229.5 | 83.7 | 1,762,096 | 0.8 |
| `ppo_lr1e-4_net64x64_seed1_ep4000.zip` | 206.7 | 78.0 | 1,797,160 | 0.8 |
| `ppo_lr1e-4_net64x64_seed2_ep4000.zip` | 191.6 | 76.7 | 1,910,435 | 0.9 |

## Watching them land

The single best models are SAC 1 seed 0 (return 276.6) and PPO 1 seed 1 (return 223.7). To watch either
on a condition, run this from the project folder:

```
python testing.py resources/sac/sac_lr3e-3_net256x256_seed0_ep4000.zip --render --conditions low_gravity heavy_wind
python testing.py resources/ppo/ppo_lr1e-4_net256x256_seed1_ep4000.zip --render --conditions low_gravity heavy_wind
```

`--render` flies the landers one at a time, so a watched landing can differ slightly from the logged one.
