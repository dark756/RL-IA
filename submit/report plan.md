# Report plan: what goes into the RL Lab CA report

Written on 2026-10-13 and revised the same day after a critical review. Every number below was checked against:
- `results/`;
- `logs/`;
- the state files and saved models in `resources/`;
- `submit/best performers.md`, CLAUDE.md and `observations/claudes observations.txt`.

## Context
The report follows `submit/LAB CA-Report Format.docx` and is marked against `submit/LAB CA-Rubrics.docx.pdf`.
- **Length:** 10-15 pages, not counting references and the appendix.
- **Marks:** problem and formulation 5, implementation 10, results and viva 10.
- **Title:** "A Comparative Study of How Hyperparameters Affect PPO and SAC Agents for Autonomous Lunar Landing".

This file covers four things:
- how our material maps onto the template's sections;
- what to leave out, and what's still missing;
- how to demo the agents while presenting ("Presenting it");
- which viva questions to prepare.

The material:
- `results/`
- `submit/best performers.md`, the top 3 settings of each algorithm
- `plots/`
- `observations/`

## Rules for the whole report
- Take findings from `observations/claudes observations.txt`, the corrected versions, never from the raw notes.
- Give mean return and landed % together for every headline claim, since they rank the settings differently.
- Use aggregates only. Never paste `test_results.csv` (129,600 rows) or `summary.csv` (3,024 rows).
- Say why, not only what. Label each explanation as likely, unless it was measured.
- Compare SAC and PPO "per episode" and "per sample", never "per hour", because SAC took 7-13x the compute.
- One correction to `best performers.md`: the widest PPO seed spread is PPO 3 (191.6-229.5, 38 points), not
  PPO 2 (37 points).

## Section by section

### Header and 1. Title (about 0.3 pages)
Fill in your name, roll number, batch and date of performance, then the title above.

### 2. Problem definition and RL formulation (about 2.5 pages; criterion 1, 5 marks)
- **2.1 Problem statement.**
  - The agent must land safely on planets whose gravity, wind and turbulence vary and aren't known in advance. Those
    values aren't in its state, so it never sees them.
  - Measure how the algorithm, learning rate, network size and training length affect how well the agent lands.
  - Name the methods:
    - training on random planets is domain randomization [Tobin 2017; Peng 2018];
    - the unseen low-gravity planets test out-of-distribution generalization.
- **2.2 Motivation.**
  - Real world:
    - Chandrayaan-2's lander crashed in its final descent (2019), and Chandrayaan-3 landed (2023).
    - The Moon has about 0.17 of Earth's gravity and Mars about 0.38. The low-gravity tests (-2 to -4, against the
      default -10) are 0.2-0.4 of the default, which is between the Moon and Mars.
    - A hook for the reader: the default -10 is Earth-like. So the Moon-like planets are exactly the ones training
      never sees.
  - Why RL:
    - there's no dataset of correct thrust;
    - the decisive reward comes at the end (+100 for coming to rest, -100 for a crash);
    - the dynamics are nonlinear and disturbed by wind;
    - a simulator allows thousands of safe practice landings;
    - one policy can cover a whole range of planets.
  - Why compare hyperparameters:
    - deep RL is sensitive to them [Henderson 2018; Andrychowicz 2021];
    - one SAC run costs 4-10 CPU hours, so knowing what matters saves compute.
- **2.3 Objectives:**
  - formulate the task as an MDP and train PPO and SAC agents;
  - run a controlled sweep: 2 algorithms x 4 lrs x 3 sizes x 3 seeds = 72 runs, saved at 6 lengths = 432 agents;
  - test every agent on the same 300 planets, including unseen low gravity, and measure what matters (ANOVA) and
    where performance drops off;
  - choose and demonstrate the best agent of each algorithm.
- **2.4 Table 1: RL formulation** (the template's table).

  | Component | Content |
  |---|---|
  | Environment | LunarLanderContinuous-v3 (Gymnasium 1.2.3, Box2D, 50 steps/s), wrapped so that each landing gets a new random planet |
  | State | 8 numbers: x and y relative to the pad, x and y velocity, angle, angular velocity, and left and right leg contact. Gravity and wind are not in it, so the task is partially observed |
  | Action | 2 numbers in [-1, 1]: the main engine (off at 0 or below, 50-100% throttle above) and the side engines (off while abs < 0.5; the sign picks the side) |
  | Reward | Each step pays the change in shaping: -100 x distance, -100 x speed, -100 x tilt, +10 per leg down. Each frame costs 0.3 x the main engine's throttle and 0.03 x the side engine's throttle (each 0.5-1.0 while firing). +100 at rest, -100 for a crash or leaving the screen. 200 = solved |
  | Policy | PPO: a Gaussian (tanh MLP for the mean, learned log std); its sampled actions are clipped to [-1, 1]. SAC: a tanh-squashed Gaussian (ReLU MLP). The mean action is used at test time |
  | Discount factor | 0.99 |
  | Termination | comes to rest, crashes, leaves the screen, or 1000 steps |

- **Table 2: planets.**
  - The training draw: gravity N(-10, 1), wind abs(N(0, 7)), turbulence abs(N(0, 0.7)), from the README.
  - The 6 test conditions with "seen in training?", from `best performers.md` ("The test conditions").
- **2.5 Algorithms: PPO and SAC.**
  - Continuous actions rule out DQN and epsilon-greedy.
  - PPO is the robust on-policy standard. SAC is a sample-efficient off-policy method that maximises entropy.
  - Give one paragraph on each, with its objective in one line: PPO's clipped surrogate, and SAC's reward plus
    alpha x entropy with twin critics. The update steps go in 3.5.
  - Note that PPO's clip epsilon of 0.2 is not an exploration rate.

### 3. Implementation (about 4.7 pages; criterion 2, 10 marks)
- **3.1** Fig 1, `plots/diagram_rl_loop.png`, with a paragraph walking through the loop.
- **3.2** Tools:
  - Python 3.13.5, SB3 2.8.0, PyTorch 2.11.0, Gymnasium 1.2.3 (Box2D), NumPy 2.4, pandas 3.0.2 and matplotlib 3.10.8;
  - Windows 11 and VS Code;
  - an i5-12450H (12 threads), 16 GB of RAM and an RTX 3050. Training ran on the CPU, because under full load the
    GPU was no faster.
- **3.3** Fig 2, `plots/diagram_pipeline.png`, plus one short paragraph per module:
  - `trainer.py`: `GaussPlanetWrapper`, `StageSaver` and `train()`;
  - `driver.py`: 72 runs, 11 at once;
  - `testing.py`: the 300 fixed planets (seed 2026), batched deterministic landings, and the outcome rules;
  - `plots.py`: the charts and the ANOVA.

  Say plainly that PPO and SAC are Stable-Baselines3's implementations, used with their defaults so the comparison
  is fair. The viva will ask. This project's own implementation is:
  - the planet randomization;
  - the staged saves, with an exact resume;
  - the batched testing;
  - the analysis.

  The code snippets go in the appendix.
- **3.4 Table 3: hyperparameters** (the template's table).

  | Parameter | Value |
  |---|---|
  | Learning rate (alpha) | 1e-4, 3e-4, 1e-3, 3e-3; constant; all four tried |
  | Discount factor (gamma) | 0.99 |
  | Exploration rate (epsilon) | Not applicable: there's no epsilon-greedy with continuous actions. PPO samples its Gaussian, whose learned std falls from 1.0 to 0.12-0.33 by 4000 episodes (PPO 1, seed 1). SAC adds an entropy bonus with an auto-tuned temperature and a target entropy of -2; the temperature falls from 1.0 to 0.17 at 125 episodes and 0.04 at 4000 (SAC 1, seed 0). Both decay the way epsilon would |
  | Number of episodes | 4000 per run; weights saved at 125, 250, 500, 1000, 2000 and 4000 |
  | Other | Networks 16x16, 64x64 and 256x256; seeds 0-2. PPO: 2048 steps per update, batch 64, 10 epochs, GAE lambda 0.95, clip 0.2. SAC: buffer 1M, batch 256, tau 0.005, 1 gradient step per step, 2 critics |

- **3.5 Training procedure.**
  - A new planet is drawn for each landing (domain randomization).
  - **How the networks are updated.** The template asks for this explicitly.
    - PPO, every 2048 steps:
      - it computes GAE advantages (gamma 0.99, lambda 0.95) and normalizes them;
      - it takes 10 epochs x 32 minibatches of 64, which is 320 Adam steps. The loss is the clipped surrogate plus
        0.5 x the value MSE, with no entropy bonus and the gradient norm capped at 0.5;
      - then it throws the data away.
    - SAC, every step:
      - it stores the step in the 1M replay buffer, then samples 256 steps;
      - the critics regress to r + gamma x (1 - done) x (the min of the 2 target critics - alpha x log pi), taken at
        the next state with a fresh action from the policy;
      - the actor minimises alpha x log pi - min Q;
      - alpha is tuned to hold the entropy near -2;
      - the target critics follow the critics with tau 0.005.
  - Weights are saved at all 6 stages, so a run can resume.
  - The 72 runs, 11 at once, took 271 run-hours: about 28 hours on one laptop (12-13 Oct 2026).
  - Table 4 gives the hours and steps per run (from CLAUDE.md):

    | Runs | Hours per run | Steps |
    |---|---|---|
    | PPO, any size | 0.7 | 1.3-1.8M |
    | SAC 16x16 | 6.6 | 2.4M |
    | SAC 64x64 | 4.7 | 1.4M |
    | SAC 256x256 | 9.1 | 1.0M |

  - Testing 129,600 landings took about 10 minutes.
  - **Engineering challenges** (about 5 lines; criterion 2 rewards these decisions):
    - A Windows Update restart killed the first attempt after 2 hours. Since then, runs save at every stage, write
      their files atomically and resume the exact planet sequence.
      - To resume exactly, the planet generator's state is saved before each draw.
      - It has to be saved then because SB3 resets the environment, drawing the next planet, before the callback
        sees the landing end.
    - For these small networks the GPU was slower than the CPU, because each kernel launch costs 30-80 us under
      Windows. So training used 11 single-thread CPU workers.
    - Testing flies all 300 landers side by side, with one batched prediction per step. That's 3.4x faster.
- **3.6 Demonstration.**
  - Fig 3: screenshots of the best SAC model landing in heavy wind and in low gravity, plus a PPO model that crashes
    (256x256, lr 3e-3).
    - These still need capturing. The commands are D4 and D5 under "Presenting it", and the first landing of each
      shows the right thing.
  - Add the console table from D6, `python testing.py <best model>`.
  - Put the exact command under each screenshot and output.

### 4. Results and analysis (about 5.5-6 pages; criterion 3, 10 marks)
- **4.1 Table 5: every setting at 4000 episodes.** Each cell is the mean test return, with landed % in brackets.
  Source: `results/summary.csv`.

  | lr | PPO 16 | PPO 64 | PPO 256 | SAC 16 | SAC 64 | SAC 256 |
  |---|---|---|---|---|---|---|
  | 1e-4 | 121 (67%) | 209 (80%) | 220 (85%) | 202 (80%) | 234 (88%) | 265 (95%) |
  | 3e-4 | 112 (66%) | 175 (73%) | 212 (86%) | 135 (59%) | 246 (90%) | 256 (94%) |
  | 1e-3 | 161 (71%) | 83 (48%) | -355 (4%) | 78 (43%) | 258 (91%) | 271 (96%) |
  | 3e-3 | 149 (63%) | -187 (18%) | -968 (0%) | 149 (67%) | 244 (89%) | 275 (96%) |

- **4.1 Table 6: the top 3 of each algorithm.**
  - Use the rank table and the "Landed %" side-by-side from `best performers.md`, without the Solved % column.
  - Add one sentence: all six winners peak at 4000. The 4th-best SAC (64x64, lr 3e-3, 264.3 at 2000) ties SAC 3 with
    about a quarter of the training time.
  - Add one sentence: the winners' seed ranges don't overlap (SAC 1: 273.8-276.6; PPO 1: 214.9-223.7), so their gap
    isn't luck.
- **4.2 Metrics.** Define only the ones the text uses:
  - mean return: the cumulative reward of a landing, averaged;
  - landed %: the success rate;
  - crashed % and timeout %;
  - mean steps;
  - the seed range;
  - the ANOVA share (eta squared; see F3);
  - 200 as the solved reference line;
  - **convergence:** the mean test return changes by less than 10 points between 2000 and 4000 episodes;
  - **improvement**, from 125 to 4000 episodes: SAC 1 goes from -113 (25% landed) to 275 (96%), and PPO 1 from -201
    (0%) to 220 (85%).

  Leave out solved %: it nearly duplicates landed %.
- **4.3 Graphs** (the template's three, plus two more):

  | Fig | File | Shows |
  |---|---|---|
  | 4 | `7_training_return.png` | reward vs episode, during training |
  | 5 | `3_return_vs_episodes.png` | test return vs episode, and where performance drops off |
  | 6 | `6_landed_vs_episodes.png` | success rate vs episode |
  | 7 | `8_training_steps.png` | steps per episode vs episode |
  | 8 | `2_importance_by_condition.png` | which setting matters most, overall and per condition |

  - Captions:
    - Fig 5's y axis is stretched by the PPO collapses.
    - In Fig 8, compare down a column, not across, because each column is a share of its own condition's spread.
  - Charts 1 and 5 go to the appendix.
    - Chart 5 pools every setting, so PPO's bars include the collapsed corner.
    - Chart 1 has no 45% "settings combined" bar; that 45% is seven gray bars added up. So it seems to contradict
      Table 7.
  - Print size: Figs 4-7 are 10.4 inches wide, so at A4 text width their 9 pt labels print at about 5.5 pt. Use
    narrow margins (1.27 cm) and full width, which gives about 6.5 pt, or put two charts on each landscape page.
- **4.4 Analysis.** Answer the template's five questions. Label explanations as likely; only the evidence was
  measured.
  - **What did the agent learn?**
    - SAC 1 (256x256, lr 3e-3) lands on 96% of the 300 test planets: 100% in ideal, 98% in heavy wind and 85% in
      unseen low gravity. Its return is 275, with seeds at 274-277.
    - Fig 7 shows the phases:
      - PPO crashes fast (about 100-140 steps) around 125 episodes. It then hovers, peaking at 650-880 steps between
        250 and 1000 episodes, and by 4000 it lands in 300-475 steps.
      - SAC 64x64 and 256x256 hover from the start (300-900 steps), then come down to 190-330.
      - SAC 16x16 never leaves the hover phase: 37% of its last 500 training landings time out.
    - Exploration falls as they learn (Table 3): PPO's std goes from 1.0 to 0.12-0.33, and SAC's temperature from
      1.0 to 0.04.
  - **How did performance change during training?**
    - SAC 256x256 improves at every stage, at every lr. But one seed at lr 3e-3 fell to -563 at 250 episodes before
      recovering; that's the wide band in Fig 5.
    - PPO with lr 1e-3 or higher on 64x64 or 256x256 collapses. From `results/peaks.csv`:
      - 64x64 at lr 3e-3 peaks at 250 episodes;
      - 64x64 at lr 1e-3 peaks at 1000, at 136, and ends at 83;
      - 256x256 at lr 1e-3 peaks at 2000, at 169, and ends at -355;
      - 256x256 at lr 3e-3 never learns, and ends at -968.
    - The collapses also show in training (Fig 4), so testing didn't cause them.
  - **Did it converge?** By the definition in 4.2, 5 of the 24 settings stopped changing.
    - Four of them are good SAC settings: 64x64 at lr 3e-4 and 1e-3, and 256x256 at lr 1e-3 and 3e-3 (246-275).
    - The fifth, SAC 16x16 at lr 1e-3, stalled at 78 rather than converged: its seeds end at 21, -9 and 222.
    - SAC 64x64 at lr 3e-3 levelled off by 2000 and then dipped 20 (see F4).
    - The three SAC 64x64 and 256x256 settings with the lowest lrs still gain 14-45 points. The other three SAC
      16x16 settings gain 41-112.
    - Every PPO setting is still changing. The stable ones gain 11-54 points, and the unstable ones fall 30-524.
  - **What do the graphs show?** Write four findings, each with its evidence.
    - **F1: SAC beats PPO per episode and per sample** (Table 5, Fig 6).
      - The best PPO setting (220, 85%) roughly equals the average SAC setting (218, 82%). Every SAC 64x64 and
        256x256 setting beats it.
      - The best SAC setting used about half the steps of the best PPO setting: 0.89M against 1.77M.
      - Fairness: the budget was equal episodes, not equal compute.
        - PPO runs 7-13x faster: about 10x for the winners (0.8-1.0 hours against 8.1-9.8).
        - Per estimated CPU-minute, the two are about even. PPO 256x256 at lr 1e-4 reaches 201 in about 34 minutes;
          SAC 64x64 at lr 3e-3 reaches 206 in about 43.
        - PPO with a low lr was still gaining at 4000.
    - **F2: PPO has an lr cut-off above 16x16, and it can collapse late** (Figs 4 and 5).
      - lr 1e-3 or higher breaks any network bigger than 16x16. It's a cut-off, not a gradual effect.
      - 16x16 tolerates a high lr, but never reaches 200.
      - The timing: 256x256 at lr 1e-3 looks fine at 2000 episodes (169, 72% landed), but ends at -355 with 4%
        landed. A short test would have picked it.
      - **Likely why:**
        - A constant high lr gives 320 Adam steps on each batch of data. Adam moves each weight by about lr per step,
          and a wider layer adds up more of those moves.
        - So the policy jumps further than the clip can hold, because the clip only stops the gradient for samples
          already outside 0.8-1.2 [Engstrom 2020].
        - On-policy, a broken policy only collects broken data, so it can't recover.
      - **Evidence:**
        - The lr 3e-3 policy's raw outputs reach +/-60, against an action range of +/-1.
        - Its std grew from 1.0 to 2.2-3.2, while PPO 1's shrank to 0.12-0.33.
        - It crashes in about 1.6 s on every planet.
      - **Why SAC stays stable at the same lr:** its replay buffer of past data, slow target critics, twin critics and
        entropy bonus. A bad step gets corrected by older, good data.
    - **F3: Which setting matters most depends on the measure** (Table 7, Fig 8).
      - The method, in one line: each setting's eta squared is its share of the total squared spread between the 432
        models. Whatever is left between the 3 seeds of a setting is seed luck.
      - By mean return: training episodes 12%, algorithm 12%, lr 7%, size 1%, settings combined 45%, seed 23%.
      - By landed %: training episodes 40%.
      - Without the unstable PPO corner (lr 1e-3 or 3e-3 on 64x64 or 256x256, 17% of the models), training episodes
        explain 49%.
      - Chart 1's biggest gray bar, "3 or 4 settings combined" (22%), is where that corner lands. It's an effect only
        of PPO, a high lr and a big network together.
    - **F4: Generalization and failure modes** (Table 6).
      - Heavy wind widens SAC's lead: PPO 1 falls from 93% to 81%, SAC 1 from 100% to 98%.
      - Heavy gravity behaves like ideal, because it's inside the training range.
      - **Low gravity is the hardest condition.**
        - A likely reason: training never sees it, and the agent can't tell which planet it's on, because gravity
          isn't in the state and the network has no memory.
        - The thrust it learned now lifts it more, so it overshoots and hovers.
        - SAC 1 times out on 10% of low-gravity landings, against at most 2.7% elsewhere.
      - **They fail differently:** PPO crashes, SAC hovers. In low gravity the top 3 SAC settings had 48 timeouts
        against 24 crashes.
      - **More training can over-specialise.** SAC 64x64 at lr 3e-3 dips 20 points after 2000 episodes, and almost
        all of that dip is in low gravity.
        - Low gravity falls from 233 to 164, and its landed % from 79% to 59%.
        - The random mix drops 36, while ideal rises from 270 to 274.
        - That's one setting with 3 seeds, so call it a sign, not proof.
      - **SAC 16x16.**
        - The seed matters more than the lr: with lr 1e-3 the seeds end at 21, -9 and 222.
        - A likely reason it hovers: its critic must score both the state and the action with only 16 units, and
          hovering avoids the -100 crash.
        - A 16x16 policy can land, though: PPO 16x16 lands 63-71%.
  - **How effective is the policy?**
    - Use SAC 1's per-condition table from `best performers.md`.
    - Its SD per landing is 16-33 in ideal, both winds and heavy gravity, against 80 in low gravity.
  - Move any numbers not used above to the appendix.

  **Table 7: importance.**

  | Setting | Mean return | Landed % | Return, without the unstable corner |
  |---|---|---|---|
  | training episodes | 12% | 40% | 49% |
  | algorithm | 12% | 8% | 7% |
  | learning rate | 7% | 2% | 0.4% |
  | network size | 1% | 4% | 10% |
  | settings combined | 45% | 29% | 15% |
  | seed (luck) | 23% | 16% | 18% |

- **4.5 Limitations.**
  - Only 3 seeds, and seed luck accounts for 23% of the spread.
  - Only the lr and network size were varied. Everything else is an SB3 default, with a constant lr.
  - The budget was equal episodes, not equal compute: SAC took 7-13x the CPU time.
  - 4000 episodes is a cap, and some settings are still improving.
  - 2D physics, infinite fuel, and a shaped reward that belongs to this simulator.
  - Low gravity is never trained, and there are only 300 fixed test planets.
  - Mean return is sensitive to collapses.
  - The explanations in F2 and F4 are likely, not proven: PPO's KL divergence and clip fraction weren't logged.
- **4.6 Future scope.**
  - Train on to 8000 episodes; the replay buffers are kept.
  - Run more seeds.
  - Log PPO's KL divergence and clip fraction, to test the explanation of the collapse.
  - PPO with a learning-rate schedule or a KL target.
  - Tune the other hyperparameters, for example with Optuna.
  - Put low gravity in training, or use a curriculum.
  - Give the agent the planet's settings, or memory to infer them.
  - Try TD3, DDPG, or DQN on the discrete version.
  - 6-DOF landing and sim-to-real [Gaudet 2020].
- **4.7 Conclusion.** One paragraph. Sum up the problem, the approach and the implementation in two sentences, as
  the template asks. Then give the headline numbers:
  - SAC 275 and 96% against PPO 220 and 85%, per episode and per sample;
  - training length matters most once the unstable corner is excluded;
  - PPO needs an lr of 3e-4 or lower above 16x16;
  - SAC is insensitive to the lr at 64x64 and 256x256;
  - low gravity is the hardest condition;
  - PPO is 7-13x faster per run, so per CPU-minute the two are about even.

### 5. References (about 18 entries; check each before citing)
- Sutton and Barto (2018), *Reinforcement Learning: An Introduction*, 2nd edition.
- Schulman et al. (2017), PPO, arXiv:1707.06347.
- Schulman et al. (2016), GAE.
- Haarnoja et al. (2018), SAC, ICML (arXiv:1801.01290).
- Haarnoja et al. (2018b), SAC algorithms and applications (arXiv:1812.05905).
- Fujimoto et al. (2018), TD3, for the clipped double Q.
- Raffin et al. (2021), Stable-Baselines3, JMLR.
- Towers et al. (2024), Gymnasium (arXiv:2407.17032).
- Brockman et al. (2016), OpenAI Gym.
- Henderson et al. (2018), "Deep RL that Matters", AAAI.
- Andrychowicz et al. (2021), "What Matters in On-Policy RL?", ICLR.
- Engstrom et al. (2020), "Implementation Matters in Deep RL: A Case Study on PPO and TRPO", ICLR. This supports
  F2.
- Tobin et al. (2017), domain randomization, IROS.
- Peng et al. (2018), "Sim-to-Real Transfer of Robotic Control with Dynamics Randomization", ICRA.
- Gaudet, Linares and Furfaro (2020), deep RL for 6-DOF planetary landing.
- Paszke et al. (2019), PyTorch.
- ISRO's Chandrayaan-3 page.
- Optional:
  - Kingma and Ba (2015), Adam, ICLR, for the Adam step in F2;
  - Acikmese and Ploen (2007), convex programming for Mars powered-descent guidance, JGCD, for "why not a classical
    controller".

### Appendix
- The GitHub link, https://github.com/dark756/RL-IA. First commit `results/`, `plots/`, `observations/`, `plots.py`
  and `diagram.py`.
- **How to reproduce,** as a short box:
  - the commands: `python driver.py train --cpu-workers 11 --gpu-workers 0`, `python driver.py test --log`,
    `python driver.py plots` and `python diagram.py`;
  - the library versions from 3.2;
  - seeds 0-2 and `TEST_SEED = 2026`.
- **Sample outputs:** the console table from D6, and the `python driver.py plots` printout.
- Code snippets: `GaussPlanetWrapper.reset`, `StageSaver._on_step`, the outcome rules in `testing.py`, and the model
  construction in `train()`.
- Extra charts: `1_importance.png`, `5_outcomes.png`, `4_return_vs_size.png` and `conditions/*.png`.
- The six winners' per-setting breakdowns (`best performers.md`) and `results/peaks.csv`.
- An AI-use statement, if the course asks for one.

## Leave out
- The raw CSVs and file paths in the body; model file names belong in the appendix only.
- History beyond the three engineering challenges in 3.5:
  - `mac.md`;
  - the progress bar;
  - the CPU and GPU queue rules;
  - the carry-on mechanics in detail.
- The wording of the observations files, such as "your point 1 is misleading". Use their findings instead.
- Solved %, and two minor limitations, which are now notes under "Viva questions".

## Gaps to fill before writing
1. **Fig 3 screenshots.** Run D4 and D5 and capture the first landing of each with Win+Shift+S. Alternatively, I can
   add a frame-saving option, which is a code change and needs your go.
2. **Table 7's last two columns.** The landed % column and the "without the unstable corner" column were worked
   out by hand, and `plots.py` doesn't produce them. Either:
   - cite them and state the method; or
   - I can add them to the `plots` printout, which is a code change and needs your go.
3. **F2's evidence.**
   - D7 reproduces the std values live.
   - The crash time comes from `summary.csv`: 82 steps on average, which is 1.6 s.
   - The +/-60 outputs came from a one-off check on 400 sample states. Either say that, or drop the number.
4. **Before linking the repo,** commit the new files, and check the course's rules on AI use.

Optional code changes, each needing your go:
- `testing.py --landings N`, so the D3-D5 demos stop by themselves. It would be refused with `--log`.
- Report-size fonts for the wide charts, or a version of chart 5 with only 3 conditions.

## Page budget
About 13-14 pages:

| Part | Pages |
|---|---|
| Header | 0.3 |
| Section 2 | 2.5 |
| Section 3 | about 4.7 |
| Section 4 | about 5.5-6 |

Charts at full text width take about half a page each.

## Presenting it: demo script
Run these in PowerShell, in the project folder. The expected results come from the logged test results. A watched
landing can differ slightly, because rendering flies one lander at a time.

**Setup**, once per terminal:
```
cd C:\Users\gurka\Desktop\RL-IA
.\.venv\Scripts\activate
$env:OMP_NUM_THREADS = 1
```
The last line makes the printed tables match the report exactly. Without it, SAC 1 seed 0 prints 276.5 instead of
276.6.

**D1. Section 2.4: the state and action spaces** (about 3 s).
```
python -c "import gymnasium as gym; e = gym.make('LunarLanderContinuous-v3'); print(e.observation_space); print(e.action_space)"
```
Expected: a Box of 8 numbers, then `Box(-1.0, 1.0, (2,), float32)`.

**D2. Section 3.3: a walk through the code.** In VS Code, press Ctrl+P and type the location. The line numbers are
as of 2026-10-13.

| Location | What it shows |
|---|---|
| `trainer.py:43` | `GaussPlanetWrapper.reset`, which draws a new planet before each landing |
| `trainer.py:146` | the model: SB3 defaults, with only the lr and network size set |
| `trainer.py:78` | `StageSaver._on_step`, which saves the weights at each stage |
| `testing.py:27` | the 6 test conditions |
| `testing.py:97` | the outcome rule: +100 at rest is landed, -100 is crashed, 1000 steps is a timeout |
| `plots.py:103` | `variance_shares`, the ANOVA |
| `driver.py:22` | the 72-run grid |

**D3. Section 4.4, "how did performance change": PPO 1's best seed at three stages, on ideal planets.**
```
python testing.py resources/ppo/ppo_lr1e-4_net256x256_seed1_ep125.zip --render --conditions ideal
python testing.py resources/ppo/ppo_lr1e-4_net256x256_seed1_ep500.zip --render --conditions ideal
python testing.py resources/ppo/ppo_lr1e-4_net256x256_seed1_ep4000.zip --render --conditions ideal
```

| Episodes | What you'll see |
|---|---|
| 125 | every landing crashes within 1-2 s |
| 500 | slow, hovering landings of about 10 s; landings 3 and 4 crash |
| 4000 | clean landings of about 5-6 s |

That's chart 8's crash, then hover, then land.

**D4. Section 3.6: the best agent on hard planets.**
```
python testing.py resources/sac/sac_lr3e-3_net256x256_seed0_ep4000.zip --render --conditions heavy_wind
python testing.py resources/sac/sac_lr3e-3_net256x256_seed0_ep4000.zip --render --conditions low_gravity
```
- Heavy wind: landing 1 lands in about 3 s. Landing 2 crashes, which is one of this seed's 4% of crashes.
- Low gravity, which it never trained on: landings 1-3 land, but take 6-9 s each, and landing 4 takes 18 s. Point
  out the hovering.

**D5. Section 4.4: the drop-off, in the same run before and after.**
```
python testing.py resources/ppo/ppo_lr1e-3_net256x256_seed1_ep2000.zip --render --conditions ideal
python testing.py resources/ppo/ppo_lr1e-3_net256x256_seed1_ep4000.zip --render --conditions ideal
python testing.py resources/ppo/ppo_lr3e-3_net256x256_seed0_ep4000.zip --render --conditions ideal
```
- At 2000 episodes, the first 5 all land, in about 5-6 s each.
- At 4000 episodes, every landing crashes within 1-2 s.
- The third command is optional. That run never learned (lr 3e-3), and it crashes along the same paths, so two
  different runs collapsed into the same failure.

**D6. Section 4.1: the numbers, live** (no window, about 12 s).
```
python testing.py resources/sac/sac_lr3e-3_net256x256_seed0_ep4000.zip
```

| Condition | Mean return | Landed |
|---|---|---|
| all | 276.6 | 97% |
| heavy wind | 272.3 | 94% |
| low gravity | 259.1 | 90% |

**D7. Table 3: exploration, live** (about 10 s each).
```
python -c "from stable_baselines3 import PPO; [print(p, PPO.load(f'resources/ppo/{p}.zip', device='cpu').policy.log_std.exp().detach().numpy().round(2)) for p in ['ppo_lr1e-4_net256x256_seed1_ep125', 'ppo_lr1e-4_net256x256_seed1_ep4000', 'ppo_lr3e-3_net256x256_seed0_ep4000']]"
python -c "from stable_baselines3 import SAC; [print(p, round(SAC.load(f'resources/sac/{p}.zip', device='cpu').log_ent_coef.exp().item(), 4)) for p in ['sac_lr3e-3_net256x256_seed0_ep125', 'sac_lr3e-3_net256x256_seed0_ep4000']]"
```

| Model | Prints |
|---|---|
| PPO 1, 125 episodes | [0.99 0.99] |
| PPO 1, 4000 episodes | [0.33 0.12] |
| the collapsed lr 3e-3 run | [3.24 2.17] |
| SAC 1, 125 episodes | 0.1685 |
| SAC 1, 4000 episodes | 0.0422 |

SAC's temperature starts at 1.0.

**D8. Section 4.3: the charts are reproducible** (seconds).
```
python driver.py plots
python diagram.py
```
These print the importance shares, the best settings and the peaks, and redraw `plots/` with identical files.

**D9. Only if they ask to see training.**
- Instant:
  ```
  Get-Content logs/sac_lr3e-3_net256x256_seed0.csv -TotalCount 6
  ```
  It shows that each training landing gets its own random planet.
- The full pipeline, which takes about 5 minutes. Start it in a second terminal at the beginning:
  ```
  python driver.py train --quick
  python driver.py test --quick --log
  python driver.py plots --quick
  ```
  Then open `quick/plots/`. Afterwards, run `Remove-Item -Recurse -Force quick`.

**Notes for the day.**
- **Timing.** Rendered landings play in real time, at 50 steps a second: a 250-step landing takes 5 s and a timeout
  20 s.
- **Stopping.** Each condition has 50 planets, so press Ctrl+C in the terminal after 2-3 landings.
  - Closing the window doesn't stop it.
  - Ctrl+C prints a KeyboardInterrupt traceback, which is harmless.
- **The window.** Each landing opens its own small window. Press Win+P and choose Duplicate, so the window shows on
  the projector.
- Never add `--log` during the demo. It would append to `results/test_results.csv`.
- **Beforehand:**
  - Rehearse every command.
  - Record D3-D5 as a backup video, with Snipping Tool's Record (Win+Shift+R) set to the full screen.
  - Keep the laptop plugged in, and pause Windows Update.
  - The models aren't in git (`resources/` is ignored), so present from this laptop.

## Viva questions to prepare
1. **Why not DQN?** The actions are continuous, and DQN needs discrete actions.
2. **Did you implement PPO and SAC?** No, they're SB3's; using its defaults keeps the comparison fair. The project's
   own code is the harness: the planets, the staged saves, the testing and the analysis.
3. **Where's epsilon?**
   - PPO samples its learned Gaussian, whose std goes from 1.0 to 0.12-0.33.
   - SAC adds an entropy bonus, whose alpha is auto-tuned from 1.0 down to 0.04.
   - D7 shows both.
4. **Is the state Markov?** Not fully: gravity and wind aren't in it, and the network has no memory.
5. **Why did PPO collapse?** See F2. The evidence is the +/-60 outputs and a std that grew to 3.2.
6. **Why is SAC stable at the same lr?** The replay buffer, the slow target critics, the twin critics and the entropy
   bonus.
7. **Is comparing at equal episodes fair?** SAC took 7-13x the compute, and per CPU-minute the two are about even. So
   the claim is per episode and per sample.
8. **What is eta squared?** Each setting's share of the total squared spread between the 432 models. Whatever is
   left between the 3 seeds of a setting is seed luck.
9. **Why do mean return and landed % disagree?** Collapses near -1000 dominate the squared spread of the return,
   while landed % stays between 0 and 100.
10. **Why test deterministically, on the same planets?** To measure the policy, not its exploration noise, with every
    model on the same 300 planets.
11. **How do you know it isn't luck?** The winners' seed ranges don't overlap. Overall, seed luck is 23% of the
    spread, mostly in the PPO corner.
12. **Why not train on the GPU?** For small networks the kernel launches dominate, so 11 CPU threads were faster.
13. **Why gravity -10 on a "lunar" lander?** That's Gymnasium's default. The Moon-like planets are exactly the unseen
    low-gravity test.
14. **Why not a classical controller?**
    - Real landers use model-based guidance, which needs an accurate model of the craft and the planet.
    - RL learns from interaction, and one policy covers planets it doesn't know.
    - But RL isn't certifiable yet.
15. **Does a resumed run match one that never stopped?** It meets the same planets, but its training noise differs.
16. **Why did a watched landing differ from the table?** Rendering flies one lander at a time, so the rounding
    differs slightly. In one check, 1 landing in 48 came to rest 4 steps later.
