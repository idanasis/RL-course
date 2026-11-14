# Exercise 4 — POMCP Box-Pushing Agent: Report

This exercise implements an online POMDP agent for the partially observable
box-pushing environment. The agent does **not** know its own `(x, y)` position;
it only receives a translation-based `3x3` egocentric observation. Four
components work together:

| File | Role |
|------|------|
| [observation_function.py](observation_function.py) | Custom translation-based (heading-agnostic) `3x3` egocentric observation. |
| [practical_filter.py](practical_filter.py) | `ParticleFilter` — belief over the hidden agent position (rejection sampling + map-knowledge fallback). |
| [pomcp.py](pomcp.py) | `POMCP` — online Monte-Carlo tree search over the belief. |
| [main.py](main.py) | Execution + evaluation loop that ties them together. |

---

## 1. How to run

**Prerequisites.** Use the project's virtual environment (it has `minigrid`,
`pettingzoo`, and `numpy` installed) and run from the **repository root**.

```powershell
# from the repo root, with the venv active:
.\venv\Scripts\Activate.ps1        # activates the venv (prompt shows "(venv)")
python3 -m exercises.ex4.main       # full evaluation (default: 30 episodes @ 1.0s then 20.0s)
```

**Quick sanity run** (a few seconds instead of the full run):

```powershell
python -m exercises.ex4.main --episodes 3 --budgets 0.3 --max-steps 40 --verbose
```

**Command-line options:**

| Flag | Default | Meaning |
|------|---------|---------|
| `--episodes` | `30` | Independent episodes per (scenario, budget). |
| `--budgets` | `1.0 20.0` | Per-step planning time budgets (seconds); one evaluation per value. |
| `--scenarios` | `single multi` | Which scenarios to run: `single` (one robot) and/or `multi` (two robots). |
| `--max-steps` | `50` | Environment step limit before an episode is truncated (failure). |
| `--verbose` | off | Print a per-step / per-episode trace (belief, action, termination). |

**Expected output** (one line per scenario × budget):

```
=== POMCP evaluation | 30 episodes per (scenario, budget) ===
single | time_budget =   1.0s | success 30/30 | steps to goal: mean=..  std=..
single | time_budget =  20.0s | success 30/30 | steps to goal: mean=..  std=..
multi  | time_budget =   1.0s | success 30/30 | steps to goal: mean=..  std=..
multi  | time_budget =  20.0s | success 30/30 | steps to goal: mean=..  std=..
```

> **Runtime note.** Wall-clock per configuration ≈ `episodes × steps × budget`,
> and **×2 agents** for `multi` (each robot gets the full budget per step). The
> `1.0s` configs are minutes; the `20.0s` configs can take **tens of minutes to
> hours**. Use `--episodes`/`--budgets`/`--scenarios` to shrink a trial run.

---

## 2. Hyperparameters

All values below are the ones actually used by [main.py](main.py) (which
constructs the planner and filter) and the underlying components.

### POMCP planner ([pomcp.py](pomcp.py) / [main.py](main.py))

| Hyperparameter | Value | Notes |
|----------------|-------|-------|
| Discount factor `γ` (gamma) | **0.95** | Standard discount; makes reaching the goal *sooner* worth more. |
| UCB-1 exploration constant `c` | **1.0** | Balances exploitation (`Q`) vs. exploration (`c·√(ln N / N(a))`) during tree search. |
| Max simulation / rollout depth | **50** | Hard cap on recursion depth for both `simulate` and `rollout`. |
| Depth cutoff `ε` (epsilon) | **0.01** | Extra stop condition `γ^depth < ε` ⇒ soft horizon ≈ 90 steps; since 90 > 50, the depth-50 cap is the **binding** limit. |
| Rollout policy | uniform random over actions | "Fast" default POMCP rollout. |
| Action space | `{0: left, 1: right, 2: forward}` | Real rotation-based env actions. |
| Generative model | **stochastic, shared with the filter** | `sample_step` samples the env's dynamics (move 0.8 with ±90° deviation, push 0.8); the **same** method drives both tree simulations and the filter's belief update. |
| Planning time budget / step | **1.0 s** and **20.0 s** | The two compared configurations (enforced by a wall-clock loop, not just noted). |
| Tree reuse | none (cleared each step) | The search tree is belief-specific, so it is rebuilt every planning step. |
| Known heading | fed in each step | Heading is observable (only position is hidden), so start states use the true heading instead of a uniform-random one. |

### Particle filter ([practical_filter.py](practical_filter.py) / [main.py](main.py))

| Hyperparameter | Value | Notes |
|----------------|-------|-------|
| Number of particles `N` | **500** | The assignment's recommended starting value; each particle is one `(x, y)` position hypothesis. |
| Observation window | **3 × 3** (`window_size = 3`) | Egocentric, translation-based. |
| Update rule | **unweighted rejection sampling** | Sample a particle, simulate the action with the shared stochastic `sample_step`, keep it iff its simulated observation matches the real one; repeat until `N` are collected (Silver & Veness, 2010). |
| Rejection attempt cap | `N × 30` | Max sampling attempts before falling back (prevents an unbounded loop when the rejection rate is high). |
| Fallback | map-knowledge recovery | On (partial or total) depletion, repopulate from the known cells whose 3×3 slice matches the observation. |
| Transition model | **shared** with POMCP | Not a separate model — `update` calls the same `sample_step`; the filter tracks position only, using the known heading + boxes. |

### Environment (`StochasticMultiAgentBoxPushEnv`)

| Hyperparameter | Value | Notes |
|----------------|-------|-------|
| Move success probability | **0.8** | With prob. 0.2 a forward move deviates ±90°. |
| Push success probability | **0.8** | A push can stochastically fail. |
| `max_steps` (truncation) | **50** | Episode fails if the goal isn't reached in time. |
| Evaluation maps | 7×7 room, `SINGLE_MAP` / `MULTI_MAP` | Same room, box one cell above the goal; single = one robot, multi = two robots (controlled comparison). |

### Evaluation protocol ([main.py](main.py))

| Hyperparameter | Value | Notes |
|----------------|-------|-------|
| Episodes per configuration | **30** | Independent runs. |
| Per-episode seed | `base_seed(0) + episode_index` | Seeds NumPy (env stochasticity) + filter + planner for reproducibility; the same seeds are reused across budgets so configs are directly comparable. |
| Reported metrics | success rate, mean & std of steps-to-goal | Mean/std computed over the episodes that reached the goal. |

---

## 3. Results

Full run: 30 episodes per configuration, `python -m exercises.ex4.main`.

| Scenario | Time budget | Episodes | Success rate | Mean steps | Std steps |
|----------|-------------|----------|--------------|-----------|-----------|
| Single agent | 1.0 s | 30 | 30/30 | 8.90 | 3.37 |
| Single agent | 20.0 s | 30 | 30/30 | 8.70 | 2.65 |
| Two agents | 1.0 s | 30 | 30/30 | 7.90 | 1.19 |
| Two agents | 20.0 s | 30 | 30/30 | 7.97 | 1.78 |

> Produced by `python -m exercises.ex4.main` (both scenarios run by default).
> Standard error of the mean ≈ std/√30 ≈ 0.2–0.6 steps, which is larger than the
> 1 s→20 s differences (see §4.1).

### Supplementary: an even smaller budget (0.25 s)

Quick 6-episode checks that add a 0.25 s budget, showing that the saturation
extends below 1 s as well (0.25 s ≈ 1 s ≈ 20 s):

| Scenario | Time budget | Episodes | Success | Mean steps | Std steps |
|----------|-------------|----------|---------|-----------|-----------|
| single | 0.25 s | 6 | 6/6 | 8.50 | 2.14 |
| multi | 0.25 s | 6 | 6/6 | 8.33 | 1.60 |

(Optimal is ~7 steps: navigate to the push cell, then push.)

---

## 4. Discussion

### 4.1 Effect of the planning time budget (1 s vs 20 s)

Over 30 episodes, a **20× larger budget did not meaningfully improve either
scenario**:

- single: 8.90 ± 3.37 (1 s) vs 8.70 ± 2.65 (20 s)
- multi: 7.90 ± 1.19 (1 s) vs 7.97 ± 1.78 (20 s)

The gaps (≤ 0.2 steps) are smaller than the standard error of the mean
(≈ std/√30 ≈ 0.5–0.6 for single, 0.2–0.3 for multi), so 1 s and 20 s are
**statistically indistinguishable** — the multi 20 s mean being a hair higher is
noise, not a regression. Every configuration solved the task in all 30 episodes.

Budget saturates this early because:

1. **Short horizon + online re-planning.** The optimum is ~7 steps and POMCP
   re-plans every step, so each decision only needs the best *next* action — a
   shallow problem that ~1 s of simulations already solves reliably. Extra
   simulations re-confirm the same action rather than changing it.
2. **The relevant state is nearly fully observed.** The heading is known and the
   3×3 observation localizes the position almost immediately, so POMCP is
   effectively planning in a small, near-fully-observed MDP.
3. **Irreducible stochasticity sets the floor.** The gap between the ~7-step
   optimum and the observed ~8–9 steps is caused by the env's 0.8 move / 0.8 push
   success (deviations and failed pushes cost extra steps). No planning budget
   removes this; only re-planning after a failure helps, which both budgets do.

So on this task 1 s is already past the point of diminishing returns; a visible
1 s→20 s gap would require a longer-horizon / more-ambiguous task where 1 s
genuinely under-plans.

### 4.2 Multi-agent (two robots) vs. single agent

**Setup.** The multi-agent run is **decentralized**: each robot has its own
particle filter and its own POMCP, localizes from its own observation, and plans
independently; the two chosen actions are submitted to the env jointly each step
(`run_episode_multi`). `SINGLE_MAP` and `MULTI_MAP` share the same room, box, and
goal — only the number of robots differs — so the comparison is controlled.

**Results.** Two robots solved the same one-box task **faster and much more
consistently** than one (30/30 success in every configuration):

- single: 8.90 ± 3.37 (1 s), 8.70 ± 2.65 (20 s)
- multi: 7.90 ± 1.19 (1 s), 7.97 ± 1.78 (20 s)

That is ~1 step faster on average and roughly **half the standard deviation**.
The gain is **redundancy**, not cooperation: with two robots there are two
chances to be well-positioned for the push, so whichever is closer finishes the
task and the outcome depends less on any single robot's stochastic luck (hence
both the lower mean and the much lower variance). The cost is ~2× planning
wall-clock per step, since each robot plans independently.

**Interference and the coordination limit.** The robots can block each other
(they cannot overlap) and each plans as if alone — its generative model does not
model the other robot — so the benefit is parallel redundancy rather than true
coordination. The heavy box (`C`), which needs two robots pushing the same cell,
in the same direction, on the same step, is beyond independent decentralized
planners; the evaluation therefore uses a single small box (solvable by either
robot), and the heavy-box case is noted as a limitation rather than measured.
