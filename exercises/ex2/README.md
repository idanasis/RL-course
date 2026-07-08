# Assignment 2 — Probabilistic Box Pushing

## Environment Setup

The environment is identical to Assignment 1. If you haven't set it up yet, see the full instructions in the Assignment 1 README or in `README.md` at the root of the repository:
[https://github.com/ronberg-bgu/RL-course](https://github.com/ronberg-bgu/RL-course)

### Create a branch for this assignment

```bash
git checkout -b student-{firstname}-{lastname}-ex2
```

For example: `student-yossi-cohen-ex2`, `student-sarah-levi-ex2`

### Submit

When finished, push your branch and open a Pull Request at:
[https://github.com/ronberg-bgu/RL-course](https://github.com/ronberg-bgu/RL-course)

---

## Assignment Description

You will solve a **stochastic** Box Pushing problem. The action definitions of the new simulator are identical to those in Assignment 1, but outcomes are no longer deterministic.

### Transition model

| Action | Success | Failure |
|--------|---------|---------|
| **move** (to empty cell) | 0.8 → intended direction | 0.1 → 90° left · 0.1 → 90° right |
| **push-small / push-heavy** (precondition met) | 0.8 → push succeeds | 0.2 → no state change |

If the precondition of an action is not met, the action has no effect (same as Assignment 1).

> Example: moving left → 0.8 go left, 0.1 go up, 0.1 go down.
> If the deviated cell is blocked by a wall or box, the agent simply stays put.

---

## Stochastic Simulator

A ready-to-use stochastic environment is provided in `environment/stochastic_env.py`.

```python
from environment.stochastic_env import StochasticMultiAgentBoxPushEnv

env = StochasticMultiAgentBoxPushEnv(ascii_map=ascii_map, max_steps=200)
obs, _ = env.reset()
```

The API is identical to `MultiAgentBoxPushEnv` from Assignment 1 (same PettingZoo `ParallelEnv` interface). The only difference is that `step()` applies the stochastic transition model above.

---

## Part 1 — Online Planning on the Stochastic Environment

Solve the problem using an **online** method: at each step, run a classical (deterministic) planner from the current state, execute only the **first action** of the resulting plan, observe the new (possibly unexpected) state, and replan.

You may choose which of the possible outcomes the planner assumes — the key idea is that you replan after every step.

### PDDL utilities (already provided)

```python
from environment.pddl_extractor import generate_pddl_for_env
from planner.pddl_solver import solve_pddl
from visualize_plan import extract_target_pos, get_required_actions
```

### Online planning loop structure

```python
obs, _ = env.reset()

while True:
    # 1. Export current state to PDDL
    domain_path, problem_path = generate_pddl_for_env(env)

    # 2. Call the planner
    plan = solve_pddl(domain_path, problem_path)
    if not plan or len(plan.actions) == 0:
        break  # goal reached or unsolvable

    # 3. Execute the FIRST PDDL action from the plan
    #    One PDDL action may require several env steps (rotations + forward).
    pddl_action = plan.actions[0]
    agent_targets = extract_target_pos(pddl_action)

    agents_in_action = list(agent_targets.keys())
    action_queues = {
        a: get_required_actions(env, a, agent_targets[a])
        for a in agents_in_action
    }

    # Pad shorter queues so all agents do their final forward together
    max_len = max(len(q) for q in action_queues.values())
    for a in agents_in_action:
        action_queues[a] = [None] * (max_len - len(action_queues[a])) + action_queues[a]

    while any(len(q) > 0 for q in action_queues.values()):
        step_actions = {}
        for a in agents_in_action:
            if action_queues[a]:
                act = action_queues[a].pop(0)
                if act is not None:
                    step_actions[a] = act

        obs, rewards, terms, truncs, _ = env.step(step_actions)
        if any(terms.values()) or any(truncs.values()):
            break

    if any(terms.values()) or any(truncs.values()):
        break
```

---

## Part 2 — Modified Policy Iteration (MPI)

Solve the problem offline using **Modified Policy Iteration**.

In MPI, instead of running policy evaluation to convergence at each iteration, you perform only **k steps** of iterative policy evaluation (k is a hyperparameter, typically 10–50) before doing a policy improvement step.

```
Initialize V(s) = 0 for all s, π(s) = arbitrary

Repeat until convergence:
    # Partial policy evaluation (k sweeps)
    for _ in range(k):
        for each state s:
            V(s) ← Σ_{s'} P(s'|s, π(s)) · [R(s,π(s),s') + γ · V(s')]

    # Policy improvement
    for each state s:
        π(s) ← argmax_a Σ_{s'} P(s'|s,a) · [R(s,a,s') + γ · V(s')]

    if policy did not change → converged
```

### State representation

A state must capture everything needed to plan: agent positions, agent directions, and box positions.

```python
# Example minimal state tuple
state = (agent0_pos, agent0_dir, agent1_pos, agent1_dir,
         box0_pos, box1_pos, heavy_pos)
```

> **Hint on tractability:** The full state space can be very large. Consider whether agent *direction* matters for the value of a state (the planner can always rotate for free — does that change the optimal value?), or whether you can reduce the state space in another principled way.

### Transition model for MPI

You can compute the MDP transition model analytically from the stochastic rules:

- Move to empty cell: 3 outcomes — intended dir (p=0.8), left (p=0.1), right (p=0.1)
- Push (precondition met): 2 outcomes — success (p=0.8), no-op (p=0.2)
- Precondition not met: 1 outcome — no-op (p=1.0)

---

## Evaluation

Run **each** of the two policies 100 times on the stochastic environment and report:
- Mean number of steps to reach the goal
- Standard deviation of steps to goal

A skeleton evaluation function is provided in `solution_ex2.py`.

---

## Submission Requirements

### Pull Request on GitHub

Your branch must include:

| File | Description |
|------|-------------|
| `exercises/ex2/solution_ex2.py` | Complete implementation |
| `exercises/ex2/results.txt` | Printed evaluation results (mean ± std for both algorithms) |

Capture the output with:

```bash
python3 exercises/ex2/solution_ex2.py 2>&1 | tee exercises/ex2/results.txt
```

### Live Demo

You will present your work live. Be prepared to:
1. Run the full pipeline end-to-end.
2. Show the online planner navigating the stochastic environment.
3. Show the MPI policy being applied and evaluated.
4. Explain your state representation and any tractability choices you made.
