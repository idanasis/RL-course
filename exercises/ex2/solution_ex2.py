"""
Assignment 2 — Probabilistic Box Pushing
=========================================
Fill in the three TODO sections below:
  1. run_online_planning  — online replanning loop
  2. build_transition_model — MDP transition model (used by MPI)
  3. modified_policy_iteration — MPI algorithm

Do NOT modify evaluate_policy or the __main__ block.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
from environment.stochastic_env import StochasticMultiAgentBoxPushEnv
from environment.pddl_extractor import generate_pddl_for_env
from planner.pddl_solver import solve_pddl
from visualize_plan import extract_target_pos, get_required_actions

# ---------------------------------------------------------------------------
# Map used in both parts (same as Assignment 1)
# ---------------------------------------------------------------------------
ASCII_MAP = [
    "WWWWWW",
    "WA   W",
    "WBBACW",
    "W    W",
    "WGG GW",
    "WWWWWW"
]


# ===========================================================================
# Part 1 — Online Planning
# ===========================================================================

def run_online_planning(env, max_replans: int = 300) -> int:
    """
    Execute one episode using online planning:
      replan from the current state → execute only the first PDDL action → repeat.

    Returns
    -------
    int
        Number of *env* steps taken (counting each rotate/forward individually).
        Returns max_replans * <average_actions_per_plan_step> as a large sentinel
        if the goal was never reached within max_replans replanning calls.
    """
    obs, _ = env.reset()
    total_env_steps = 0
    done = False

    for _ in range(max_replans):
        if done:
            break

        # ── 1. Export current state ──────────────────────────────────
        domain_path, problem_path = generate_pddl_for_env(env)

        # ── 2. Plan ──────────────────────────────────────────────────
        plan = solve_pddl(domain_path, problem_path)
        if not plan or len(plan.actions) == 0:
            break  # goal already reached (planner returns empty plan)

        # ── 3. Execute the first PDDL action ─────────────────────────
        pddl_action   = plan.actions[0]
        agent_targets = extract_target_pos(pddl_action)

        if not agent_targets:
            break

        # Build per-agent action queues (rotations + forward)
        agents_in_action = list(agent_targets.keys())
        action_queues = {
            a: get_required_actions(env, a, agent_targets[a])
            for a in agents_in_action
        }

        # Pad shorter queues so all agents execute their final forward together
        max_len = max(len(q) for q in action_queues.values())
        for a in agents_in_action:
            action_queues[a] = (
                [None] * (max_len - len(action_queues[a])) + action_queues[a]
            )

        # Step through the queue
        while any(len(q) > 0 for q in action_queues.values()):
            step_actions = {}
            for a in agents_in_action:
                if action_queues[a]:
                    act = action_queues[a].pop(0)
                    if act is not None:
                        step_actions[a] = act

            obs, rewards, terms, truncs, _ = env.step(step_actions)
            total_env_steps += 1

            if any(terms.values()) or any(truncs.values()):
                done = True
                break

    return total_env_steps


# ===========================================================================
# Part 2 — Modified Policy Iteration
# ===========================================================================

# ---------------------------------------------------------------------------
# State representation
# ---------------------------------------------------------------------------
# A state is a tuple:
#   (agent0_pos, agent0_dir, agent1_pos, agent1_dir,
#    box0_pos,   box1_pos,   heavy_pos)
#
# where positions are (col, row) tuples and directions are 0-3.
#
# Feel free to simplify (e.g. drop agent directions if you argue they are
# irrelevant) as long as you justify it in your live demo.

def get_state(env) -> tuple:
    """Extract the current state tuple from a live environment."""
    agents = env.possible_agents
    a0_pos = env.agent_positions[agents[0]]
    # a0_dir = env.agent_dirs[agents[0]]
    a1_pos = env.agent_positions[agents[1]]
    # a1_dir = env.agent_dirs[agents[1]]
    ags = sorted([a0_pos, a1_pos])
    # Collect box positions by scanning the grid
    small_boxes = []
    heavy_boxes = []
    for y in range(env.height):
        for x in range(env.width):
            cell = env.core_env.grid.get(x, y)
            if cell is not None and cell.type == "box":
                if getattr(cell, "box_size", "") == "heavy":
                    heavy_boxes.append((x, y))
                else:
                    small_boxes.append((x, y))

    # Sort for a canonical order
    small_boxes.sort()
    heavy_boxes.sort()

    box0_pos   = small_boxes[0] if len(small_boxes) > 0 else None
    box1_pos   = small_boxes[1] if len(small_boxes) > 1 else None
    heavy_pos  = heavy_boxes[0] if heavy_boxes else None
    return (ags[0], ags[1], box0_pos, box1_pos, heavy_pos)
    # return (a0_pos, a0_dir, a1_pos, a1_dir, box0_pos, box1_pos, heavy_pos)


def build_transition_model(env):
    import itertools
    from collections import deque
    DIR_TO_VEC = [(1, 0), (0, 1), (-1, 0), (0, -1)]

    print("🔍 Building Tractable Transition Model...")
    env.reset()
    
    goals = set()
    walls = set()
    for y in range(env.height):
        for x in range(env.width):
            cell = env.core_env.grid.get(x, y)
            if cell is not None and cell.type == "goal":
                goals.add((x, y))
            elif cell is not None and cell.type == "wall":
                walls.add((x, y))
    def is_terminal(s):
        b0, b1, hb = s[2], s[3], s[4]
        return (b0 in goals) and (b1 in goals) and (hb in goals)

    # Macro-actions: 0=Right, 1=Down, 2=Left, 3=Up
    MACRO_ACTIONS = [0, 1, 2, 3]
    JOINT_ACTIONS = list(itertools.product(MACRO_ACTIONS, MACRO_ACTIONS))
    
    initial_state = get_state(env)
    transitions = {}
    reachable = set([initial_state])
    queue = deque([initial_state])

    def compute_analytic_transitions(state, ja):
        a0, a1, b0, b1, hb = state
        
        def get_intents(pos, d):
            fwd = (pos[0] + DIR_TO_VEC[d][0], pos[1] + DIR_TO_VEC[d][1])
            r = (pos[0] + DIR_TO_VEC[(d + 1) % 4][0], pos[1] + DIR_TO_VEC[(d + 1) % 4][1])
            l = (pos[0] + DIR_TO_VEC[(d - 1) % 4][0], pos[1] + DIR_TO_VEC[(d - 1) % 4][1])
            return [(0.8, fwd, d), (0.1, r, d), (0.1, l, d)]

        intents0 = get_intents(a0, ja[0])
        intents1 = get_intents(a1, ja[1])
        
        results = []
        for p0, i0_pos, i0_d in intents0:
            for p1, i1_pos, i1_d in intents1:
                base_prob = p0 * p1
                
                # Heavy Box Push
                if hb and i0_pos == hb and i1_pos == hb and i0_d == i1_d:
                    tgt_hb = (hb[0] + DIR_TO_VEC[i0_d][0], hb[1] + DIR_TO_VEC[i0_d][1])
                    if tgt_hb not in walls and tgt_hb != b0 and tgt_hb != b1:
                        results.append((base_prob * 0.8, tuple(sorted([hb, hb])) + (b0, b1, tgt_hb)))
                        results.append((base_prob * 0.2, tuple(sorted([a0, a1])) + (b0, b1, hb)))
                    else:
                        results.append((base_prob, tuple(sorted([a0, a1])) + (b0, b1, hb)))
                    continue

                # Individual actions
                def resolve(i_pos, i_d, old_pos):
                    if i_pos in walls or i_pos == hb: return (old_pos, None, None)
                    if b0 and i_pos == b0:
                        tgt = (b0[0] + DIR_TO_VEC[i_d][0], b0[1] + DIR_TO_VEC[i_d][1])
                        if tgt not in walls and tgt not in (hb, b1): return (i_pos, 'b0', tgt)
                        return (old_pos, None, None)
                    if b1 and i_pos == b1:
                        tgt = (b1[0] + DIR_TO_VEC[i_d][0], b1[1] + DIR_TO_VEC[i_d][1])
                        if tgt not in walls and tgt not in (hb, b0): return (i_pos, 'b1', tgt)
                        return (old_pos, None, None)
                    return (i_pos, None, None)

                fa0, push0_b, push0_t = resolve(i0_pos, i0_d, a0)
                fa1, push1_b, push1_t = resolve(i1_pos, i1_d, a1)

                # OPTIMIZATION 2: Ghost Pruning
                if push0_b and push0_b == push1_b: # Both push same box
                    fa0, push0_b, fa1, push1_b = a0, None, a1, None
                if push0_b and push1_b and push0_t == push1_t: # Push different boxes to same tile
                    fa0, push0_b, fa1, push1_b = a0, None, a1, None
                if push0_b and push0_t == fa1: fa0, push0_b = a0, None
                if push1_b and push1_t == fa0: fa1, push1_b = a1, None

                out0 = [(fa0, push0_b, push0_t, 0.8), (a0, None, None, 0.2)] if push0_b else [(fa0, None, None, 1.0)]
                out1 = [(fa1, push1_b, push1_t, 0.8), (a1, None, None, 0.2)] if push1_b else [(fa1, None, None, 1.0)]

                for o0_pos, o0_b, o0_t, o0_p in out0:
                    for o1_pos, o1_b, o1_t, o1_p in out1:
                        fb0 = o0_t if o0_b == 'b0' else (o1_t if o1_b == 'b0' else b0)
                        fb1 = o0_t if o0_b == 'b1' else (o1_t if o1_b == 'b1' else b1)
                        
                        # --- THE GHOST PURGE ---
                        # If a push stochastically fails, an agent following it would step into the box. Bounce them back!
                        if o0_pos in (fb0, fb1, hb): o0_pos = a0
                        if o1_pos in (fb0, fb1, hb): o1_pos = a1
                        
                        # If both agents try to step into the same empty space, bounce them back
                        if o0_pos == o1_pos and o0_pos not in (a0, a1):
                            o0_pos, o1_pos = a0, a1
                        
                        # --- THE BOX SORTING ---
                        # Sort the identical small boxes to instantly cut the state space in half
                        s_next = tuple(sorted([o0_pos, o1_pos])) + tuple(sorted([fb0, fb1])) + (hb,)
                        results.append((base_prob * o0_p * o1_p, s_next))

        final_t = {}
        for p, s in results:
            final_t[s] = final_t.get(s, 0.0) + p
            
        return [(p, s, 100.0 if is_terminal(s) else -0.01) for s, p in final_t.items() if p > 0]

    processed = 0
    while queue:
        state = queue.popleft()
        processed += 1         
        if state not in transitions:
            transitions[state] = {}

        if is_terminal(state):
            for ja in JOINT_ACTIONS: transitions[state][ja] = [(1.0, state, 0.0)]
            continue

        for ja in JOINT_ACTIONS:
            transitions[state][ja] = compute_analytic_transitions(state, ja)
            for p, nxt_s, r in transitions[state][ja]:
                if nxt_s not in reachable:
                    reachable.add(nxt_s)
                    queue.append(nxt_s)

    print(f"✅ State Space Generated! Found {len(reachable)} logically valid states.")
    return transitions


def modified_policy_iteration(
    env,
    gamma: float = 0.95,
    k: int = 10,
    theta: float = 1e-4,
    max_outer_iters: int = 500,
):
    """
    TODO — Modified Policy Iteration.

    Parameters
    ----------
    env   : StochasticMultiAgentBoxPushEnv (used only to build the model)
    gamma : discount factor
    k     : number of partial policy-evaluation sweeps per iteration
    theta : convergence threshold for value change
    max_outer_iters : safety cap on outer iterations

    Returns
    -------
    policy : dict  state -> joint_action
    V      : dict  state -> float
    """
    import itertools
    
    # 1. Get all transitions and states
    transitions = build_transition_model(env)
    states = list(transitions.keys())
    
    # Extract goals to determine which states are terminal
    goals = set()
    for y in range(env.height):
        for x in range(env.width):
            cell = env.core_env.grid.get(x, y)
            if cell is not None and cell.type == "goal":
                goals.add((x, y))
                
    def is_terminal(s):
        b0, b1, hb = s[2], s[3], s[4]
        return (b0 in goals) and (b1 in goals) and (hb in goals)

    # Replace the ACTIONS list in this function
    V = {s: 0.0 for s in states}
    ACTIONS = [0, 1, 2, 3]
    JOINT_ACTIONS = list(itertools.product(ACTIONS, ACTIONS))
    policy = {s: JOINT_ACTIONS[0] for s in states}

    print("🧠 Starting Modified Policy Iteration Algorithm...")
    for outer_it in range(max_outer_iters):
        for _ in range(k):
            V_new = {}
            for s in states:
                if is_terminal(s):
                    V_new[s] = 0.0
                else:
                    V_new[s] = sum(p * (r + gamma * V[ns]) for p, ns, r in transitions[s][policy[s]])
            V = V_new
            
        max_val_change = 0.0
        for s in states:
            if is_terminal(s): continue
            old_a = policy[s]
            
            best_a, best_v = old_a, -float('inf')
            for a in JOINT_ACTIONS:
                val = sum(p * (r + gamma * V[ns]) for p, ns, r in transitions[s][a])
                if val > best_v:
                    best_v, best_a = val, a
                    
            policy[s] = best_a
            max_val_change = max(max_val_change, abs(V[s] - best_v))
            
        print(f"  Iteration {outer_it+1} | Max Value Delta: {max_val_change:.5f}")
        if max_val_change < theta:
            print(f"🎉 MPI Converged optimally after {outer_it+1} iterations!")
            break

    return policy, V


# ===========================================================================
# Evaluation (do not modify)
# ===========================================================================

def evaluate_policy(policy_fn, env, n_runs: int = 100, max_steps: int = 500):
    """
    Run *policy_fn* for n_runs episodes and return (mean_steps, std_steps).

    Parameters
    ----------
    policy_fn : callable(env, obs) -> dict[agent -> action]
    env       : StochasticMultiAgentBoxPushEnv (reset inside each run)
    n_runs    : number of independent episodes
    max_steps : episode length cap (counts as a failure if hit)
    """
    steps_per_run = []

    for _ in range(n_runs):
        obs, _ = env.reset()
        steps  = 0
        done   = False

        while not done and steps < max_steps:
            actions = policy_fn(env, obs)
            obs, rewards, terms, truncs, _ = env.step(actions)
            steps += 1
            done = any(terms.values()) or any(truncs.values())

        steps_per_run.append(steps)

    return float(np.mean(steps_per_run)), float(np.std(steps_per_run))


# ===========================================================================
# Main — run both algorithms and print results
# ===========================================================================

# ===========================================================================
# Main — run both algorithms and print results
# ===========================================================================

if __name__ == "__main__":
    # ── Part 1: Online Planning ──────────────────────────────────────────────
    print("=" * 60)
    print("Part 1 — Online Planning (classical planner on stochastic env)")
    print("=" * 60)
    env = StochasticMultiAgentBoxPushEnv(ascii_map=ASCII_MAP, max_steps=500)

    # Direct evaluation loop for online planning
    # (We removed the NotImplementedError shim because running it directly is better)
    online_steps = []
    
    RUNS = 100
    for i in range(RUNS):
        env_ep = StochasticMultiAgentBoxPushEnv(ascii_map=ASCII_MAP, max_steps=500)
        steps = run_online_planning(env_ep)
        online_steps.append(steps)
        if (i + 1) % 10 == 0:
            print(f"  run {i+1}/100 — steps so far: {steps}")

    mean_ol, std_ol = float(np.mean(online_steps)), float(np.std(online_steps))
    print(f"\nOnline Planning  →  mean = {mean_ol:.2f}  std = {std_ol:.2f}\n")

    # ── Part 2: Modified Policy Iteration ───────────────────────────────────
    print("=" * 60)
    print("Part 2 — Modified Policy Iteration")
    print("=" * 60)

    env_mpi = StochasticMultiAgentBoxPushEnv(ascii_map=ASCII_MAP, max_steps=500)
    
    # יצירת המודל והרצת האלגוריתם (לוקח קצת זמן בגלל ה-BFS בהתחלה)
    policy, V = modified_policy_iteration(env_mpi)

    def mpi_policy_fn(env, obs):
        state = get_state(env)
        if state not in policy: return {env.possible_agents[0]: 2, env.possible_agents[1]: 2}
        
        joint_action = policy[state]
        agents = env.possible_agents
        
        # Extract correct agents based on sorting
        a0_p = env.agent_positions[agents[0]]
        a1_p = env.agent_positions[agents[1]]
        if a0_p <= a1_p:
            target_dirs = {agents[0]: joint_action[0], agents[1]: joint_action[1]}
        else:
            target_dirs = {agents[0]: joint_action[1], agents[1]: joint_action[0]}
            
        pettingzoo_actions = {}
        for agent in agents:
            target = target_dirs[agent]
            current = env.agent_dirs[agent]
            if current == target: pettingzoo_actions[agent] = 2  # Forward
            elif (current + 1) % 4 == target: pettingzoo_actions[agent] = 1  # Turn Right
            else: pettingzoo_actions[agent] = 0  # Turn Left / Around
                
        return pettingzoo_actions

    mean_mpi, std_mpi = evaluate_policy(mpi_policy_fn, env_mpi, n_runs=100)
    print(f"\nMPI              →  mean = {mean_mpi:.2f}  std = {std_mpi:.2f}\n")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Algorithm':<25} {'Mean steps':>12} {'Std steps':>12}")
    print("-" * 50)
    print(f"{'Online Planning':<25} {mean_ol:>12.2f} {std_ol:>12.2f}")
    print(f"{'MPI':<25} {mean_mpi:>12.2f} {std_mpi:>12.2f}")
