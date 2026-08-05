"""
Execution & evaluation loop for the POMCP box-pushing agent.

This ties the four ex4 pieces together:

  * ``StochasticMultiAgentBoxPushEnv`` -- the real (stochastic) environment,
  * ``get_agent_observation``          -- the custom translation-based 3x3 obs,
  * ``ParticleFilter``                 -- belief over the hidden agent position,
  * ``POMCP``                          -- the online planner,

and runs the standard online-planning loop:

    plan an action with POMCP  ->  execute it in the env  ->  read the real
    observation  ->  ParticleFilter.update(action, observation)  ->  repeat.

Integration notes
------------------
* Only the agent's *position* is hidden. Its *heading* is not: we know the
  start heading (``env.agent_dirs``) and every rotation we command, so the loop
  tracks the heading exactly and (a) feeds it to POMCP via ``known_dir`` and
  (b) uses it to translate a "forward" action into the cardinal move vector the
  position-only particle filter needs. Rotations are "stay" moves for the
  filter (position unchanged).

* The particle filter's transition model is deterministic while the env is
  stochastic, but that is fine: observations are deterministic and highly
  informative, so mispredicted particles are rejected and the map-knowledge
  fallback re-derives the consistent positions. The observation always drives
  localization.

* The filter's map treats boxes as static. In the single-box-onto-goal tasks
  used here a box only ever moves on the terminal push (which ends the episode),
  so the filter's map stays valid throughout the localization-relevant portion.

* POMCP's search tree is history-indexed and belief-specific, so it is cleared
  before every planning step (fresh search from the current belief).
"""

import argparse

import numpy as np
from minigrid.core.constants import DIR_TO_VEC

from environment import StochasticMultiAgentBoxPushEnv

try:
    from .observation_function import get_agent_observation
    from .practical_filter import ParticleFilter
    from .pomcp import POMCP, LEFT, RIGHT, FORWARD
except ImportError:  # pragma: no cover - fallback for direct-script execution
    from observation_function import get_agent_observation
    from practical_filter import ParticleFilter
    from pomcp import POMCP, LEFT, RIGHT, FORWARD


# Single-agent room. The agent (A, start (1,1), heading South) must navigate the
# walled path -- down the left column, then right along the bottom -- to reach
# cell (3,5), from which pushing East drives the small box (B, (4,5)) one cell
# onto the goal (G, (5,5)). The reward is sparse (only the terminal push pays
# off) and the horizon is ~8 actions, so the planning time budget matters. The
# box moves only on that terminal push, so the filter's static-box map stays
# valid, and the internal walls give every cell a distinct enough observation
# for drift-free localization.
EVAL_MAP = [
    "WWWWWWW",
    "WA    W",
    "W WWW W",
    "W W   W",
    "W W W W",
    "W   BGW",
    "WWWWWWW",
]

ACTION_NAMES = {LEFT: "left", RIGHT: "right", FORWARD: "forward"}


def filter_move_and_heading(action, heading):
    """
    Translate a real env action + current heading into (filter_move, new_heading).

    The filter tracks position only, so:
      * forward -> the cardinal (dx, dy) vector of the current heading,
      * left/right -> "stay" (position unchanged) while the heading rotates.
    """
    if action == FORWARD:
        vx, vy = DIR_TO_VEC[heading]
        return (int(vx), int(vy)), heading
    if action == LEFT:
        return "stay", (heading - 1) % 4
    # RIGHT
    return "stay", (heading + 1) % 4


def run_episode(ascii_map, time_budget, seed, max_steps=50, n_particles=300, verbose=False):
    """
    Run one online-planning episode. Returns ``(steps, reached_goal)``.
    """
    # Seed the global numpy RNG that drives the env's stochastic transitions,
    # plus the filter/planner RNGs, so episodes are independent & reproducible.
    np.random.seed(seed)

    env = StochasticMultiAgentBoxPushEnv(ascii_map=ascii_map, max_steps=max_steps)
    env.reset(seed=seed)
    agent = env.possible_agents[0]

    pf = ParticleFilter(ascii_map, n_particles=n_particles, seed=seed)
    planner = POMCP(pf, gamma=0.95, c=1.0, seed=seed)

    # Heading is known (only position is hidden); track it exactly.
    heading = env.agent_dirs[agent]

    # Localize from the very first real observation before planning.
    pf.update("stay", get_agent_observation(env, agent))

    steps = 0
    reached_goal = False
    while env.agents:
        # Fresh tree each step: the belief (and thus the root) has changed.
        planner.tree.clear()
        action = planner.search(time_budget, known_dir=heading)

        # Execute in the real environment.
        _obs, _rewards, terminations, truncations, _infos = env.step({agent: action})
        steps += 1

        move, heading = filter_move_and_heading(action, heading)

        if verbose:
            print(
                f"  step {steps:2d}: belief={sorted(pf.distinct_positions())} "
                f"action={ACTION_NAMES[action]:7s} -> "
                f"term={any(terminations.values())}"
            )

        if any(terminations.values()):
            reached_goal = True
            break
        if any(truncations.values()) or not env.agents:
            break

        # Read the real observation and correct the belief.
        real_obs = get_agent_observation(env, agent)
        if not pf.particles:  # defensive: never expected with informative obs
            pf.initialize_particles()
        pf.update(move, real_obs)

    return steps, reached_goal


def evaluate(time_budget, n_episodes=30, ascii_map=EVAL_MAP, base_seed=0,
             max_steps=50, verbose=False):
    """
    Run ``n_episodes`` independent episodes at a fixed ``time_budget`` per step.
    Returns a stats dict with success rate and mean/std of steps-to-goal.
    """
    steps_per_ep = []
    successes = 0
    for ep in range(n_episodes):
        steps, ok = run_episode(
            ascii_map, time_budget, seed=base_seed + ep,
            max_steps=max_steps, verbose=verbose,
        )
        steps_per_ep.append((steps, ok))
        successes += int(ok)
        if verbose:
            print(f"episode {ep:2d}: steps={steps} reached_goal={ok}")

    solved_steps = np.array([s for s, ok in steps_per_ep if ok], dtype=float)
    return {
        "time_budget": time_budget,
        "n_episodes": n_episodes,
        "successes": successes,
        "success_rate": successes / n_episodes,
        "mean_steps": float(solved_steps.mean()) if solved_steps.size else float("nan"),
        "std_steps": float(solved_steps.std()) if solved_steps.size else float("nan"),
    }


def _print_stats(stats):
    tb = stats["time_budget"]
    print(
        f"time_budget = {tb:>5.1f}s | "
        f"success {stats['successes']:2d}/{stats['n_episodes']} | "
        f"steps to goal: mean={stats['mean_steps']:5.2f}  std={stats['std_steps']:5.2f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate the POMCP box-pushing agent.")
    parser.add_argument("--episodes", type=int, default=30,
                        help="episodes per configuration (default: 30)")
    parser.add_argument("--budgets", type=float, nargs="+", default=[1.0, 20.0],
                        help="per-step planning time budgets in seconds (default: 1.0 20.0)")
    parser.add_argument("--max-steps", type=int, default=50,
                        help="max env steps before truncation (default: 50)")
    parser.add_argument("--verbose", action="store_true",
                        help="print a per-step / per-episode trace")
    args = parser.parse_args()

    print(f"=== POMCP evaluation | map {EVAL_MAP} | {args.episodes} episodes/config ===")
    print("(note: wall-clock cost per config ~= episodes * steps * time_budget;\n"
          " the 20.0s configuration can take tens of minutes.)\n")

    results = []
    for tb in args.budgets:
        stats = evaluate(tb, n_episodes=args.episodes,
                         max_steps=args.max_steps, verbose=args.verbose)
        results.append(stats)
        _print_stats(stats)

    return results


if __name__ == "__main__":
    main()
