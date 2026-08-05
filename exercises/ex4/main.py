"""
Execution & evaluation loop for the POMCP box-pushing agent.

This ties the four ex4 pieces together:

  * ``StochasticMultiAgentBoxPushEnv`` -- the real (stochastic) environment,
  * ``get_agent_observation``          -- the custom translation-based 3x3 obs,
  * ``ParticleFilter``                 -- belief over the hidden agent position,
  * ``POMCP``                          -- the online planner,

and runs the standard online-planning loop:

    plan an action with POMCP  ->  execute it in the env  ->  read the real
    observation  ->  ParticleFilter.update(...)  ->  repeat.

Integration notes
------------------
* One shared generative model. Both POMCP (tree simulations) and the particle
  filter (rejection-sampling belief update) call the SAME stochastic
  ``planner.sample_step`` -- the env's dynamics (move 0.8 / +-90 deg, push 0.8).

* Only the agent's *position* is hidden. Its *heading* is observable: we know the
  start heading (``env.agent_dirs``) and every rotation we command, so the loop
  tracks it exactly and feeds it to both POMCP (``known_dir``) and the filter's
  transition. This keeps the hidden state = position, as the assignment defines.

* Action space is the real env's rotation-based ``{0: left, 1: right, 2:
  forward}``. (The assignment text describes direct cardinal moves, but the
  provided environment is rotation-based; we follow the actual environment.)

* POMCP's search tree is history-indexed and belief-specific, so it is cleared
  before every planning step (a fresh search from the current belief).

* Multi-agent: each robot runs its own filter + POMCP (decentralized). They act
  simultaneously; each localizes from its own observation. The single-agent
  generative model does not model the other robot, so this is an approximation
  (fine for the one-box task here; tight coordination such as the heavy box is
  out of reach for independent planners).
"""

import argparse

import numpy as np

from environment import StochasticMultiAgentBoxPushEnv

try:
    from .observation_function import get_agent_observation
    from .practical_filter import ParticleFilter
    from .pomcp import POMCP, LEFT, RIGHT, FORWARD
except ImportError:  # pragma: no cover - fallback for direct-script execution
    from observation_function import get_agent_observation
    from practical_filter import ParticleFilter
    from pomcp import POMCP, LEFT, RIGHT, FORWARD


# Open room, box (B) sits one cell above the goal (G) so a single South push
# solves it. Both scenarios share this layout; only the number of agents differs,
# giving a controlled single-vs-multi comparison.
#   single: agent at (1,1); multi: agents at (1,1) and (5,1). Box (3,4), goal (3,5).
SINGLE_MAP = [
    "WWWWWWW",
    "WA    W",
    "W     W",
    "W     W",
    "W  B  W",
    "W  G  W",
    "WWWWWWW",
]

MULTI_MAP = [
    "WWWWWWW",
    "WA   AW",
    "W     W",
    "W     W",
    "W  B  W",
    "W  G  W",
    "WWWWWWW",
]

ACTION_NAMES = {LEFT: "left", RIGHT: "right", FORWARD: "forward"}


def next_heading(action, heading):
    """Deterministic heading after a rotation; forward leaves it unchanged."""
    if action == LEFT:
        return (heading - 1) % 4
    if action == RIGHT:
        return (heading + 1) % 4
    return heading


def _make_agent(ascii_map, n_particles, seed):
    """Build a (filter, planner) pair for one agent."""
    pf = ParticleFilter(ascii_map, n_particles=n_particles, seed=seed)
    planner = POMCP(pf, gamma=0.95, c=1.0, seed=seed)
    return pf, planner


def _localize_initial(pf, env, agent):
    """Filter the uniform prior by the first real observation."""
    pf.recover_from_map(get_agent_observation(env, agent))
    if not pf.particles:
        pf.initialize_particles()


def run_episode(ascii_map, time_budget, seed, max_steps=50, n_particles=500, verbose=False):
    """One single-agent online-planning episode. Returns ``(steps, reached_goal)``."""
    np.random.seed(seed)  # controls the env's stochastic transitions
    env = StochasticMultiAgentBoxPushEnv(ascii_map=ascii_map, max_steps=max_steps)
    env.reset(seed=seed)
    agent = env.possible_agents[0]

    pf, planner = _make_agent(ascii_map, n_particles, seed)
    boxes = planner.initial_boxes
    heading = env.agent_dirs[agent]            # heading is known (only position is hidden)
    _localize_initial(pf, env, agent)

    steps, reached_goal = 0, False
    while env.agents:
        planner.tree.clear()                   # fresh search from the current belief
        action = planner.search(time_budget, known_dir=heading)

        _obs, _r, terminations, truncations, _i = env.step({agent: action})
        steps += 1
        pre_heading, heading = heading, next_heading(action, heading)

        if verbose:
            print(f"  step {steps:2d}: belief={sorted(pf.distinct_positions())} "
                  f"action={ACTION_NAMES[action]:7s} -> term={any(terminations.values())}")

        if any(terminations.values()):
            reached_goal = True
            break
        if any(truncations.values()) or not env.agents:
            break

        real_obs = get_agent_observation(env, agent)
        if not pf.particles:
            pf.initialize_particles()
        pf.update(action, real_obs, pre_heading, boxes, planner.sample_step)

    return steps, reached_goal


def run_episode_multi(ascii_map, time_budget, seed, max_steps=50, n_particles=500, verbose=False):
    """One multi-agent (decentralized) online-planning episode -> ``(steps, goal)``."""
    np.random.seed(seed)
    env = StochasticMultiAgentBoxPushEnv(ascii_map=ascii_map, max_steps=max_steps)
    env.reset(seed=seed)
    agents = list(env.possible_agents)

    pf, planner = {}, {}
    for i, a in enumerate(agents):
        pf[a], planner[a] = _make_agent(ascii_map, n_particles, seed + i)
    heading = {a: env.agent_dirs[a] for a in agents}
    for a in agents:
        _localize_initial(pf[a], env, a)

    steps, reached_goal = 0, False
    while env.agents:
        current = list(env.agents)
        actions = {}
        for a in current:                      # each robot plans independently
            planner[a].tree.clear()
            actions[a] = planner[a].search(time_budget, known_dir=heading[a])

        _obs, _r, terminations, truncations, _i = env.step(actions)
        steps += 1
        pre_heading = dict(heading)
        for a in current:
            heading[a] = next_heading(actions[a], heading[a])

        if verbose:
            trace = " | ".join(f"{a}:{ACTION_NAMES[actions[a]]}" for a in current)
            print(f"  step {steps:2d}: {trace} -> term={any(terminations.values())}")

        if any(terminations.values()):
            reached_goal = True
            break
        if any(truncations.values()) or not env.agents:
            break

        for a in current:
            real_obs = get_agent_observation(env, a)
            if not pf[a].particles:
                pf[a].initialize_particles()
            pf[a].update(actions[a], real_obs, pre_heading[a],
                         planner[a].initial_boxes, planner[a].sample_step)

    return steps, reached_goal


def evaluate(time_budget, n_episodes=30, scenario="single", base_seed=0,
             max_steps=50, verbose=False):
    """
    Run ``n_episodes`` independent episodes of ``scenario`` ("single"/"multi") at
    a fixed per-step ``time_budget``. Returns a stats dict.
    """
    ascii_map = MULTI_MAP if scenario == "multi" else SINGLE_MAP
    runner = run_episode_multi if scenario == "multi" else run_episode

    results = []
    successes = 0
    for ep in range(n_episodes):
        steps, ok = runner(ascii_map, time_budget, seed=base_seed + ep,
                           max_steps=max_steps, verbose=verbose)
        results.append((steps, ok))
        successes += int(ok)
        if verbose:
            print(f"[{scenario}] episode {ep:2d}: steps={steps} reached_goal={ok}")

    solved = np.array([s for s, ok in results if ok], dtype=float)
    return {
        "scenario": scenario,
        "time_budget": time_budget,
        "n_episodes": n_episodes,
        "successes": successes,
        "success_rate": successes / n_episodes,
        "mean_steps": float(solved.mean()) if solved.size else float("nan"),
        "std_steps": float(solved.std()) if solved.size else float("nan"),
    }


def _print_stats(stats):
    print(
        f"{stats['scenario']:6s} | time_budget = {stats['time_budget']:>5.1f}s | "
        f"success {stats['successes']:2d}/{stats['n_episodes']} | "
        f"steps to goal: mean={stats['mean_steps']:5.2f}  std={stats['std_steps']:5.2f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate the POMCP box-pushing agent.")
    parser.add_argument("--episodes", type=int, default=30,
                        help="episodes per (scenario, budget) configuration (default: 30)")
    parser.add_argument("--budgets", type=float, nargs="+", default=[1.0, 20.0],
                        help="per-step planning time budgets in seconds (default: 1.0 20.0)")
    parser.add_argument("--scenarios", nargs="+", default=["single", "multi"],
                        choices=["single", "multi"],
                        help="which scenarios to run (default: single multi)")
    parser.add_argument("--max-steps", type=int, default=50,
                        help="max env steps before truncation (default: 50)")
    parser.add_argument("--verbose", action="store_true",
                        help="print a per-step / per-episode trace")
    args = parser.parse_args()

    print(f"=== POMCP evaluation | {args.episodes} episodes per (scenario, budget) ===")
    print("(note: wall-clock ~= episodes * steps * budget, x2 agents for 'multi';\n"
          " the 20.0s configurations can take a long time.)\n")

    results = []
    for scenario in args.scenarios:
        for tb in args.budgets:
            stats = evaluate(tb, n_episodes=args.episodes, scenario=scenario,
                             max_steps=args.max_steps, verbose=args.verbose)
            results.append(stats)
            _print_stats(stats)

    return results


if __name__ == "__main__":
    main()
