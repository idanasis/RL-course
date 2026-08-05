"""
Partially Observable Monte-Carlo Planning (POMCP) for the Box Pushing agent.

POMCP (Silver & Veness, 2010) plans in a POMDP by running many Monte-Carlo
simulations from states sampled out of the current belief, growing a search
tree indexed by *action-observation history* rather than by state. This
implementation runs directly on top of the ``ParticleFilter`` built in
``practical_filter.py``: each simulation starts by sampling a candidate agent
position from the filter's particle cloud.

State vs. belief
----------------
The filter tracks uncertainty over the agent's ``(x, y)`` position only. POMCP,
however, plans over the *real* rotation-based action space
``{0: left, 1: right, 2: forward}``, so a full simulation state also needs the
agent's heading and the box layout:

    state = (agent_pos, agent_dir, boxes)

- ``agent_pos`` is sampled from the particle filter.
- ``agent_dir`` is unobserved, so it is sampled uniformly from {0,1,2,3} at the
  start of every simulation (principled treatment of hidden heading).
- ``boxes`` is known from the static map and threaded (and mutated on pushes)
  through the generative model.

The observation model is the translation-based 3x3 egocentric window and depends
only on ``(agent_pos, boxes)`` -- never on heading -- which is exactly what
makes the filter's position-only belief compatible with this planner.

Generative model ``G(s, a) -> (s', o, r, terminal)``
----------------------------------------------------
- left/right rotate the heading in place.
- forward moves one cell along the heading if free; walls/out-of-bounds block
  it; a *small* box is pushed if the cell beyond is free; a *heavy* box cannot
  be moved by a single agent (blocked).
- reward is 1.0 (and terminal) once every goal cell holds a box, else 0.0.

No external planning libraries are used.
"""

import math
import random
import time

import numpy as np
from minigrid.core.constants import DIR_TO_VEC
from minigrid.core.world_object import Wall, Goal

try:
    from .observation_function import WALL, EMPTY, GOAL, SMALL_BOX, HEAVY_BOX
except ImportError:  # pragma: no cover - fallback for direct-script execution
    from observation_function import WALL, EMPTY, GOAL, SMALL_BOX, HEAVY_BOX


# Real env action space.
LEFT, RIGHT, FORWARD = 0, 1, 2
ACTIONS = (LEFT, RIGHT, FORWARD)


class POMCPNode:
    """A history node in the search tree: visit counts and Q-values per action."""

    __slots__ = ("N", "Na", "Qa")

    def __init__(self, num_actions):
        self.N = 0                       # N(h)   -- visits to this history
        self.Na = [0] * num_actions      # N(h,a) -- visits to each action
        self.Qa = [0.0] * num_actions    # V(h,a) -- estimated action values


class POMCP:
    def __init__(
        self,
        particle_filter,
        gamma=0.95,
        c=1.0,
        max_depth=50,
        epsilon=0.01,
        seed=None,
    ):
        self.pf = particle_filter
        self.gamma = gamma
        self.c = c
        self.max_depth = max_depth
        self.epsilon = epsilon                 # depth cutoff: gamma^depth < epsilon
        self.rng = random.Random(seed)
        # Optional known heading. The agent's *position* is hidden, but its
        # heading is not (we know the start dir and every rotation we command),
        # so the online loop can pin it here instead of sampling uniformly.
        self.known_dir = None

        self.width = self.pf.width
        self.height = self.pf.height

        # Precompute the static layout (walls / goals / empty) from the filter's
        # known map. Boxes are tracked dynamically in the state, so box cells are
        # recorded here as empty floor.
        self._static = [[EMPTY] * self.width for _ in range(self.height)]
        self.goals = set()
        initial_boxes = set()
        for y in range(self.height):
            for x in range(self.width):
                cell = self.pf.grid.get(x, y)
                if cell is None:
                    continue
                size = getattr(cell, "box_size", None)
                if size == "small":
                    initial_boxes.add((x, y, "small"))
                elif size == "heavy":
                    initial_boxes.add((x, y, "heavy"))
                elif isinstance(cell, Wall):
                    self._static[y][x] = WALL
                elif isinstance(cell, Goal):
                    self._static[y][x] = GOAL
                    self.goals.add((x, y))
        self.initial_boxes = frozenset(initial_boxes)
        self.window_size = self.pf.window_size

        # tree: history (tuple of (action, obs_key)) -> POMCPNode
        self.tree = {}

    # ------------------------------------------------------------------
    # Generative model
    # ------------------------------------------------------------------
    def _in_bounds(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height

    def _observation(self, agent_pos, box_map):
        """3x3 egocentric window from ``agent_pos``; out-of-bounds padded WALL."""
        half = self.window_size // 2
        ax, ay = agent_pos
        obs = np.full((self.window_size, self.window_size), WALL, dtype=np.int8)
        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                gx, gy = ax + dx, ay + dy
                if not self._in_bounds(gx, gy):
                    continue  # keep WALL padding
                row, col = dy + half, dx + half
                if (gx, gy) in box_map:
                    obs[row, col] = (
                        SMALL_BOX if box_map[(gx, gy)] == "small" else HEAVY_BOX
                    )
                else:
                    obs[row, col] = self._static[gy][gx]
        return obs

    def _all_on_goals(self, box_positions):
        return self.goals.issubset(box_positions)

    def step_model(self, state, action):
        """Generative model G(s, a) -> (next_state, obs_key, reward, terminal)."""
        pos, d, boxes = state
        box_map = {(x, y): s for (x, y, s) in boxes}

        if action == LEFT:
            new_pos, new_dir, new_boxes = pos, (d - 1) % 4, boxes
        elif action == RIGHT:
            new_pos, new_dir, new_boxes = pos, (d + 1) % 4, boxes
        else:  # FORWARD
            new_dir = d
            vx, vy = DIR_TO_VEC[d]
            fx, fy = pos[0] + vx, pos[1] + vy

            if not self._in_bounds(fx, fy) or self._static[fy][fx] == WALL:
                new_pos, new_boxes = pos, boxes                       # blocked
            elif (fx, fy) in box_map:
                size = box_map[(fx, fy)]
                bx, by = fx + vx, fy + vy
                if (
                    size == "small"
                    and self._in_bounds(bx, by)
                    and self._static[by][bx] != WALL
                    and (bx, by) not in box_map
                ):
                    nb = set(boxes)
                    nb.discard((fx, fy, size))
                    nb.add((bx, by, size))
                    new_pos, new_boxes = (fx, fy), frozenset(nb)      # push small box
                else:
                    new_pos, new_boxes = pos, boxes                   # heavy / blocked
            else:
                new_pos, new_boxes = (fx, fy), boxes                  # free move

        next_box_map = {(x, y): s for (x, y, s) in new_boxes}
        box_positions = set(next_box_map.keys())
        terminal = self._all_on_goals(box_positions)
        reward = 1.0 if terminal else 0.0
        obs = self._observation(new_pos, next_box_map)
        return (new_pos, new_dir, new_boxes), obs.tobytes(), reward, terminal

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def _sample_state(self):
        """Draw a full state from the belief: position ~ filter, heading ~ U{0..3}."""
        if not self.pf.particles:
            raise ValueError("Particle filter is empty; cannot sample a state.")
        pos = self.rng.choice(self.pf.particles)
        d = self.known_dir if self.known_dir is not None else self.rng.randrange(4)
        return (pos, d, self.initial_boxes)

    def search(self, time_budget, root_history=(), known_dir=None):  # noqa: C901
        """
        Run POMCP simulations from the current belief until ``time_budget``
        seconds elapse, then return the best action at the root.

        If ``known_dir`` is given, every sampled start state uses that heading
        instead of a uniformly-random one (the heading is observable in the
        online loop even though the position is not).
        """
        self.known_dir = known_dir
        if root_history not in self.tree:
            self.tree[root_history] = POMCPNode(len(ACTIONS))

        start = time.perf_counter()
        n_sims = 0
        while time.perf_counter() - start < time_budget:
            state = self._sample_state()
            self.simulate(state, root_history, depth=0)
            n_sims += 1

        self.last_num_simulations = n_sims
        return self.best_action(root_history)

    def simulate(self, state, history, depth):
        """Recursively traverse/expand the tree, backing up the discounted return."""
        if self.gamma ** depth < self.epsilon or depth >= self.max_depth:
            return 0.0

        # Unexpanded history: add the node and hand off to a rollout.
        if history not in self.tree:
            self.tree[history] = POMCPNode(len(ACTIONS))
            return self.rollout(state, depth)

        node = self.tree[history]
        action = self._ucb_select(node)

        next_state, obs_key, reward, terminal = self.step_model(state, action)
        if terminal:
            total = reward
        else:
            future = self.simulate(
                next_state, history + ((action, obs_key),), depth + 1
            )
            total = reward + self.gamma * future

        # Backpropagate: visit counts and incremental-mean Q update.
        node.N += 1
        node.Na[action] += 1
        node.Qa[action] += (total - node.Qa[action]) / node.Na[action]
        return total

    def rollout(self, state, depth):
        """Fast random-policy simulation returning the discounted return."""
        if self.gamma ** depth < self.epsilon or depth >= self.max_depth:
            return 0.0
        action = self.rng.choice(ACTIONS)
        next_state, _obs, reward, terminal = self.step_model(state, action)
        if terminal:
            return reward
        return reward + self.gamma * self.rollout(next_state, depth + 1)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------
    def _ucb_select(self, node):
        """UCB1: prefer unvisited actions, else argmax Q + c*sqrt(ln N / N(a))."""
        log_n = math.log(node.N + 1)
        best_action, best_score = ACTIONS[0], -math.inf
        for a in ACTIONS:
            if node.Na[a] == 0:
                return a  # explore every action at least once
            score = node.Qa[a] + self.c * math.sqrt(log_n / node.Na[a])
            if score > best_score:
                best_score, best_action = score, a
        return best_action

    def best_action(self, history=()):
        """Greedy best action at a history node by estimated value (ties: visits)."""
        node = self.tree.get(history)
        if node is None:
            return None
        return max(ACTIONS, key=lambda a: (node.Qa[a], node.Na[a]))

    def action_values(self, history=()):
        """Return (Qa, Na) at a history node for inspection/testing."""
        node = self.tree.get(history)
        if node is None:
            return None, None
        return list(node.Qa), list(node.Na)


# ---------------------------------------------------------------------------
# Test block
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from environment import MultiAgentBoxPushEnv
    try:
        from .practical_filter import ParticleFilter
        from .observation_function import get_agent_observation
    except ImportError:
        from practical_filter import ParticleFilter
        from observation_function import get_agent_observation

    # Corridor: agent must face East and push the small box (B) onto the goal (G).
    #   (1,1)=agent  (3,1)=small box  (4,1)=goal
    ascii_map = [
        "WWWWWW",
        "WA BGW",
        "WWWWWW",
    ]

    env = MultiAgentBoxPushEnv(ascii_map=ascii_map)
    env.reset()
    agent = env.possible_agents[0]

    pf = ParticleFilter(ascii_map, n_particles=300, seed=0)
    # Localize from the first real observation.
    pf.update("stay", get_agent_observation(env, agent))
    print(f"Belief after first observation: {sorted(pf.distinct_positions())}")

    planner = POMCP(pf, gamma=0.95, c=1.0, seed=0)

    # ── Unit-test the generative model ────────────────────────────────────
    # Agent at (2,1) facing East (dir 0), pushes box (3,1) -> goal (4,1): terminal.
    s0 = ((2, 1), 0, planner.initial_boxes)
    s1, _o, r, term = planner.step_model(s0, FORWARD)
    assert s1[0] == (3, 1), "agent should advance into the box's old cell"
    assert (4, 1, "small") in s1[2], "small box should be pushed onto the goal"
    assert term and r == 1.0, "all-boxes-on-goals must be terminal with reward 1"
    print("Generative model: forward-push onto goal is terminal (reward 1.0). OK")

    # Facing West into the wall at (0,1): blocked, no movement, not terminal.
    s_block, _o, r_b, term_b = planner.step_model(((1, 1), 2, planner.initial_boxes), FORWARD)
    assert s_block[0] == (1, 1) and not term_b and r_b == 0.0
    print("Generative model: forward into wall is a no-op. OK")

    # Rotations change heading only.
    s_rot, _o, _r, _t = planner.step_model(((1, 1), 0, planner.initial_boxes), LEFT)
    assert s_rot[0] == (1, 1) and s_rot[1] == 3
    print("Generative model: rotate-left changes heading only. OK")

    # ── Run the planner ───────────────────────────────────────────────────
    best = planner.search(time_budget=0.5)
    Qa, Na = planner.action_values()
    root = planner.tree[()]

    assert best in ACTIONS
    assert root.N > 0 and planner.last_num_simulations > 0
    assert sum(Na) == root.N, "per-action visit counts must sum to the node's visits"
    assert all(a >= 0 for a in Na) and all(math.isfinite(q) for q in Qa)

    action_names = {LEFT: "left", RIGHT: "right", FORWARD: "forward"}
    print(f"\nPlanned {planner.last_num_simulations} simulations "
          f"({root.N} root visits, {len(planner.tree)} tree nodes).")
    for a in ACTIONS:
        print(f"  action={action_names[a]:8s}  N(h,a)={Na[a]:5d}  Q(h,a)={Qa[a]:.4f}")
    print(f"Best action at root: {action_names[best]}")

    print("\nAll POMCP tests passed.")
