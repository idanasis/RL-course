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
- ``agent_dir`` (heading) is *observable* in the online loop -- we know the start
  dir and every rotation we command -- so it is pinned via ``known_dir`` (it can
  still fall back to a uniform sample from {0,1,2,3} when unknown).
- ``boxes`` is known from the static map and threaded (and mutated on pushes)
  through the generative model.

The observation model is the translation-based 3x3 egocentric window and depends
only on ``(agent_pos, boxes)`` -- never on heading -- which is exactly what
makes the filter's position-only belief compatible with this planner.

Stochastic generative model ``G(s, a) -> (s', o, r, terminal)`` (``sample_step``)
--------------------------------------------------------------------------------
Matches the env's stochastic dynamics, and is the SAME model the particle filter
uses for its rejection-sampling belief update:
- left/right rotate the heading in place (deterministic).
- forward: if the intended cell is free the agent MOVES, but the move direction
  succeeds w.p. ``move_success_prob`` (0.8) and deviates +/-90 deg otherwise; a
  deviation into an obstacle leaves the agent put. If the intended cell holds a
  *small* box it is PUSHED w.p. ``push_success_prob`` (0.8); a *heavy* box cannot
  be moved by a single agent; walls/out-of-bounds block the action.
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
        move_success_prob=0.8,
        push_success_prob=0.8,
        seed=None,
    ):
        self.pf = particle_filter
        self.gamma = gamma
        self.c = c
        self.max_depth = max_depth
        self.epsilon = epsilon                 # depth cutoff: gamma^depth < epsilon
        # Stochastic dynamics of the generative model (match the env defaults):
        # a move succeeds in the intended direction w.p. move_success_prob and
        # deviates +/-90 deg otherwise; a push succeeds w.p. push_success_prob.
        self.move_success_prob = move_success_prob
        self.push_success_prob = push_success_prob
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

    def _sample_move_dir(self, intended_dir, rng):
        """Stochastic move direction: intended w.p. p, else deviate +/-90 deg."""
        side = (1.0 - self.move_success_prob) / 2.0
        r = rng.random()
        if r < self.move_success_prob:
            return intended_dir
        elif r < self.move_success_prob + side:
            return (intended_dir - 1) % 4
        else:
            return (intended_dir + 1) % 4

    def sample_step(self, state, action, rng):
        """
        Stochastic generative model G(s, a) -> (next_state, obs_key, reward,
        terminal), matching StochasticMultiAgentBoxPushEnv. Rotations are
        deterministic; a forward MOVE deviates +/-90 deg (prob 1-move_success),
        and a forward PUSH of a small box fails w.p. 1-push_success. This SAME
        method is used both for POMCP tree simulations and for the particle
        filter's rejection-sampling belief update.
        """
        pos, d, boxes = state
        box_map = {(x, y): s for (x, y, s) in boxes}

        if action == LEFT:
            new_pos, new_dir, new_boxes = pos, (d - 1) % 4, boxes
        elif action == RIGHT:
            new_pos, new_dir, new_boxes = pos, (d + 1) % 4, boxes
        else:  # FORWARD
            new_dir = d
            vx, vy = int(DIR_TO_VEC[d][0]), int(DIR_TO_VEC[d][1])
            fx, fy = pos[0] + vx, pos[1] + vy

            if not self._in_bounds(fx, fy) or self._static[fy][fx] == WALL:
                new_pos, new_boxes = pos, boxes                       # intended blocked
            elif (fx, fy) in box_map:
                # PUSH branch (intended cell is a box): no directional deviation.
                size = box_map[(fx, fy)]
                bx, by = fx + vx, fy + vy
                pushable = (
                    size == "small"
                    and self._in_bounds(bx, by)
                    and self._static[by][bx] != WALL
                    and (bx, by) not in box_map
                )
                if pushable and rng.random() < self.push_success_prob:
                    nb = set(boxes)
                    nb.discard((fx, fy, size))
                    nb.add((bx, by, size))
                    new_pos, new_boxes = (fx, fy), frozenset(nb)      # push succeeds
                else:
                    new_pos, new_boxes = pos, boxes                   # heavy / blocked / push failed
            else:
                # MOVE branch (intended cell free): direction may deviate +/-90 deg.
                adir = self._sample_move_dir(d, rng)
                avx, avy = int(DIR_TO_VEC[adir][0]), int(DIR_TO_VEC[adir][1])
                ax, ay = pos[0] + avx, pos[1] + avy
                if (
                    self._in_bounds(ax, ay)
                    and self._static[ay][ax] != WALL
                    and (ax, ay) not in box_map
                ):
                    new_pos, new_boxes = (ax, ay), boxes              # move (possibly deviated)
                else:
                    new_pos, new_boxes = pos, boxes                   # deviated into obstacle

        next_box_map = {(x, y): s for (x, y, s) in new_boxes}
        terminal = self._all_on_goals(set(next_box_map.keys()))
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

        next_state, obs_key, reward, terminal = self.sample_step(state, action, self.rng)
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
        next_state, _obs, reward, terminal = self.sample_step(state, action, self.rng)
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

