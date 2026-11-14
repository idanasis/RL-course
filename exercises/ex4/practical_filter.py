"""
Particle Filter for localizing the agent in the Partially Observable Box
Pushing environment.

The agent does not know its own (x, y) position, but the grid size and the
static wall / box / goal layout ARE known. Observations are the deterministic
3x3 egocentric windows produced by ``observation_function.py``. This filter
maintains a belief over the agent's position as a cloud of particles, each
particle being a single ``(x, y)`` coordinate hypothesis.

The belief is updated by **unweighted rejection sampling** (Silver & Veness,
2010): sample a particle from the current belief, simulate the real action with
the SAME stochastic generative model POMCP uses (``sample_step``, transition +
observation), and keep the resulting position only if its simulated observation
matches the real one -- repeated until ``N`` particles are collected. Because
the transition is stochastic and the observation deterministic, the rejection
rate can be high and the cloud can empty out, so a deterministic "map knowledge"
fallback re-derives the consistent positions directly from the known map.

Localization model
------------------
The filter tracks only the agent's ``(x, y)`` position. Its transition is not a
separate model: ``update`` receives the shared ``sample_step`` plus the known
``heading`` and ``boxes``, and simulates the real env action ``(0=left, 1=right,
2=forward)`` exactly as POMCP does. This guarantees the filter and the planner
use one and the same generative model.

No external POMDP libraries are used.
"""

from collections import Counter, defaultdict

import numpy as np
from minigrid.core.grid import Grid
from minigrid.core.world_object import Wall, Goal

from environment.objects import SmallBox, HeavyBox

# Import the shared egocentric observation machinery. Support running both as a
# package module (``python -m exercises.ex4.practical_filter``) and as a plain
# script from inside the ex4 directory.
try:
    from .observation_function import extract_egocentric_window
except ImportError:  # pragma: no cover - fallback for direct-script execution
    from observation_function import extract_egocentric_window


# ---------------------------------------------------------------------------
# Static-map helpers
# ---------------------------------------------------------------------------
def build_static_grid(ascii_map):
    """
    Build a minigrid ``Grid`` holding only the *static* world objects (walls,
    boxes, goals) from an ascii map, using the exact same tokens the env uses
    (`W` wall, `G` goal, `B` small box, `C` heavy box). Agent (`A`) and blank
    cells become empty floor. Returns ``(grid, width, height)``.
    """
    height = len(ascii_map)
    width = len(ascii_map[0])
    grid = Grid(width, height)
    for y, row in enumerate(ascii_map):
        for x, char in enumerate(row):
            if char == "W":
                grid.set(x, y, Wall())
            elif char == "G":
                grid.set(x, y, Goal())
            elif char == "B":
                grid.set(x, y, SmallBox())
            elif char == "C":
                grid.set(x, y, HeavyBox())
            # 'A' and ' ' -> empty floor (None)
    return grid, width, height


class ParticleFilter:
    """Rejection-sampling particle filter over the agent's ``(x, y)`` position."""

    def __init__(
        self,
        ascii_map,
        n_particles=500,
        window_size=3,
        seed=None,
    ):
        self.grid, self.width, self.height = build_static_grid(ascii_map)
        self.n_particles = n_particles
        self.window_size = window_size
        self.rng = np.random.default_rng(seed)

        # All cells the agent could legally occupy (free floor + overlappable
        # cells such as goals). Boxes and walls are excluded.
        self.free_cells = [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if self._is_free(x, y)
        ]

        # The map is static, so every cell's egocentric observation can be
        # precomputed once. We cache both the observation and a hashable key,
        # and build a reverse index (observation-key -> list of cells) so the
        # deterministic fallback is an O(1) lookup instead of an O(cells) scan
        # on every failure.
        self._obs_by_cell = {}
        self._key_by_cell = {}
        self._cells_by_obs = defaultdict(list)
        for pos in self.free_cells:
            obs = extract_egocentric_window(
                self.grid, pos, self.width, self.height, self.window_size
            )
            key = self._key(obs)
            self._obs_by_cell[pos] = obs
            self._key_by_cell[pos] = key
            self._cells_by_obs[key].append(pos)

        self.particles = []
        self.initialize_particles()

    # ------------------------------------------------------------------
    # Map / geometry helpers
    # ------------------------------------------------------------------
    def _is_free(self, x, y):
        """True iff (x, y) is in bounds and an agent could stand there."""
        if not (0 <= x < self.width and 0 <= y < self.height):
            return False
        cell = self.grid.get(x, y)
        return cell is None or cell.can_overlap()

    @staticmethod
    def _key(obs):
        """Hashable, fast-comparison key for a 3x3 observation matrix."""
        return np.asarray(obs, dtype=np.int8).tobytes()

    def observation_at(self, pos):
        """The egocentric observation an agent at ``pos`` would perceive."""
        if pos in self._obs_by_cell:
            return self._obs_by_cell[pos]
        return extract_egocentric_window(
            self.grid, pos, self.width, self.height, self.window_size
        )

    # ------------------------------------------------------------------
    # Belief lifecycle
    # ------------------------------------------------------------------
    def initialize_particles(self):
        """Spread N particles uniformly over all valid empty cells."""
        if not self.free_cells:
            raise ValueError("No free cells to place particles on.")
        idx = self.rng.integers(0, len(self.free_cells), size=self.n_particles)
        self.particles = [self.free_cells[i] for i in idx]
        return self.particles

    def update(self, action, real_observation, heading, boxes, sample_step,
               max_attempt_factor=30):
        """
        POMCP-style *unweighted rejection-sampling* belief update
        (Silver & Veness, 2010).

        Repeat until ``N`` new particles are collected (or the attempt budget is
        spent):

          1. Sample a particle (position) from the current belief.
          2. Simulate ``action`` on the full state ``(pos, heading, boxes)`` with
             the SAME stochastic generative model POMCP uses (``sample_step``),
             which returns the next state and the observation it would produce.
          3. Keep the resulting position iff that simulated observation exactly
             matches ``real_observation``; otherwise reject and resample.

        Because the transition is stochastic and the observation deterministic,
        the rejection rate can be high and the cloud can empty out. On depletion
        we fall back to deterministic **map-knowledge recovery**: repopulate from
        the known cells whose egocentric slice matches the real observation.

        Args:
            action: real env action (0=left, 1=right, 2=forward).
            real_observation: the 3x3 window actually received.
            heading: the agent's (known) heading before the action.
            boxes: the known box layout (frozenset of ``(x, y, size)``).
            sample_step: the generative model ``G(state, action, rng)`` shared
                with POMCP (typically ``planner.sample_step``).
        """
        real_key = self._key(real_observation)
        n = self.n_particles
        prior = self.particles                       # sample from the current belief
        if not prior:
            return self.recover_from_map(real_observation)

        new_particles = []
        attempts = 0
        max_attempts = n * max_attempt_factor
        while len(new_particles) < n and attempts < max_attempts:
            pos = prior[self.rng.integers(0, len(prior))]
            next_state, obs_key, _r, _t = sample_step((pos, heading, boxes), action, self.rng)
            attempts += 1
            if obs_key == real_key:
                new_particles.append(next_state[0])

        if len(new_particles) == n:
            self.particles = new_particles
        elif new_particles:
            # Partial collection: top up via map-knowledge reinvigoration.
            candidates = self._cells_by_obs.get(real_key) or new_particles
            fill = [candidates[self.rng.integers(0, len(candidates))]
                    for _ in range(n - len(new_particles))]
            self.particles = new_particles + fill
        else:
            # Total depletion -> map-knowledge recovery.
            self.particles = self.recover_from_map(real_observation)
        return self.particles

    def recover_from_map(self, real_observation):
        """
        Deterministic fallback used when the particle cloud is empty. Returns
        the freshly repopulated particle list (empty only if the observation is
        inconsistent with every cell on the known map).
        """
        real_key = self._key(real_observation)
        candidates = self._cells_by_obs.get(real_key, [])
        if candidates:
            self.particles = self._resample(candidates)
        else:
            # Observation matches nowhere on the known map -> no valid belief.
            self.particles = []
        return self.particles

    def _resample(self, population):
        """Sample N particles (with replacement) from a non-empty population."""
        idx = self.rng.integers(0, len(population), size=self.n_particles)
        return [population[i] for i in idx]

    # ------------------------------------------------------------------
    # Belief queries
    # ------------------------------------------------------------------
    def belief_counts(self):
        """Counter mapping each hypothesized position to its particle count."""
        return Counter(self.particles)

    def distinct_positions(self):
        """The set of distinct positions currently hypothesized."""
        return set(self.particles)

    def estimate(self):
        """Maximum-a-posteriori position estimate (most common particle)."""
        if not self.particles:
            return None
        return self.belief_counts().most_common(1)[0][0]

    def __len__(self):
        return len(self.particles)
