"""
Particle Filter for localizing the agent in the Partially Observable Box
Pushing environment.

The agent does not know its own (x, y) position, but the grid size and the
static wall / box / goal layout ARE known. Observations are the deterministic
3x3 egocentric windows produced by ``observation_function.py``. This filter
maintains a belief over the agent's position as a cloud of particles, each
particle being a single ``(x, y)`` coordinate hypothesis.

Because both the transition model and the observation model are treated as
deterministic, this is essentially *rejection sampling*: a particle survives an
update only if the observation it would generate exactly matches the real
observation. That makes the belief collapse quickly, but also means the cloud
can be wiped out entirely -- so a deterministic "map knowledge" fallback
re-derives the full set of consistent positions directly from the known map.

Localization model
------------------
Consistent with the translation-based (heading-agnostic) observation, the
filter treats actions as cardinal moves. An action is one of the strings
``"north" / "south" / "east" / "west" / "stay"`` (or an explicit ``(dx, dy)``
vector, or ``None`` for stay). A particle moves one cell in the action's
direction if that cell is free (not a wall/box); otherwise it stays put. Env
rotations / no-ops map to ``"stay"``. The transition model is pluggable via the
``transition_fn`` constructor argument if different dynamics are needed.

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


# Cardinal action -> (dx, dy). y grows downward (South), matching the grid.
ACTION_VECTORS = {
    "north": (0, -1),
    "south": (0, 1),
    "east": (1, 0),
    "west": (-1, 0),
    "stay": (0, 0),
}


class ParticleFilter:
    """Rejection-sampling particle filter over the agent's ``(x, y)`` position."""

    def __init__(
        self,
        ascii_map,
        n_particles=500,
        window_size=3,
        transition_fn=None,
        seed=None,
    ):
        self.grid, self.width, self.height = build_static_grid(ascii_map)
        self.n_particles = n_particles
        self.window_size = window_size
        self.rng = np.random.default_rng(seed)
        self.transition_fn = transition_fn or self._default_transition

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
    # Transition model
    # ------------------------------------------------------------------
    @staticmethod
    def _action_vector(action):
        if action is None:
            return (0, 0)
        if isinstance(action, str):
            return ACTION_VECTORS[action.lower()]
        # Assume an explicit (dx, dy) vector.
        dx, dy = action
        return (int(dx), int(dy))

    def _default_transition(self, pos, action):
        """Deterministic cardinal move; blocked moves leave the particle put."""
        dx, dy = self._action_vector(action)
        if dx == 0 and dy == 0:
            return pos
        nx, ny = pos[0] + dx, pos[1] + dy
        if self._is_free(nx, ny):
            return (nx, ny)
        return pos

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

    def update(self, action, real_observation):
        """
        Advance the belief by one step.

        1. Move every particle through the known transition model.
        2. Keep a moved particle only if the observation it would generate
           exactly matches ``real_observation`` (rejection sampling); resample
           the survivors back up to ``N`` to keep the cloud size constant.
        3. If *no* particle survives, fall back to full map knowledge: scan
           every valid cell, collect those whose egocentric slice matches the
           real observation, and repopulate ``N`` particles from that set.
        """
        real_key = self._key(real_observation)

        survivors = []
        for p in self.particles:
            moved = self.transition_fn(p, action)
            if self._key_by_cell.get(moved, self._key(self.observation_at(moved))) == real_key:
                survivors.append(moved)

        if survivors:
            self.particles = self._resample(survivors)
            return self.particles

        # ── Failsafe: map-knowledge recovery ──────────────────────────
        return self.recover_from_map(real_observation)

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


# ---------------------------------------------------------------------------
# Test block
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # ── Section A: filter mechanics on a known, asymmetric map ────────────
    ascii_map = [
        "WWWWWWW",
        "W     W",
        "W WWW W",
        "W   W W",
        "WGW B W",
        "W  C  W",
        "WWWWWWW",
    ]

    N = 500
    pf = ParticleFilter(ascii_map, n_particles=N, seed=0)

    # Init: exactly N particles, all on free cells.
    assert len(pf) == N
    assert all(p in set(pf.free_cells) for p in pf.particles)
    print(f"Initialized {len(pf)} particles over {len(pf.free_cells)} free cells.")

    # Drive a known ground-truth trajectory and feed the filter the exact
    # observations that trajectory produces. The true position must remain in
    # the belief at every step, and the belief should collapse as we move.
    true_pos = (1, 1)
    path = ["south", "south", "east", "east", "north", "stay"]

    real_obs = pf.observation_at(true_pos)
    pf.update("stay", real_obs)  # first observation, no motion
    assert true_pos in pf.distinct_positions()
    print(f"\nStep 0  action=stay   true={true_pos}  "
          f"belief_size={len(pf.distinct_positions())}  est={pf.estimate()}")

    for i, action in enumerate(path, start=1):
        true_pos = pf._default_transition(true_pos, action)
        real_obs = pf.observation_at(true_pos)
        pf.update(action, real_obs)
        assert true_pos in pf.distinct_positions(), (
            f"true position {true_pos} dropped from belief after action '{action}'"
        )
        assert len(pf) == N  # cloud size is maintained
        print(f"Step {i}  action={action:5s} true={true_pos}  "
              f"belief_size={len(pf.distinct_positions())}  est={pf.estimate()}")

    # After an informative trajectory the belief should have collapsed well
    # below the number of free cells.
    assert len(pf.distinct_positions()) < len(pf.free_cells)

    # ── Section B: failsafe when the cloud is wiped out ───────────────────
    # Force a catastrophic loss (0 particles) and confirm the deterministic
    # map-knowledge recovery repopulates a valid, true-position-containing set.
    target = (4, 5)  # some free cell
    real_obs = pf.observation_at(target)
    pf.particles = []  # simulate total depletion
    pf.update("stay", real_obs)
    assert len(pf) == N, "failsafe did not repopulate N particles"
    assert target in pf.distinct_positions()
    # Every repopulated particle must be genuinely consistent with the map.
    assert all(
        np.array_equal(pf.observation_at(p), real_obs) for p in pf.distinct_positions()
    )
    print(f"\nFailsafe: recovered {len(pf)} particles across "
          f"{len(pf.distinct_positions())} candidate cell(s) consistent with obs.")

    # Failsafe also fires when particles exist but none can match the obs.
    pf.particles = [(1, 1)] * N            # all wrong for `target`'s observation
    pf.update("stay", pf.observation_at(target))
    assert target in pf.distinct_positions()
    print("Failsafe: also triggered correctly from a fully-mismatched cloud.")

    # Observation inconsistent with the whole map -> empty belief (no crash).
    bogus = np.full((3, 3), 9, dtype=np.int8)
    pf.recover_from_map(bogus)
    assert len(pf) == 0
    print("Failsafe: impossible observation yields an empty belief, no crash.")

    # ── Section C: integration with the real env + observation function ───
    from environment import MultiAgentBoxPushEnv
    try:
        from .observation_function import get_agent_observation
    except ImportError:
        from observation_function import get_agent_observation

    env_map = [
        "WWWWWW",
        "W A  W",
        "W B  W",
        "WC  GW",
        "WWWWWW",
    ]
    env = MultiAgentBoxPushEnv(ascii_map=env_map)
    env.reset()
    agent = env.possible_agents[0]

    pf2 = ParticleFilter(env_map, n_particles=200, seed=1)
    true = env.agent_positions[agent]

    # The filter's hypothetical observation for the true cell must match the
    # observation the env's custom observation function actually produces.
    real = get_agent_observation(env, agent)
    assert np.array_equal(pf2.observation_at(true), real), (
        "filter observation model disagrees with the env observation function"
    )

    pf2.update("stay", real)
    assert true in pf2.distinct_positions()
    print(f"\nIntegration: env agent truly at {true}; filter belief narrowed to "
          f"{sorted(pf2.distinct_positions())}, estimate={pf2.estimate()}.")

    print("\nAll particle filter tests passed.")
