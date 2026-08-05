"""
Egocentric 3x3 observation extraction for the Partially Observable Box Pushing env.

The agent's own position is hidden from it (POMDP), but the map layout and box
positions are known/static per episode. Instead of relying on Minigrid's
forward-facing `gen_obs` (which depends on the agent's `dir` heading), we read
directly from `env.core_env.grid` and slice out a fixed, translation-based
window: row -1 is always the world's North neighbor, row +1 is always South,
col -1 is always West, col +1 is always East -- regardless of which way the
agent is currently facing.

The agent's true (x, y) position is only ever used internally (by the
simulator/this function) to know *where* to center the slice; it is never
included in the returned observation itself.
"""

import numpy as np
from minigrid.core.world_object import Wall, Goal

# ---------------------------------------------------------------------------
# Cell encoding
# ---------------------------------------------------------------------------
EMPTY = 0
WALL = 1
SMALL_BOX = 2
HEAVY_BOX = 3
GOAL = 4

CELL_NAMES = {
    EMPTY: "empty",
    WALL: "wall",
    SMALL_BOX: "small_box",
    HEAVY_BOX: "heavy_box",
    GOAL: "goal",
}


def encode_cell(cell):
    """Map a single minigrid grid cell (WorldObj or None) to an integer code."""
    if cell is None:
        return EMPTY

    box_size = getattr(cell, "box_size", None)
    if box_size == "small":
        return SMALL_BOX
    if box_size == "heavy":
        return HEAVY_BOX

    if isinstance(cell, Wall):
        return WALL
    if isinstance(cell, Goal):
        return GOAL

    # Anything else (e.g. another agent's sprite) is treated as open/empty
    # floor for observation purposes -- it isn't part of the static map.
    return EMPTY


# ---------------------------------------------------------------------------
# Core, grid-agnostic extraction (easy to unit test without a full env)
# ---------------------------------------------------------------------------
def extract_egocentric_window(grid, center_pos, grid_width, grid_height, window_size=3):
    """
    Slice a `window_size x window_size` translation-based window out of `grid`,
    centered on `center_pos`. Out-of-bounds cells are padded as walls.

    Args:
        grid: a minigrid Grid instance exposing `.get(x, y)`.
        center_pos: (x, y) true position of the agent.
        grid_width, grid_height: dimensions of the full grid.
        window_size: must be odd; the agent is always at the exact center.

    Returns:
        np.ndarray of shape (window_size, window_size), dtype=int8, where
        row index increases South (+y) and column index increases East (+x).
        obs[half, half] is always the agent's own cell.
    """
    assert window_size % 2 == 1, "window_size must be odd so the agent sits at the center"
    half = window_size // 2
    ax, ay = center_pos

    obs = np.full((window_size, window_size), WALL, dtype=np.int8)
    for dy in range(-half, half + 1):
        for dx in range(-half, half + 1):
            gx, gy = ax + dx, ay + dy
            row, col = dy + half, dx + half
            if 0 <= gx < grid_width and 0 <= gy < grid_height:
                obs[row, col] = encode_cell(grid.get(gx, gy))
            # else: leave as WALL (out-of-bounds padding)
    return obs


# ---------------------------------------------------------------------------
# Env-facing helper
# ---------------------------------------------------------------------------
def get_agent_observation(env, agent, window_size=3):
    """
    Build the egocentric 3x3 observation for `agent` from `env.core_env.grid`.

    `env` is a MultiAgentBoxPushEnv / StochasticMultiAgentBoxPushEnv instance.
    The agent's own sprite is temporarily removed from the grid (if present)
    so it doesn't shadow whatever is actually beneath it.
    """
    core_env = env.core_env
    pos = env.agent_positions[agent]
    agent_obj = env.agent_objects[agent]

    occupies_own_cell = core_env.grid.get(*pos) is agent_obj
    if occupies_own_cell:
        core_env.grid.set(pos[0], pos[1], None)
    try:
        obs = extract_egocentric_window(
            core_env.grid, pos, env.width, env.height, window_size=window_size
        )
    finally:
        if occupies_own_cell:
            core_env.grid.set(pos[0], pos[1], agent_obj)
    return obs


# ---------------------------------------------------------------------------
# Optional environment wrapper
# ---------------------------------------------------------------------------
class EgocentricGridObservationWrapper:
    """
    Thin PettingZoo ParallelEnv wrapper that replaces the wrapped env's
    default (forward-facing, `gen_obs`-based) observations with the
    translation-based 3x3 egocentric window defined above.
    """

    def __init__(self, env, window_size=3):
        self.env = env
        self.window_size = window_size

    def __getattr__(self, name):
        # Delegate anything not explicitly overridden (agents, agent_positions, ...)
        return getattr(self.env, name)

    def _build_observations(self, agents):
        return {a: get_agent_observation(self.env, a, self.window_size) for a in agents}

    def reset(self, *args, **kwargs):
        _, infos = self.env.reset(*args, **kwargs)
        return self._build_observations(self.env.agents), infos

    def step(self, actions):
        _, rewards, terminations, truncations, infos = self.env.step(actions)
        acted_agents = [a for a in actions if a in self.env.possible_agents]
        observations = self._build_observations(acted_agents)
        return observations, rewards, terminations, truncations, infos


# ---------------------------------------------------------------------------
# Test block
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from environment import MultiAgentBoxPushEnv

    # 6x5 map: border walls, one small box (B), one heavy box (C), one goal (G).
    ascii_map = [
        "WWWWWW",
        "W A  W",
        "W B  W",
        "WC  GW",
        "WWWWWW",
    ]

    env = MultiAgentBoxPushEnv(ascii_map=ascii_map)
    env.reset()
    agent = env.possible_agents[0]

    def show(pos, label):
        env.agent_positions[agent] = pos
        obs = get_agent_observation(env, agent)
        print(f"\n{label} -- true pos={pos}")
        for row in obs:
            print(" ".join(CELL_NAMES[c][:5].ljust(5) for c in row))
        return obs

    # 1) Agent in open floor, box just South of it.
    obs_mid = show((2, 1), "Center: open floor, box below")
    assert obs_mid[1, 1] == EMPTY  # agent's own cell reads as empty floor
    assert obs_mid[2, 1] == SMALL_BOX  # South neighbor is the small box

    # 2) Move the agent one cell South: the window must shift with it,
    #    proving the slice is recomputed from the *true* position each time,
    #    not cached from the previous call.
    obs_after_move = show((2, 2), "Moved South: box now under agent")
    assert obs_after_move[1, 1] == SMALL_BOX
    assert not np.array_equal(obs_mid, obs_after_move)

    # 3) Corner case: top-left playable cell (1,1) is adjacent to the map
    #    border. North and West neighbors fall outside the grid and must be
    #    padded as walls, not crash or wrap around.
    obs_corner = show((1, 1), "Corner: North/West padded as walls")
    assert obs_corner[0, 1] == WALL  # North neighbor -> out of bounds
    assert obs_corner[1, 0] == WALL  # West neighbor -> out of bounds
    assert obs_corner[1, 1] == EMPTY  # agent's own cell is open floor
    assert obs_corner[2, 1] == EMPTY  # (1,2) is open floor in the ascii map

    # 4) Heavy box and goal are correctly distinguished elsewhere on the map.
    obs_heavy = show((1, 3), "On the heavy box row")
    assert obs_heavy[1, 1] == HEAVY_BOX

    obs_goal = show((3, 3), "Next to the goal")
    assert obs_goal[1, 2] == GOAL  # East neighbor (4,3) is the goal cell

    print("\nAll egocentric observation tests passed.")
