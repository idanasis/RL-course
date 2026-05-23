import math
import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Wall, Goal, WorldObj
from minigrid.utils.rendering import fill_coords, point_in_circle, point_in_triangle, rotate_fn
from minigrid.core.constants import COLORS
from minigrid.minigrid_env import MiniGridEnv
from environment.objects import SmallBox, HeavyBox

class AgentObj(WorldObj):
    """
    Representation of an agent inside the MiniGrid so it can be seen by other agents.
    """
    def __init__(self, color="red", dir=0):
        super().__init__("agent", color)
        self.dir = dir

    def can_overlap(self):
        return False

    def render(self, img):
        c = COLORS.get(self.color, (255, 0, 0))
        tri_fn = point_in_triangle(
            (0.12, 0.19),
            (0.87, 0.50),
            (0.12, 0.81),
        )
        tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * self.dir)
        fill_coords(img, tri_fn, c)

class MultiAgentBoxPushEnv(ParallelEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "multi_agent_box_push_v0"
    }

    def __init__(self, ascii_map=None, max_steps=100, render_mode=None):
        if ascii_map is None:
            self.ascii_map = [
                "WWWWW",
                "W A W",
                "W B W",
                "W A W",
                "WWWWW"
            ]
        else:
            self.ascii_map = ascii_map
            
        self.width = len(self.ascii_map[0])
        self.height = len(self.ascii_map)
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.steps = 0
        
        self.possible_agents = []
        for row in self.ascii_map:
            for char in row:
                if char == 'A':
                    self.possible_agents.append(f"agent_{len(self.possible_agents)}")
                    
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        
        mission_space = MissionSpace(mission_func=lambda: "Collaborate to push the boxes to the goal.")
        self.core_env = MiniGridEnv(
            mission_space=mission_space,
            width=self.width,
            height=self.height,
            max_steps=self.max_steps,
            render_mode=render_mode
        )
        self.core_env._gen_grid = self._gen_grid
        
        # Override get_frame to decouple agent rendering from grid logic
        def custom_get_frame(*args, **kwargs):
            tile_size = kwargs.get('tile_size', 32)
            if len(args) > 1:
                tile_size = args[1]
                
            img = self.core_env.grid.render(tile_size, agent_pos=(-1, -1), agent_dir=0, highlight_mask=None)
            
            for agent in self.possible_agents:
                if agent in self.agent_positions:
                    pos = self.agent_positions[agent]
                    ag_obj = self.agent_objects[agent]
                    
                    ymin = pos[1] * tile_size
                    ymax = (pos[1] + 1) * tile_size
                    xmin = pos[0] * tile_size
                    xmax = (pos[0] + 1) * tile_size
                    
                    tile_img = img[ymin:ymax, xmin:xmax, :]
                    ag_obj.render(tile_img)
                    
            return img
            
        self.core_env.get_frame = custom_get_frame
        
    def observation_space(self, agent):
        return self.core_env.observation_space
        
    def action_space(self, agent):
        # 0: left, 1: right, 2: forward
        return spaces.Discrete(3)
        
    def _gen_grid(self, width, height):
        self.core_env.grid = Grid(width, height)
        self.core_env.agent_pos = (-1, -1)
        self.core_env.agent_dir = 0
        self.agent_positions = {}
        self.agent_dirs = {}
        self.agent_objects = {}
        # Store goal positions from the ascii_map so they never get lost
        # when a box is pushed onto a goal cell (overwriting the Goal object).
        self.goal_positions = []
        
        agent_idx = 0
        colors = ["green", "red", "blue", "purple"]
        
        for y, row in enumerate(self.ascii_map):
            for x, char in enumerate(row):
                if char == 'W':
                    self.core_env.grid.set(x, y, Wall())
                elif char == 'G':
                    self.core_env.grid.set(x, y, Goal())
                    self.goal_positions.append((x, y))
                elif char == 'B':
                    self.core_env.grid.set(x, y, SmallBox())
                elif char == 'C':
                    self.core_env.grid.set(x, y, HeavyBox())
                elif char == 'A':
                    agent_name = f"agent_{agent_idx}"
                    color = colors[agent_idx % len(colors)]
                    
                    self.agent_positions[agent_name] = (x, y)
                    self.agent_dirs[agent_name] = 1 # down
                    
                    obj = AgentObj(color=color, dir=1)
                    self.agent_objects[agent_name] = obj
                    
                    agent_idx += 1

        # Find an empty cell for the dummy agent position (required by MiniGrid assertions)
        for y, row in enumerate(self.ascii_map):
            for x, char in enumerate(row):
                if char == ' ':
                    self.core_env.agent_pos = (x, y)
                    self.core_env.agent_dir = 0
                    return
        # Fallback if no empty cell found
        self.core_env.agent_pos = (1, 1)
        self.core_env.agent_dir = 0
        
    def _all_boxes_on_goals(self):
        """
        Return True iff every goal cell is occupied by a box (SmallBox or HeavyBox).
        Termination should only happen when ALL goals are satisfied.
        """
        for gx, gy in self.goal_positions:
            cell = self.core_env.grid.get(gx, gy)
            if cell is None or not hasattr(cell, 'box_size'):
                return False
        return len(self.goal_positions) > 0

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.steps = 0
        
        self.core_env.reset(seed=seed)
        
        # Place agents in grid for initial render and correct obs
        for agent in self.agents:
            pos = self.agent_positions[agent]
            self.core_env.grid.set(*pos, self.agent_objects[agent])
            
        observations = {}
        for agent in self.agents:
            pos = self.agent_positions[agent]
            self.core_env.agent_pos = pos
            self.core_env.agent_dir = self.agent_dirs[agent]
            
            # Temporarily hide agent from grid so it doesn't observe itself as an obstacle
            self.core_env.grid.set(*pos, None)
            observations[agent] = self.core_env.gen_obs()
            self.core_env.grid.set(*pos, self.agent_objects[agent])
            
        self.core_env.agent_pos = (-1, -1) # Reset back to dummy invisible space 
        
        return observations, {agent: {} for agent in self.agents}

    def step(self, actions):
        self.steps += 1
        
        observations = {}
        rewards = {agent: 0 for agent in self.agents}
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}

        # Clear agent objects from grid to prevent trailing clones
        for agent in self.agents:
            px, py = self.agent_positions[agent]
            cell = self.core_env.grid.get(px, py)
            if cell is self.agent_objects[agent]:
                self.core_env.grid.set(px, py, None)

        # Pass 1: Gather Intents
        agent_intents = {}
        from minigrid.core.constants import DIR_TO_VEC
        for agent, action in actions.items():
            pos = self.agent_positions[agent]
            d = self.agent_dirs[agent]
            
            if action == 0: # left
                self.agent_dirs[agent] = (d - 1) % 4
                self.agent_objects[agent].dir = self.agent_dirs[agent]
            elif action == 1: # right
                self.agent_dirs[agent] = (d + 1) % 4
                self.agent_objects[agent].dir = self.agent_dirs[agent]
            elif action == 2: # forward
                vec = DIR_TO_VEC[d]
                fwd_pos = (pos[0] + vec[0], pos[1] + vec[1])
                agent_intents[agent] = {"target_pos": fwd_pos, "dir": d, "vec": vec}

        # Pass 2: Heavy Push Resolution
        heavy_box_pushes = {}
        for agent, intent in agent_intents.items():
            fwd_pos = intent["target_pos"]
            fwd_cell = self.core_env.grid.get(*fwd_pos)
            if fwd_cell is not None and getattr(fwd_cell, "box_size", "") == "heavy":
                if fwd_pos not in heavy_box_pushes:
                    heavy_box_pushes[fwd_pos] = []
                heavy_box_pushes[fwd_pos].append((agent, intent))
                    
        for box_pos, pushers in heavy_box_pushes.items():
            if len(pushers) >= 2:
                # Require identical direction and IDENTICAL originating grid space for both pushers
                # Since multiple agents can be on the same HeavyBox cell, we just verify they came from the same pos
                origins = set(self.agent_positions[a] for a, i in pushers)
                dirs = set(i["dir"] for a, i in pushers)
                
                if len(dirs) == 1 and len(origins) == 1:
                    push_dir = list(dirs)[0]
                    vec = DIR_TO_VEC[push_dir]
                    
                    nx, ny = box_pos[0] + vec[0], box_pos[1] + vec[1]
                    n_cell = self.core_env.grid.get(nx, ny)
                    
                    if n_cell is None or n_cell.can_overlap():
                        # Move HeavyBox one cell
                        bx, by = box_pos
                        box_obj = self.core_env.grid.get(bx, by)
                        self.core_env.grid.set(bx, by, None)
                        self.core_env.grid.set(nx, ny, box_obj)
                        
                        # Move all pushing agents
                        for agent, _ in pushers:
                            self.agent_positions[agent] = box_pos
                            
                        # Clear intents for successful pushers so they don't try to move again in Pass 3
                        for agent, _ in pushers:
                            del agent_intents[agent]

        # Pass 3: Process remaining individual forward movements
        for agent, intent in agent_intents.items():
            pos = self.agent_positions[agent]
            fwd_pos = intent["target_pos"]
            vec = intent["vec"]
            fwd_cell = self.core_env.grid.get(*fwd_pos)
            
            if fwd_cell is None or fwd_cell.can_overlap():
                self.agent_positions[agent] = fwd_pos
                        
            elif fwd_cell is not None and getattr(fwd_cell, "box_size", "") == "small":
                fwd_fwd_pos = (fwd_pos[0] + vec[0], fwd_pos[1] + vec[1])
                fwd_fwd_cell = self.core_env.grid.get(*fwd_fwd_pos)
                if fwd_fwd_cell is None or fwd_fwd_cell.can_overlap():
                    self.core_env.grid.set(*fwd_fwd_pos, fwd_cell)
                    self.core_env.grid.set(*fwd_pos, None)
                    self.agent_positions[agent] = fwd_pos

        # Check if all boxes are on goal positions → terminate with reward
        if self._all_boxes_on_goals():
            for a in self.agents:
                rewards[a] = 1.0
                terminations[a] = True

        if self.steps >= self.max_steps:
            for a in self.agents:
                truncations[a] = True
                
        if any(terminations.values()) or any(truncations.values()):
            self.agents = []

        for agent in self.possible_agents:
            if agent in actions:
                pos = self.agent_positions[agent]
                self.core_env.agent_pos = pos
                self.core_env.agent_dir = self.agent_dirs[agent]
                self.core_env.grid.set(*pos, None)
                observations[agent] = self.core_env.gen_obs()
                self.core_env.grid.set(*pos, self.agent_objects[agent])

        return observations, rewards, terminations, truncations, infos
