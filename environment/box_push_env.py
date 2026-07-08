from minigrid.core.grid import Grid
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.world_object import Wall, Goal
from environment.objects import SmallBox, HeavyBox
import numpy as np

class BoxPushEnv(MiniGridEnv):
    """
    Base Box Pushing environment that generates grids from text-based maps.
    Handles standard agent movement and basic SmallBox pushing logic.
    """
    def __init__(self, ascii_map=None, max_steps=100, **kwargs):
        if ascii_map is None:
            self.ascii_map = [
                "WWWWW",
                "W A W",
                "W B W",
                "WWWWW"
            ]
        else:
            self.ascii_map = ascii_map
            
        width = len(self.ascii_map[0])
        height = len(self.ascii_map)
        
        self.agent_start_positions = []
        
        from minigrid.core.mission import MissionSpace
        mission_space = MissionSpace(mission_func=lambda: "Push the boxes to the goal.")
        
        super().__init__(
            mission_space=mission_space,
            max_steps=max_steps,
            width=width,
            height=height,
            **kwargs
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.agent_start_positions = []
        # Store goal positions from the ascii_map so they persist
        self.goal_positions = []
        
        for y, row in enumerate(self.ascii_map):
            for x, char in enumerate(row):
                if char == 'W':
                    self.grid.set(x, y, Wall())
                elif char == 'G':
                    self.grid.set(x, y, Goal())
                    self.goal_positions.append((x, y))
                elif char == 'B':
                    self.grid.set(x, y, SmallBox())
                elif char == 'C':
                    self.grid.set(x, y, HeavyBox())
                elif char == 'A':
                    self.agent_start_positions.append((x, y))
                    
        # Set agent initial position to first A if exists
        if len(self.agent_start_positions) > 0:
            self.agent_pos = self.agent_start_positions[0]
            # 0=right, 1=down, 2=left, 3=up
            self.agent_dir = 1  
        else:
            self.agent_pos = (1, 1)
            self.agent_dir = 0
            
        self.mission = "Push the boxes to the goal."

    def _all_boxes_on_goals(self):
        """
        Return True iff every goal cell is occupied by a box (SmallBox or HeavyBox).
        """
        for gx, gy in self.goal_positions:
            cell = self.grid.get(gx, gy)
            if cell is None or not hasattr(cell, 'box_size'):
                return False
        return len(self.goal_positions) > 0

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed)
        # Resync agent position in case super().reset() mangled it
        if len(self.agent_start_positions) > 0:
            self.agent_pos = self.agent_start_positions[0]
            self.agent_dir = 1 
        return obs, info
        
    def step(self, action):
        self.step_count += 1
        reward = 0
        terminated = False
        truncated = False

        # Get the position in front of the agent
        fwd_pos = self.front_pos
        fwd_cell = self.grid.get(*fwd_pos)

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell is None or fwd_cell.can_overlap():
                self.agent_pos = tuple(fwd_pos)
                    
            elif fwd_cell is not None and fwd_cell.type == "box" and getattr(fwd_cell, "box_size", "") == "small":
                # Check what is behind the box
                fwd_fwd_pos = fwd_pos + self.dir_vec
                fwd_fwd_cell = self.grid.get(*fwd_fwd_pos)
                
                # If space behind box is empty or goal, move it
                if fwd_fwd_cell is None or fwd_fwd_cell.can_overlap():
                    # Move the box
                    self.grid.set(*fwd_fwd_pos, fwd_cell)
                    self.grid.set(*fwd_pos, None)
                    # Move the agent into the space the box was occupying
                    self.agent_pos = tuple(fwd_pos)

        # Check if all boxes are on goal positions
        if self._all_boxes_on_goals():
            terminated = True
            reward = self._reward()
                    
        if self.step_count >= self.max_steps:
            truncated = True

        obs = self.gen_obs()
        info = {}
        return obs, reward, terminated, truncated, info
