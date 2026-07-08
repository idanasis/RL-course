import pygame
import time
import re
from environment.multi_agent_env import MultiAgentBoxPushEnv
from planner.pddl_solver import solve_pddl
from minigrid.core.constants import DIR_TO_VEC

def extract_target_pos(pddl_action):
    """
    Returns a dictionary mapping agent names to their target coordinates.
    Handles single-agent actions (move, push-small) and dual-agent (push-big).
    """
    action_str = str(pddl_action)
    action_name = action_str.split('(')[0]
    params = re.findall(r'[\w_]+', action_str[action_str.find("(")+1:action_str.find(")")])
    
    agent_targets = {}
    
    if action_name.startswith('win'):
        return {}
    elif action_name.startswith('push-heavy') and len(params) >= 5:
        # push-heavy(?a1, ?a2, ?from, ?boxloc, ?toloc, ?h)
        a1, a2 = params[0], params[1]
        target_loc = params[3] # boxloc is the target for both pushing agents
        
        parts = target_loc.split('_')
        if len(parts) == 3:
            tgt = (int(parts[1]), int(parts[2]))
            agent_targets[a1] = tgt
            agent_targets[a2] = tgt
    elif len(params) >= 3:
        # move or push-small
        agent_name = params[0]
        target_loc = params[2] 
        parts = target_loc.split('_')
        if len(parts) == 3:
            agent_targets[agent_name] = (int(parts[1]), int(parts[2]))
            
    return agent_targets

def get_required_actions(env, agent, target_pos):
    """
    Returns a list of PettingZoo actions (0=left, 1=right, 2=forward) 
    needed to move the agent into the target adjacent position.
    """
    current_pos = env.agent_positions[agent]
    current_dir = env.agent_dirs[agent]
    
    dx = target_pos[0] - current_pos[0]
    dy = target_pos[1] - current_pos[1]
    
    target_dir = None
    for d, vec in enumerate(DIR_TO_VEC):
        if vec[0] == dx and vec[1] == dy:
            target_dir = d
            break
            
    if target_dir is None:
        raise ValueError(f"Target {target_pos} is not adjacent to {current_pos}")
        
    actions = []
    while current_dir != target_dir:
        actions.append(1) # 1 = turn right
        current_dir = (current_dir + 1) % 4
    actions.append(2) # 2 = forward
    
    return actions

def visualize_pddl_plan(ascii_map, domain_file, problem_file):
    print("🧠 Solving PDDL problem...")
    plan = solve_pddl(domain_file, problem_file)
    
    if not plan:
        print("❌ No plan could be found. Cannot visualize.")
        return
        
    print(f"✅ Plan found with {len(plan.actions)} steps. Booting visualizer...")
    
    env = MultiAgentBoxPushEnv(ascii_map=ascii_map, render_mode="human")
    env.reset()
    env.core_env.render()
    
    time.sleep(1) 
    
    for pddl_action in plan.actions:
        print(f"Executing: {pddl_action}")
        
        agent_targets = extract_target_pos(pddl_action)
        
        # We need to execute the rotations for all involved agents first
        # then execute their forward movement simultaneously.
        agents = list(agent_targets.keys())
        agent_action_queues = {a: get_required_actions(env, a, agent_targets[a]) for a in agents}
        
        # Pad queues with None so all agents execute their final 'forward' simultaneously
        if agent_action_queues:
            max_len = max(len(q) for q in agent_action_queues.values())
            for a in agents:
                agent_action_queues[a] = [None] * (max_len - len(agent_action_queues[a])) + agent_action_queues[a]
        
        # Iterate until all agents have exhausted their rotation/forward actions
        while any(len(q) > 0 for q in agent_action_queues.values()):
            step_actions = {}
            for a in agents:
                if len(agent_action_queues[a]) > 0:
                    act = agent_action_queues[a].pop(0)
                    if act is not None:
                        step_actions[a] = act
            
            env.step(step_actions)
            env.core_env.render()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                    
            time.sleep(0.4) 
                
    print("🎉 Plan execution complete! Closing in 3 seconds.")
    time.sleep(3)
    pygame.quit()

if __name__ == "__main__":
    from environment.pddl_extractor import generate_pddl_for_env
    # We test visualizer on a large map with a joint BigBox push
    new_map = [
    "WWWWWW",
    "WA   W",
    "WBBACW",
    "W    W",
    "WGG GW",
    "WWWWWW"
]
    
    print("🌍 Generating Environment and extracting PDDL...")
    # env_sim = MultiAgentBoxPushEnv(ascii_map=new_map, render_mode="rgb_array")
    # env_sim.reset()
    # generate_pddl_for_env(env_sim, "exercises\ex1\pddl")
    
    print("🚀 Running visualization pipeline...")
    visualize_pddl_plan(new_map, "exercises\ex1\pddl\domain.pddl", "exercises\ex1\pddl\problem.pddl")
