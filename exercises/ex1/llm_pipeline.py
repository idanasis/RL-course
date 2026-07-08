import os
import subprocess
import sys

# ==============================================================================
# PROMPT HISTORY (Required by Q4)
# ==============================================================================
# The following outlines the iterative prompting strategy used with the LLM 
# to generate the final PDDL files, debug the logic, and refine the mechanics.
#
# Prompt 1 (Domain Generation): 
# "I need to write a PDDL domain file for a multi-agent box-pushing puzzle. 
# There are agents, regular boxes, and a heavy box. Please create the domain.pddl 
# defining the basic movement and the pushing mechanics based on my instructions. verify the physics is right"
# 
# Prompt 2 (Problem Generation): 
# "Now that we have the domain, please generate a corresponding problem.pddl file. 
# Create a grid map, set up an initial state with 2 agents, 2 small boxes, and 
# 1 heavy box, and define a goal state to test the domain."
#
# Prompt 3 (Debugging & Physical Constraints): 
# "I ran the planner and got a 'No plan found' output. Can you help me figure out 
# why the planner cannot find a valid path?" 
# (The LLM successfully identified that the initial grid was physically too small 
# to allow the agents to maneuver around the box, so we expanded the map).
#
# Prompt 4 (Applying Course Q&A Updates): 
# "I received an official Q&A update from the course staff. The heavy box is 
# actually 1x1 in size, not 1x2. To push it, both agents must occupy the exact 
# same cell adjacent to the box. Also, the exact types are 'heavybox' and 'hbx_0'. 
# Please rewrite both the domain and problem files to perfectly match these new 
# authoritative simulator rules."
#
# Prompt 5 (Finalizing Submission Format):
# "What are the exact requirements for the llm_pipeline.py script based on the 
# Q&A, and how should I structure this Python file to meet all the guidelines?"
# ==============================================================================

def run_planner():
    """Runs the planner using the generated files included in the submission."""
    print("Running planner...")
    
    # Paths to the PDDL files (adjust if your folder structure requires it)
    domain_file = "exercises/ex1/pddl/domain.pddl"
    problem_file = "exercises/ex1/pddl/problem.pddl"
    
    # Run the solver script
    subprocess.run([sys.executable, "planner/pddl_solver.py", domain_file, problem_file])

if __name__ == "__main__":
    # We are using Approach #1 from Q4: The files are already included in the repo.
    # Therefore, we just run the planner.
    run_planner()