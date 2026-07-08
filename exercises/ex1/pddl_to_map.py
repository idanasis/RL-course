import re
import os

def generate_ascii_map(pddl_file_path):
    # Make sure the file exists
    if not os.path.exists(pddl_file_path):
        print(f"Error: Could not find {pddl_file_path}")
        return

    with open(pddl_file_path, 'r') as file:
        content = file.read()

    # Initialize a 4x4 grid with empty spaces (inner playable area)
    grid = [[' ' for _ in range(4)] for _ in range(4)]

    # Split the file into init and goal sections
    if '(:init' not in content:
        return "Invalid PDDL: No (:init) block found."
    
    init_block = content.split('(:init')[1].split('(:goal')[0]
    
    # 1. Find Goals (G)
    if '(:goal' in content:
        goal_block = content.split('(:goal')[1]
        for match in re.finditer(r'loc_(\d+)_(\d+)', goal_block):
            col, row = int(match.group(1)), int(match.group(2))
            grid[row-1][col-1] = 'G'

    # 2. Find Agents (A)
    for match in re.finditer(r'\(agent-at \w+ loc_(\d+)_(\d+)\)', init_block):
        col, row = int(match.group(1)), int(match.group(2))
        grid[row-1][col-1] = 'A'

    # 3. Find Small Boxes (B)
    for match in re.finditer(r'\(box-at \w+ loc_(\d+)_(\d+)\)', init_block):
        col, row = int(match.group(1)), int(match.group(2))
        grid[row-1][col-1] = 'B'

    # 4. Find Heavy Box (C)
    for match in re.finditer(r'\(heavybox-at \w+ loc_(\d+)_(\d+)\)', init_block):
        col, row = int(match.group(1)), int(match.group(2))
        grid[row-1][col-1] = 'C'

    # --- Build the final map with walls ---
    final_map = []
    final_map.append("WWWWWW") # Top Wall
    
    for row in grid:
        final_map.append("W" + "".join(row) + "W") # Left Wall + row + Right Wall
        
    final_map.append("WWWWWW") # Bottom Wall

    # --- Save to file without the variable name ---
    output_str = "[\n"
    for i, line in enumerate(final_map):
        if i < len(final_map) - 1:
            output_str += f'    "{line}",\n'
        else:
            output_str += f'    "{line}"\n'
    output_str += "]"

    output_filename = "ascii_map.txt"
    with open(output_filename, "w") as out:
        out.write(output_str)
        
    print(f"✅ Map successfully saved to {output_filename}!")


if __name__ == "__main__":
    # Point this to wherever your problem.pddl is currently saved
    pddl_path = "exercises/ex1/pddl/problem.pddl" 
    generate_ascii_map(pddl_path)