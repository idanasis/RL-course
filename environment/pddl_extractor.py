import os

def generate_domain(domain_path):
    # YOUR exact PDDL domain logic with directional adjacencies
    domain_str = """(define (domain box-push)
  (:requirements :typing :equality)
  (:types agent box heavybox location direction)
  
  (:predicates
    (agent-at ?a - agent ?l - location)
    (box-at ?b - box ?l - location)
    (heavybox-at ?h - heavybox ?l - location)
    (clear ?l - location)
    (adj ?l1 - location ?l2 - location ?d - direction)
  )

  (:action move
    :parameters (?a - agent ?from - location ?to - location ?d - direction)
    :precondition (and (agent-at ?a ?from) (clear ?to) (adj ?from ?to ?d))
    :effect (and 
      (not (agent-at ?a ?from)) 
      (agent-at ?a ?to)
    )
  )

  (:action push-small
    :parameters (?a - agent ?pos - location ?boxpos - location ?newboxpos - location ?d - direction ?b - box)
    :precondition (and 
      (agent-at ?a ?pos) 
      (box-at ?b ?boxpos) 
      (clear ?newboxpos)
      (adj ?pos ?boxpos ?d) 
      (adj ?boxpos ?newboxpos ?d) 
    )
    :effect (and 
      (not (agent-at ?a ?pos)) 
      (agent-at ?a ?boxpos)
      (not (box-at ?b ?boxpos)) 
      (box-at ?b ?newboxpos)
      (clear ?boxpos)
      (not (clear ?newboxpos))
    )
  )

  (:action push-heavy
    :parameters (?a1 - agent ?a2 - agent ?pos - location ?boxpos - location ?newboxpos - location ?d - direction ?h - heavybox)
    :precondition (and 
      (not (= ?a1 ?a2))
      (agent-at ?a1 ?pos) 
      (agent-at ?a2 ?pos) 
      (heavybox-at ?h ?boxpos) 
      (clear ?newboxpos)
      (adj ?pos ?boxpos ?d) 
      (adj ?boxpos ?newboxpos ?d) 
    )
    :effect (and 
      (not (agent-at ?a1 ?pos)) 
      (not (agent-at ?a2 ?pos)) 
      (agent-at ?a1 ?boxpos)
      (agent-at ?a2 ?boxpos)
      (not (heavybox-at ?h ?boxpos)) 
      (heavybox-at ?h ?newboxpos)
      (clear ?boxpos)
      (not (clear ?newboxpos))
    )
  )
)
"""
    with open(domain_path, "w") as f:
        f.write(domain_str)

def generate_problem(env, problem_path):
    """
    Generate a PDDL problem file from the current environment state.
    
    IMPORTANT: Goals are read from env.goal_positions (set once at init from
    the ascii_map) — NOT from the live grid. When a box is pushed onto a goal
    cell the Goal object is overwritten, so scanning the grid would lose goals.
    Only the :init section (current state) changes between calls; the :goal
    and :objects remain stable.
    """
    w, h = env.width, env.height
    
    locations = []
    adjacencies = []
    agents = env.agents
    boxes = []
    heavyboxes = []

    # ── Scan the grid for topology and movable objects ────────────────
    for y in range(h):
        for x in range(w):
            cell = env.core_env.grid.get(x, y)
            # A cell is walkable if it's empty, a goal, a box, or an agent
            # — basically anything that is NOT a wall.
            is_wall = (cell is not None and cell.type == 'wall')
            if not is_wall:
                loc = f"loc_{x}_{y}"
                locations.append(loc)

                # Directional adjacencies
                if x < w - 1:
                    r_cell = env.core_env.grid.get(x+1, y)
                    if r_cell is None or r_cell.type != 'wall':
                        adjacencies.append((loc, f"loc_{x+1}_{y}", "right"))
                        adjacencies.append((f"loc_{x+1}_{y}", loc, "left"))
                if y < h - 1:
                    d_cell = env.core_env.grid.get(x, y+1)
                    if d_cell is None or d_cell.type != 'wall':
                        adjacencies.append((loc, f"loc_{x}_{y+1}", "down"))
                        adjacencies.append((f"loc_{x}_{y+1}", loc, "up"))

                # Collect boxes from the grid (their CURRENT positions)
                if cell is not None and cell.type == "box":
                    if getattr(cell, "box_size", "") == "heavy":
                        heavyboxes_coords.append((x, y))
                    else:
                        boxes.append((f"box_{len(boxes)}", loc))

    # ── Use stored goal positions (stable, from ascii_map) ────────────
    # These never change even when a box sits on top of a goal cell.
    goals = [f"loc_{gx}_{gy}" for gx, gy in env.goal_positions]

    # ── Agent positions ───────────────────────────────────────────────
    agent_locs = []
    for a in agents:
        px, py = env.agent_positions[a]
        agent_locs.append((a, f"loc_{px}_{py}"))
            
    # ── Clear set: locations not occupied by an agent or box ──────────
    clear_set = set(locations)
    for _, loc in boxes:
        clear_set.discard(loc)
    for _, loc in heavyboxes:
        clear_set.discard(loc)

    # ── Build :objects ────────────────────────────────────────────────
    obj_str = "    " + " ".join(locations) + " - location\n"
    if agents:
        obj_str += "    " + " ".join(agents) + " - agent\n"
    if boxes:
        obj_str += "    " + " ".join([b[0] for b in boxes]) + " - box\n"
    if heavyboxes:
        obj_str += "    " + " ".join([b[0] for b in heavyboxes]) + " - heavybox\n"

    # ── Build :init ──────────────────────────────────────────────────
    init_str = ""
    for loc in clear_set:
        init_str += f"    (clear {loc})\n"
    for a, loc in agent_locs:
        init_str += f"    (agent-at {a} {loc})\n"
    for b, loc in boxes:
        init_str += f"    (box-at {b} {loc})\n"
    for h, loc in heavyboxes:
        init_str += f"    (heavybox-at {h} {loc})\n"
    for l1, l2, d in adjacencies:
        init_str += f"    (adj {l1} {l2} {d})\n"

    # ── Build :goal ──────────────────────────────────────────────────
    # Pair boxes then heavyboxes with goal locations (scan order)
    goal_conditions = []
    for i in range(len(boxes_coords)):
        gx, gy = assigned_boxes[i]
        goal_conditions.append(f"(box-at box_{i} loc_{gx}_{gy})")
        
    for i in range(len(heavyboxes_coords)):
        # Safely assign heavy goals
        if i < len(heavy_goals):
            gx, gy = heavy_goals[i]
            goal_conditions.append(f"(heavybox-at hbx_{i} loc_{gx}_{gy})")

    if goal_conditions:
        goal_str = "(and\n" + "\n".join(f"    {g}" for g in goal_conditions) + "\n  )"
    else:
        goal_str = "(and)"

    problem_str = f"""(define (problem bp-map)
  (:domain box-push)
  (:objects
{obj_str}  )
  (:init
{init_str}  )
  (:goal
    {goal_str}
  )
)
"""
    with open(problem_path, "w") as f:
        f.write(problem_str)


def generate_pddl_for_env(env, pddl_folder="pddl"):
    os.makedirs(pddl_folder, exist_ok=True)
    domain_path = os.path.join(pddl_folder, "domain.pddl")
    problem_path = os.path.join(pddl_folder, "problem.pddl")
    
    generate_domain(domain_path)
    generate_problem(env, problem_path)
    
    return domain_path, problem_path