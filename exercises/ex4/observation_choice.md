# Observation Function Choice and Justification

**Selected Alternative: Alternative B (Egocentric / Centered 3x3 Window)**

For this POMDP assignment, we selected **Alternative B**, where the agent receives a 3x3 observation window perfectly centered on its true location, providing a 1-cell view in all cardinal directions (North, South, East, West, and diagonals). 

## Reasoning

1. **Symmetric Spatial Awareness:** 
   In the Box Pushing environment, the agent frequently needs to navigate tight corridors and maneuver around heavy boxes. A centered window ensures the agent is constantly aware of its immediate surroundings regardless of the direction it is moving. 

2. **Particle Filter Stability (Rejection Sampling):**
   Our Particle Filter relies on exact matches between hypothetical and real observations to update the belief state. If we had chosen Alternative A (a fixed, North-facing window), any movement to the South, East, or West would effectively be a "blind" step. Taking blind steps drastically increases the likelihood of the agent bumping into unseen walls or boxes, which in turn causes massive particle rejection and premature depletion of the belief state. Alternative B provides consistent, continuous feedback about the agent's immediate vicinity, ensuring the particle filter can smoothly and continuously localize the agent.

3. **Decoupling Position from Heading:**
   By using an egocentric window that remains translation-based (row -1 is always North, row +1 is always South), we successfully decouple the agent's spatial `(x, y)` location from its rotational heading. This cleanly aligns with our architecture, allowing the Particle Filter to track just the position while POMCP handles the hidden heading variables during tree search.