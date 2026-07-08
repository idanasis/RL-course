# Exercise 3: Tabular Reinforcement Learning

## Running the Experiment
To run the main experiment and generate the comparison plot, execute the following command from the repository root:
```bash
python exercises/ex3/main_experiment.py
```

## 4. Explanation of the $V(s_0)$ Evaluation Method 
To evaluate the value of the initial state $V(s_0)$, the training process is temporarily paused, and no updates are made to the Q-table or the policy parameters $\theta$. Then, 500 evaluation episodes are executed, always starting from the initial state (state 0). 

During these evaluation episodes, the agent follows a strictly greedy policy: it chooses the action with the maximum Q-value for Q-learning and SARSA, or the action with the highest probability for REINFORCE. For each episode, the discounted return ($G$) is calculated, and the reported empirical value for that training step is the mean of these 500 returns. This provides an objective, empirical measure of the policy's quality at that exact moment in training.

## 5. Discussion of the Results

### Who learned faster?
Q-learning generally learns faster and reaches its peak performance earlier because it assumes the future policy will be perfectly optimal and greedy. 

### Who reached better performance?
SARSA ultimately reaches a similar average performance level, but it requires more environment steps to stabilize its learning curve. Both plateau at similar levels due to the environment's complexity.

### Did Q-learning and SARSA act differently?
Yes. Q-learning is an off-policy algorithm that strives to find the absolute shortest "optimal" path to the goal, even if that path runs directly alongside holes. This risk-taking behavior often causes dips in the performance graph because the agent randomly slips into holes during evaluation. 

Conversely, SARSA is an on-policy algorithm that evaluates the actual policy being executed (including the risky $\epsilon$-greedy exploration). Therefore, it learns a "safe" path that avoids holes, resulting in a more consistent learning curve but a longer, more conservative route.

### Was REINFORCE less stable?
The REINFORCE algorithm proves to be highly unstable and exhibits poor performance. Because it is a Monte Carlo method, it relies on the cumulative return of the entire episode ($G_t$) as its learning signal. In a complex, random environment with sparse rewards (reward is only given at the very end), the variance of these returns is massive. Without a Baseline mechanism to reduce this variance, the signal is too noisy, and the algorithm struggles to figure out which specific actions actually led to success.

### How the stochasticity of Frozen Lake affected the learning
The sliding dynamics of the environment (`is_slippery=True`) completely dictate the learning process. It is the primary reason why the maximum performance of the algorithms plateaus around 0.22 (meaning the agent only reaches the goal about 22% of the time). This stochasticity makes short paths extremely dangerous (hindering Q-learning), forces long detours (for SARSA), and destroys the learning capability of high-variance algorithms like Vanilla REINFORCE.
