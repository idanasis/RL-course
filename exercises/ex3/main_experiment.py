import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

# Import the algorithms
from q_learning import TabularQLearning
from sarsa import TabularSARSA
from reinforce import TabularREINFORCE

def main():
    seeds = [0, 1, 2, 3, 4]
    max_steps_per_episode = 200
    total_training_steps = 500_000
    eval_interval = 10_000

    # 51 evaluation points: 0, 10_000, 20_000, ..., 500_000
    num_evals = (total_training_steps // eval_interval) + 1

    # Data structures: shape (n_seeds, n_evals)
    q_learning_results = np.zeros((len(seeds), num_evals))
    sarsa_results = np.zeros((len(seeds), num_evals))
    reinforce_results = np.zeros((len(seeds), num_evals))

    eval_steps = None

    for seed_idx, seed in enumerate(seeds):
        print(f"--- Running seed {seed} ---")
        np.random.seed(seed)

        # Initialize environments
        env = gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=True, max_episode_steps=max_steps_per_episode)
        eval_env = gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=True, max_episode_steps=max_steps_per_episode)

        # Seed environments completely
        env.reset(seed=seed)
        env.action_space.seed(seed)
        eval_env.reset(seed=seed)
        eval_env.action_space.seed(seed)

        # Q-Learning
        print(f"Training Q-Learning (seed {seed})...")
        agent_ql = TabularQLearning(env, eval_env)
        evals_ql = agent_ql.train(total_training_steps=total_training_steps)
        if eval_steps is None:
            eval_steps = [e[0] for e in evals_ql]
        q_learning_results[seed_idx, :] = [e[1] for e in evals_ql]

        # SARSA
        print(f"Training SARSA (seed {seed})...")
        agent_sarsa = TabularSARSA(env, eval_env)
        evals_sarsa = agent_sarsa.train(total_training_steps=total_training_steps)
        sarsa_results[seed_idx, :] = [e[1] for e in evals_sarsa]

        # REINFORCE
        print(f"Training REINFORCE (seed {seed})...")
        agent_reinforce = TabularREINFORCE(env, eval_env)
        evals_reinforce = agent_reinforce.train(total_training_steps=total_training_steps, use_baseline=False)
        reinforce_results[seed_idx, :] = [e[1] for e in evals_reinforce]

    # Calculate means and stds
    ql_mean = np.mean(q_learning_results, axis=0)
    ql_std = np.std(q_learning_results, axis=0)

    sarsa_mean = np.mean(sarsa_results, axis=0)
    sarsa_std = np.std(sarsa_results, axis=0)

    rf_mean = np.mean(reinforce_results, axis=0)
    rf_std = np.std(reinforce_results, axis=0)

    # Plotting
    plt.figure(figsize=(10, 6))

    # Q-Learning
    plt.plot(eval_steps, ql_mean, label='Q-Learning', color='blue')
    plt.fill_between(eval_steps, ql_mean - ql_std, ql_mean + ql_std, color='blue', alpha=0.2)

    # SARSA
    plt.plot(eval_steps, sarsa_mean, label='SARSA', color='green')
    plt.fill_between(eval_steps, sarsa_mean - sarsa_std, sarsa_mean + sarsa_std, color='green', alpha=0.2)

    # REINFORCE
    plt.plot(eval_steps, rf_mean, label='REINFORCE', color='red')
    plt.fill_between(eval_steps, rf_mean - rf_std, rf_mean + rf_std, color='red', alpha=0.2)

    plt.xlabel('Number of environment steps')
    plt.ylabel('Estimated value of initial state')
    plt.title('Comparison of Tabular RL Algorithms on FrozenLake-v1 (8x8)')
    plt.legend()
    plt.grid(True)

    plt.savefig('frozenlake_comparison.png')
    plt.show()

if __name__ == "__main__":
    main()
