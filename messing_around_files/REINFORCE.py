import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import json
from typing import List, Tuple, Dict, Any

# For Bayesian optimization
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical as SkoptCategorical
    from skopt.utils import use_named_args
    BAYESIAN_AVAILABLE = True
except ImportError:
    print("scikit-optimize not available. Install with: pip install scikit-optimize")
    print("Falling back to random search...")
    BAYESIAN_AVAILABLE = False
    import random

# ENV
env_string = "LunarLander-v3"

class Trajectory:
    """Container for a single trajectory"""
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.returns = []
    
    def add_step(self, state, action, reward, log_prob):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
    
    def compute_returns(self, gamma: float):
        """Compute discounted returns for this trajectory"""
        T = len(self.rewards)
        self.returns = np.zeros(T, dtype=np.float32)
        future_return = 0.0
        for t in reversed(range(T)):
            future_return = self.rewards[t] + gamma * future_return
            self.returns[t] = future_return
        return torch.tensor(self.returns)

class Pi(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, num_layers=2):
        super(Pi, self).__init__()
        
        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        # Add additional hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_dim, out_dim))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def act(self, state):
        x = torch.from_numpy(state.astype(np.float32))
        pdparam = self.forward(x)
        pd = Categorical(logits=pdparam)
        action = pd.sample()
        log_prob = pd.log_prob(action)
        return action.item(), log_prob

def collect_trajectory(env, policy, max_steps=1000) -> Trajectory:
    """Collect a single trajectory"""
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        state = reset_result[0]
    else:
        state = reset_result
    
    trajectory = Trajectory()
    done = False
    steps = 0
    
    while not done and steps < max_steps:
        action, log_prob = policy.act(state)
        step_result = env.step(action)
        next_state, reward, terminated, truncated = step_result[:4]
        
        trajectory.add_step(state, action, reward, log_prob)
        
        state = next_state
        done = terminated or truncated
        steps += 1
    
    return trajectory

def collect_batch(env, policy, batch_size: int, max_steps=1000) -> List[Trajectory]:
    """Collect a batch of trajectories"""
    trajectories = []
    for _ in range(batch_size):
        trajectory = collect_trajectory(env, policy, max_steps)
        trajectories.append(trajectory)
    return trajectories

def moving_average(data, window_size=10):
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def main(
    batch_size: int = 8,
    user_baseline: bool = False,
    gamma: float = 0.99,
    hidden_dim: int = 64,
    num_layers: int = 2,
    optim_learn_rate: float = 0.01,
    render: bool = False,
    plot: bool = False,
    episodes: int = 1000,
    verbose: bool = True,
    normalize_returns: bool = True,
    max_steps_per_episode: int = 1000
) -> tuple:
    """
    Batched REINFORCE implementation
    Returns (mean_reward, final_loss) for optimization
    """
    if render:
        env = gym.make(env_string, render_mode="human")
    else:
        env = gym.make(env_string)
    
    in_dim = env.observation_space.shape[0]
    out_dim = env.action_space.n
    pi = Pi(in_dim, out_dim, hidden_dim, num_layers)
    optimizer = optim.Adam(pi.parameters(), lr=optim_learn_rate)
    
    episode_rewards = []
    losses = []
    episodes_completed = 0
    
    while episodes_completed < episodes:
        # Collect batch of trajectories
        trajectories = collect_batch(env, pi, batch_size, max_steps_per_episode)
        episodes_completed += len(trajectories)
        
        # Compute returns for each trajectory
        all_log_probs = []
        all_returns = []
        batch_rewards = []
        
        for trajectory in trajectories:
            returns = trajectory.compute_returns(gamma)
            log_probs = torch.stack(trajectory.log_probs)
            
            all_log_probs.append(log_probs)
            all_returns.append(returns)
            batch_rewards.append(sum(trajectory.rewards))
        
        # Concatenate all trajectories
        combined_log_probs = torch.cat(all_log_probs)
        combined_returns = torch.cat(all_returns)
        
        # Normalize returns if requested
        if normalize_returns and len(combined_returns) > 1:
            combined_returns = (combined_returns - combined_returns.mean()) / (combined_returns.std() + 1e-8)
        
        # Apply baseline
        if user_baseline:
            baseline = combined_returns.mean()
            advantages = combined_returns - baseline
        else:
            advantages = combined_returns
        
        # Compute loss
        loss = -torch.sum(combined_log_probs * advantages)
        
        # Update policy
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pi.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()
        
        # Track metrics
        mean_batch_reward = np.mean(batch_rewards)
        episode_rewards.extend(batch_rewards)
        losses.append(loss.item())
        
        if verbose and episodes_completed % (batch_size * 10) == 0:
            print(f"Episodes {episodes_completed}: Avg Reward {mean_batch_reward:.2f}, Loss {loss.item():.4f}")

    env.close()
    
    if plot:
        # Plot performance
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Rewards plot
        ax1.set_title("Reward over time")
        ax1.plot(episode_rewards, alpha=0.3, label="Episode Reward")
        if len(episode_rewards) > 50:
            smoothed = moving_average(episode_rewards, window_size=50)
            ax1.plot(range(len(smoothed)), smoothed, label="Moving Avg (50)", linewidth=2)
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Total reward")
        ax1.grid(True)
        ax1.legend()
        
        # Loss plot  
        ax2.set_title("Loss over time (per batch)")
        ax2.plot(losses, alpha=0.7)
        if len(losses) > 10:
            smoothed_loss = moving_average(losses, window_size=10)
            ax2.plot(range(len(smoothed_loss)), smoothed_loss, label="Moving Avg (10)", linewidth=2)
        ax2.set_xlabel("Batch")
        ax2.set_ylabel("Loss")
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    # Return metrics for optimization
    mean_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
    final_loss = np.mean(losses[-10:]) if len(losses) >= 10 else np.mean(losses)
    
    return mean_reward, final_loss

def bayesian_optimization(n_calls=25, episodes_per_run=400):
    """
    Use Bayesian optimization to find best hyperparameters
    """
    if not BAYESIAN_AVAILABLE:
        return random_grid_search(n_calls, episodes_per_run)
    
    # Define search space
    dimensions = [
        Integer(4, 32, name='batch_size'),
        Real(0.9, 0.999, name='gamma'),
        Real(0.0005, 0.05, name='optim_learn_rate'),
        Integer(32, 256, name='hidden_dim'),
        Integer(1, 4, name='num_layers'),
        SkoptCategorical([True, False], name='user_baseline'),
        SkoptCategorical([True, False], name='normalize_returns'),
    ]
    
    results_history = []
    
    @use_named_args(dimensions)
    def objective(**params):
        print(f"Testing: {params}")
        
        try:
            # Convert params for main function
            main_params = {
                'batch_size': params['batch_size'],
                'gamma': params['gamma'],
                'optim_learn_rate': params['optim_learn_rate'],
                'hidden_dim': params['hidden_dim'],
                'num_layers': params['num_layers'],
                'user_baseline': params['user_baseline'],
                'normalize_returns': params['normalize_returns'],
                'episodes': episodes_per_run,
                'verbose': False,
                'plot': False,
                'render': False
            }
            
            mean_reward, final_loss = main(**main_params)
            
            # Store result
            result = {
                'params': params.copy(),
                'mean_reward': mean_reward,
                'final_loss': final_loss
            }
            results_history.append(result)
            
            print(f"  -> Mean Reward: {mean_reward:.2f}")
            
            # Return negative reward since gp_minimize minimizes
            return -mean_reward
            
        except Exception as e:
            print(f"  -> ERROR: {e}")
            return 0  # Return neutral value for failed runs
    
    print(f"Starting Bayesian optimization with {n_calls} evaluations...")
    print(f"Each run will train for {episodes_per_run} episodes")
    print("-" * 60)
    
    # Run Bayesian optimization
    result = gp_minimize(
        func=objective,
        dimensions=dimensions,
        n_calls=n_calls,
        n_initial_points=5,
        acq_func='EI',  # Expected Improvement
        random_state=42
    )
    
    # Extract best parameters
    best_params = {}
    for i, dim in enumerate(dimensions):
        best_params[dim.name] = result.x[i]
    
    best_reward = -result.fun
    
    print("=" * 60)
    print("BAYESIAN OPTIMIZATION COMPLETE!")
    print("=" * 60)
    print(f"Best parameters found:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print(f"Best mean reward: {best_reward:.2f}")
    print()
    
    # Show convergence plot
    plt.figure(figsize=(10, 6))
    rewards = [-r['mean_reward'] for r in results_history]  # Convert back to positive
    plt.plot(rewards, 'bo-', alpha=0.7)
    plt.axhline(y=best_reward, color='r', linestyle='--', label=f'Best: {best_reward:.2f}')
    plt.xlabel('Evaluation')
    plt.ylabel('Mean Reward')
    plt.title('Bayesian Optimization Progress')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return best_params, results_history

def random_grid_search(n_samples=20, episodes_per_run=400):
    """
    Fallback random search if Bayesian optimization not available
    """
    import random
    
    param_ranges = {
        "batch_size": [4, 8, 16, 24, 32],
        "gamma": [0.9, 0.95, 0.99, 0.995, 0.999],
        "optim_learn_rate": [0.0005, 0.001, 0.005, 0.01, 0.02, 0.03],
        "hidden_dim": [32, 64, 96, 128, 192, 256],
        "num_layers": [1, 2, 3, 4],
        "user_baseline": [True, False],
        "normalize_returns": [True, False],
    }
    
    print(f"Starting random grid search with {n_samples} samples...")
    print(f"Each run will train for {episodes_per_run} episodes")
    print("-" * 60)
    
    results = []
    best_reward = float('-inf')
    best_params = None
    
    for i in range(n_samples):
        params = {
            "batch_size": random.choice(param_ranges["batch_size"]),
            "gamma": random.choice(param_ranges["gamma"]),
            "optim_learn_rate": random.choice(param_ranges["optim_learn_rate"]),
            "hidden_dim": random.choice(param_ranges["hidden_dim"]),
            "num_layers": random.choice(param_ranges["num_layers"]),
            "user_baseline": random.choice(param_ranges["user_baseline"]),
            "normalize_returns": random.choice(param_ranges["normalize_returns"]),
            "episodes": episodes_per_run,
            "verbose": False,
            "plot": False,
            "render": False
        }
        
        print(f"Run {i+1}/{n_samples}: {params}")
        
        try:
            mean_reward, final_loss = main(**params)
            
            result = {
                "run": i+1,
                "params": params.copy(),
                "mean_reward": mean_reward,
                "final_loss": final_loss
            }
            results.append(result)
            
            print(f"  -> Mean Reward: {mean_reward:.2f}, Final Loss: {final_loss:.4f}")
            
            if mean_reward > best_reward:
                best_reward = mean_reward
                best_params = params.copy()
                print(f"  -> NEW BEST! 🎉")
            
        except Exception as e:
            print(f"  -> ERROR: {e}")
            continue
        
        print()
    
    results.sort(key=lambda x: x["mean_reward"], reverse=True)
    
    print("=" * 60)
    print("RANDOM SEARCH COMPLETE!")
    print("=" * 60)
    print(f"Best parameters found:")
    for key, value in best_params.items():
        if key not in ['episodes', 'verbose', 'plot', 'render']:
            print(f"  {key}: {value}")
    print(f"Best mean reward: {best_reward:.2f}")
    
    return best_params, results

def run_final_training(best_params, episodes=2000):
    """
    Run final training with best parameters and full visualization
    """
    print("=" * 60)
    print("RUNNING FINAL TRAINING WITH OPTIMAL PARAMETERS")
    print("=" * 60)
    
    final_params = best_params.copy()
    final_params.update({
        "episodes": episodes,
        "plot": True,
        "verbose": True,
        "render": False  # Set to True if you want to see the agent play
    })
    
    print("Final parameters:")
    for key, value in final_params.items():
        print(f"  {key}: {value}")
    print()
    
    mean_reward, final_loss = main(**final_params)
    
    print(f"\nFinal training complete!")
    print(f"Final mean reward (last 100 episodes): {mean_reward:.2f}")
    print(f"Final loss: {final_loss:.4f}")
    
    return mean_reward, final_loss

if __name__ == "__main__":
    # Choose optimization method
    if BAYESIAN_AVAILABLE:
        print("Using Bayesian Optimization (scikit-optimize)")
        best_params, all_results = bayesian_optimization(n_calls=20, episodes_per_run=300)
    else:
        print("Using Random Search")
        best_params, all_results = random_grid_search(n_samples=20, episodes_per_run=300)
    
    # Run final training with best parameters
    run_final_training(best_params, episodes=1500)
    
    # Save results
    with open('optimization_results.json', 'w') as f:
        serializable_results = []
        for result in all_results:
            serializable_result = result.copy()
            if 'mean_reward' in serializable_result:
                serializable_result['mean_reward'] = float(serializable_result['mean_reward'])
            if 'final_loss' in serializable_result:
                serializable_result['final_loss'] = float(serializable_result['final_loss'])
            serializable_results.append(serializable_result)
        
        json.dump({
            'best_params': best_params,
            'all_results': serializable_results
        }, f, indent=2)
    
    print(f"\nResults saved to 'optimization_results.json'")