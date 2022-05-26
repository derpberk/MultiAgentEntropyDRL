import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams['font.family'] = 'Times New Roman'
plt.style.use('bmh')

nav_map = np.genfromtxt('../../Environment/example_map.csv', delimiter=',')

# 1. Decoupled vs coupled

k1 = (0.6,0.6,0.6)
k2 = (0.15, 0.15, 0.15)

uncertainty_noise_decoupled = pd.read_csv('../../Evaluation/run-logs/run-Noisy_Decoupled_Reward-tag-train_mean_uncertainty.csv')
uncertainty_noise_coupled = pd.read_csv('../../Evaluation/run-logs/run-Noise_Coupled_Reward-tag-train_mean_uncertainty.csv')
uncertainty_epsilon_decoupled = pd.read_csv('../../Evaluation/run-logs/run-Epsilon_Decoupled_Reward-tag-train_mean_uncertainty.csv')


uncertainty_noise_coupled['Value'] = uncertainty_noise_coupled['Value']
uncertainty_noise_decoupled['Value'] = uncertainty_noise_decoupled['Value']
uncertainty_epsilon_decoupled['Value'] = uncertainty_epsilon_decoupled['Value']

alpha = 0.95
plt.plot(uncertainty_noise_coupled['Step'], uncertainty_noise_coupled['Value'].ewm(alpha=1-alpha).mean(), color=k1, label = 'Coupled reward')
plt.plot(uncertainty_noise_decoupled['Step'], uncertainty_noise_decoupled['Value'].ewm(alpha=1-alpha).mean(), color=k2, label = 'Decoupled reward')
plt.legend()

plt.xlabel('Episodes')
plt.ylabel(r'Final uncertainty   $U^{t=T}$')
plt.show()

# 2. Epsilon vs noisy reward

reward_noise_decoupled = pd.read_csv('../../Evaluation/run-logs/run-Noisy_Decoupled_Reward-tag-train_accumulated_reward.csv')
reward_epsilon_decoupled = pd.read_csv('../../Evaluation/run-logs/run-Epsilon_Decoupled_Reward-tag-train_accumulated_reward.csv')

plt.plot(reward_epsilon_decoupled['Step'], reward_epsilon_decoupled['Value'].ewm(alpha=1-alpha).mean(), color=k1, label = r'$\epsilon$-greedy Policy')
plt.plot(reward_noise_decoupled['Step'], reward_noise_decoupled['Value'].ewm(alpha=1-alpha).mean(), color=k2, label = 'Noisy Policy')
plt.legend()

plt.xlabel('Episodes')
plt.ylabel(r'Accumulated agents mean reward')
plt.show()

# 2. Epsilon vs noisy length

reward_noise_decoupled = pd.read_csv('../../Evaluation/run-logs/run-Noisy_Decoupled_Reward-tag-train_accumulated_length.csv')
reward_epsilon_decoupled = pd.read_csv('../../Evaluation/run-logs/run-Epsilon_Decoupled_Reward-tag-train_accumulated_length.csv')

plt.plot(reward_epsilon_decoupled['Step'], reward_epsilon_decoupled['Value'].ewm(alpha=1-alpha).mean(), color=k1, label = r'$\epsilon$-greedy Policy')
plt.plot(reward_noise_decoupled['Step'], reward_noise_decoupled['Value'].ewm(alpha=1-alpha).mean(), color=k2, label = 'Noisy Policy')
plt.legend()

plt.xlabel('Episodes')
plt.ylabel(r'Mission length (movements)')
plt.show()

# 2. Epsilon vs noisy length

reward_noise_decoupled = pd.read_csv('../../Evaluation/run-logs/run-Noisy_Decoupled_Reward-tag-train_mean_uncertainty.csv')
reward_epsilon_decoupled = pd.read_csv('../../Evaluation/run-logs/run-Epsilon_Decoupled_Reward-tag-train_mean_uncertainty.csv')

plt.plot(reward_epsilon_decoupled['Step'], reward_epsilon_decoupled['Value'].ewm(alpha=1-alpha).mean(), color=k1, label = r'$\epsilon$-greedy Policy')
plt.plot(reward_noise_decoupled['Step'], reward_noise_decoupled['Value'].ewm(alpha=1-alpha).mean(), color=k2, label = 'Noisy Policy')
plt.legend()

plt.xlabel('Episodes')
plt.ylabel(r'Final uncertainty   $U^{t=T}$')
plt.show()