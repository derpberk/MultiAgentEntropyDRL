from Environment.MultiAgentEnvironment import UncertaintyReductionMA
from Algorithm.RainbowDQL.Agent.DuelingDQNAgent import MultiAgentDuelingDQNAgent
import numpy as np

nav = np.genfromtxt('../Environment/ypacarai_map.csv')
n_agents = 4
init_pos = np.array([[65, 75], [75, 80], [85, 90], [100, 95]])

env = UncertaintyReductionMA(navigation_map=nav, number_of_agents=n_agents, initial_positions=init_pos, movement_length=10, distance_budget=200, initial_meas_locs=None)

multiagent = MultiAgentDuelingDQNAgent(env=env,
                                       memory_size=int(1E5),
                                       batch_size=64,
                                       target_update=1,
                                       soft_update=True,
                                       tau=0.0001,
                                       epsilon_values=[1.0, 0.05],
                                       epsilon_interval=[0.0, 0.33],
                                       learning_starts=0,
                                       gamma=0.99,
                                       lr=1e-4,
                                       noisy=False,
                                       safe_actions=False)

multiagent.train(episodes=10000)
