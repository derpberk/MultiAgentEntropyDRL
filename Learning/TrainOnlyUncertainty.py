from Algorithm.RainbowDQL.Agent.DuelingDQNAgent import MultiAgentDuelingDQNAgent
from Environment.MultiAgentEnvironment import UncertaintyReductionMA
import numpy as np

nav = np.genfromtxt('../Environment/example_map.csv', delimiter=',')
n_agents = 4
init_pos = np.array([[66, 74], [50, 50], [60, 50], [65, 50]])/3
init_pos = init_pos.astype(int)

env = UncertaintyReductionMA(navigation_map=nav,
                             number_of_agents=n_agents,
                             initial_positions=init_pos,
                             random_initial_positions=True,
                             movement_length=1,
                             distance_budget=100,
                             initial_meas_locs=None)

multiagent = MultiAgentDuelingDQNAgent(env=env,
                                       memory_size=int(1E6),
                                       batch_size=64,
                                       target_update=1,
                                       soft_update=True,
                                       tau=0.0001,
                                       epsilon_values=[1.0, 0.05],
                                       epsilon_interval=[0.0, 0.33],
                                       learning_starts=500,
                                       gamma=0.99,
                                       lr=1e-4,
                                       noisy=True,
                                       safe_actions=False,
                                       save_every=5000)

multiagent.train(episodes=50000)
