from Algorithm.RainbowDQL.Agent.DuelingDQNAgent import MultiAgentDuelingDQNAgent
from Environment.MultiAgentEnvironment import UncertaintyReductionMA
import numpy as np
import matplotlib.pyplot as plt

nav = np.genfromtxt('../Environment/example_map.csv', delimiter=',')
n_agents = 4
init_pos = np.array([[66, 74], [50, 50], [60, 50], [65, 50]])/3
init_pos = init_pos.astype(int)

env = UncertaintyReductionMA(navigation_map=nav,
                             number_of_agents=n_agents,
                             initial_positions=init_pos,
                             movement_length=1,
                             distance_budget=100,
                             initial_meas_locs=None,
                             only_uncertainty=True)

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


multiagent.load_model('/home/samuel/PycharmProjects/MultiAgentEntropyDRL/Learning/runs/Apr27_12-37-46_samuel-linux/BestPolicy.pth')

multiagent.epsilon = 0

done = False
s = env.reset()

#env.render()
R = []

while not done:

    a = multiagent.select_action(s)
    s,r,done,i = env.step(a)
    print(env.get_metrics())
    env.render()
