
import numpy as np
from Environment.MultiAgentEnvironment import UncertaintyReductionMA

import matplotlib
matplotlib.use('TKAgg')

nav = np.genfromtxt('../Environment/example_map.csv', delimiter=',')
n_agents = 4
init_pos = np.array([[66, 74], [50, 50], [60, 50], [65, 50]]) / 3
init_pos = init_pos.astype(int)

# Create the environment #
env = UncertaintyReductionMA(navigation_map=nav,
                             number_of_agents=n_agents,
                             initial_positions=init_pos,
                             random_initial_positions=False,
                             movement_length=1,
                             distance_budget=100,
                             initial_meas_locs=None)
env.reset()
A = np.genfromtxt('MPC_with_GA_results_HORIZON_10.csv', delimiter=',')
A = A.astype(int)
R=0

for a in A:
	_,r,_,_ = env.step(a)
	print(env.get_metrics())
	env.render()
	R += np.mean(r)

print(R)
