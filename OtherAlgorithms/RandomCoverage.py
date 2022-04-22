from Environment.MultiAgentEnvironment import UncertaintyReductionMA
import numpy as np
import matplotlib.pyplot as plt


nav = np.genfromtxt('../Environment/ypacarai_map.csv')
n_agents = 4
init_pos = np.array([[66, 74],[50,50], [60,50], [65,50]])

env = UncertaintyReductionMA(navigation_map=nav,
                   movement_length=5,
                   number_of_agents=n_agents,
                   initial_positions=init_pos,
                   initial_meas_locs=None,
                   distance_budget=200,
                   device='cpu')

for t in range(1):
	env.reset()
	done = False

	action = np.random.randint(0,8,n_agents)
	while any(env.fleet.check_collisions(action)):
		action = np.random.randint(0,8,n_agents)

	R = []

	while not done:

		_, r, done, info = env.step(action)

		if any(env.fleet.check_collisions(action)):

			valid = False

			while not valid:
				new_actions = np.random.randint(0,8,n_agents)
				invalid_mask = env.fleet.check_collisions(action)
				action[invalid_mask] = new_actions[invalid_mask]
				valid = not any(env.fleet.check_collisions(action))

		print("Reward")
		print(r)


		env.render()

	#plt.show()
	#plt.close()
	#plt.plot(np.cumsum(R, axis=0))
	plt.show(block=True)
	#plt.close()
