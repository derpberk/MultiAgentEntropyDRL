from Environment.MultiAgentEnvironment import UncertaintyReductionMA
import numpy as np
import matplotlib.pyplot as plt

nav = np.genfromtxt('../Environment/example_map.csv', delimiter=',')
n_agents = 4
init_pos = np.array([[66, 74], [50, 50], [60, 50], [65, 50]])/3
init_pos = init_pos.astype(int)

env = UncertaintyReductionMA(navigation_map=nav, number_of_agents=n_agents, initial_positions=init_pos, movement_length=1, distance_budget=100, initial_meas_locs=None)
env.return_individual_rewards = True


for t in range(1):

	env.reset()
	done = False

	action = np.random.randint(0, 8, n_agents)

	while any(env.fleet.check_collisions(action)):
		action = np.random.randint(0, 8, n_agents)

	R = []
	Unc = []
	Dist = []
	Colls = []
	Regr = []

	while not done:

		s, r, done, info = env.step(action)

		if any(env.fleet.check_collisions(action)):

			valid = False

			while not valid:
				new_actions = np.random.randint(0, 8, n_agents)
				invalid_mask = env.fleet.check_collisions(action)
				action[invalid_mask] = new_actions[invalid_mask]
				valid = not any(env.fleet.check_collisions(action))

		print("Reward")
		print(r[0])
		R.append(r[0])
		Unc.append(r[1])
		Dist.append(r[2])
		Colls.append(r[3])
		Regr.append(r[4])

		env.render()


	plt.show(block=True)

	fig,axs = plt.subplots(5,1, sharex=True)

	axs[0].plot(np.cumsum(R,axis=0))
	axs[0].set_title('Reward')
	axs[0].legend(['Agent {}'.format(i) for i in range(n_agents)])
	axs[1].plot(np.asarray(Unc))
	axs[1].set_title('Uncertainty')
	axs[2].plot(np.asarray(Dist))
	axs[2].set_title('Distance')
	axs[3].plot(np.asarray(Colls))
	axs[3].set_title('Collisions')
	axs[4].plot(np.asarray(Regr))
	axs[4].set_title('Inverse regret')
	plt.show(block=True)

# plt.close()
