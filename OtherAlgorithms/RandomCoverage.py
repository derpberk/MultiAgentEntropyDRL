from Environment.MultiAgentEnvironment import UncertaintyReductionMA
import numpy as np
import matplotlib.pyplot as plt
from Evaluation.path_plotter import plot_trajectory
from Evaluation.metrics_wrapper import MetricsDataCreator, BenchmarkEvaluator

plt.switch_backend('TkAgg')
plt.ion()

nav = np.genfromtxt('../Environment/example_map.csv', delimiter=',')
n_agents = 4
init_pos = np.array([[6,16],
                     [14,18],
                     [21,18],
                     [28,23],
                     [36,29],
                     [45,28],
                     [41,21],
                     [33,13],
                     [26,8],
                     [15,10]])

init_pos = init_pos.astype(int)

env = UncertaintyReductionMA(navigation_map=nav,
                             number_of_agents=n_agents,
                             initial_positions=init_pos[0:4,:],
                             movement_length=1,
                             distance_budget=100,
                             random_initial_positions=False,
                             initial_meas_locs=None)


evaluator = MetricsDataCreator(metrics_names=['Mean Reward', 'Uncertainty', 'Distance', 'Collisions', 'RMSE'], algorithm_name='Random Wanderer', experiment_name='RandomWandererResults')
benchmark = BenchmarkEvaluator(navigation_map=nav)
benchmark.reset_values()

env.return_individual_rewards = True

np.random.seed(0)


for run in range(10):

	print("Run ", run)
	done, t = False, 0
	R = 0

	selected_positions = np.random.choice(np.arange(0, len(init_pos)), size=n_agents, replace=False)
	env.initial_positions = init_pos[selected_positions]
	s = env.reset()

	benchmark.reset_values()
	benchmark.update_rmse(positions=env.fleet.get_positions())

	positions = env.fleet.get_positions().flatten()

	action = np.random.randint(0, 8, n_agents)

	while any(env.fleet.check_collisions(action)):
		action = np.random.randint(0, 8, n_agents)

	while not done:

		s, r, done, info = env.step(action)
		R += np.mean(r[0])

		if any(env.fleet.check_collisions(action)):

			valid = False

			while not valid:
				new_actions = np.random.randint(0, 8, n_agents)
				invalid_mask = env.fleet.check_collisions(action)
				action[invalid_mask] = new_actions[invalid_mask]
				valid = not any(env.fleet.check_collisions(action))

		rmse,_ = benchmark.update_rmse(positions=env.fleet.get_positions())

		metrics = [R, np.mean(env.uncertainty),
				   np.mean(np.sum(env.fleet.get_distance_matrix(), axis=1) / (n_agents - 1)),
				   env.fleet.fleet_collisions]

		evaluator.register_step(run_num=run, step=t, algorithm_name='Random Wanderer', metrics=[*metrics, rmse])

		positions = np.vstack((positions, env.fleet.get_positions().flatten()))

		t += 1

	#plot_trajectory(nav, positions)
	#plt.show(block=True)

evaluator.register_experiment()