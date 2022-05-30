from Algorithm.RainbowDQL.Agent.DuelingDQNAgent import MultiAgentDuelingDQNAgent
from Environment.MultiAgentEnvironment import UncertaintyReductionMA
import numpy as np
import matplotlib.pyplot as plt
from metrics_wrapper import MetricsDataCreator, BenchmarkEvaluator
from path_plotter import plot_trajectory


nav = np.genfromtxt('../Environment/example_map.csv', delimiter=',')
n_agents = 4
init_pos = np.array([[22,24],
                     [16,16],
                     [10,13],
                     [40,28],
                     [31,11],
                     [33,26],
                     [47,30],
                     [39,18],
                     [22,6],
                     [6,17]])

init_pos = init_pos.astype(int)

env = UncertaintyReductionMA(navigation_map=nav,
                             number_of_agents=n_agents,
                             initial_positions=init_pos,
                             movement_length=1,
                             distance_budget=100,
                             random_initial_positions=False,
                             initial_meas_locs=None)

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

evaluator = MetricsDataCreator(metrics_names=['Mean Reward', 'Uncertainty', 'Distance', 'Collisions', 'RMSE'], algorithm_name='Epsilon DRL', experiment_name='BestEpsilonDRLResults')
benchmark = BenchmarkEvaluator(navigation_map=nav)
benchmark.reset_values()

env.return_individual_rewards = True

multiagent.load_model('/home/azken/Samuel/MultiAgentEntropyDRL/Learning/runs/Epsilon_Decoupled_Reward/BestPolicy.pth')

multiagent.epsilon = 0.0

np.random.seed(0)

positions = []

for run in range(10):

    done, t = False, 0

    print("Run ", run)

    selected_positions = np.random.choice(np.arange(0,len(init_pos)), size=n_agents, replace = False)
    env.initial_positions = init_pos[selected_positions]
    s = env.reset()
    R = 0
    benchmark.reset_values()
    benchmark.update_rmse(positions=env.fleet.get_positions())

    positions = env.fleet.get_positions().flatten()

    #env.render()

    while not done:


        a = multiagent.select_action(s)

        s,r,done,i = env.step(a)

        R += np.mean(r[0])

        rmse, _ = benchmark.update_rmse(positions=env.fleet.get_positions())

        metrics = [R, np.mean(env.uncertainty), np.mean(np.sum(env.fleet.get_distance_matrix(), axis=1)/(n_agents-1)), env.fleet.fleet_collisions]

        evaluator.register_step(run_num=run, step = t, metrics=[*metrics, rmse])

        t += 1

        positions = np.vstack((positions, env.fleet.get_positions().flatten()))

        #env.render(pauseint=0.02)

    #plot_trajectory(nav, positions)
    #plt.show(block=True)

    # plt.show(block=True)
evaluator.register_experiment()