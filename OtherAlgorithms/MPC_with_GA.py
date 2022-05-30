import copy
import random
import numpy as np
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
import time
from Environment.MultiAgentEnvironment import UncertaintyReductionMA
from Evaluation.metrics_wrapper import MetricsDataCreator, BenchmarkEvaluator
import matplotlib.pyplot as plt



# --------- PARAMETERS ---------- #
OPTIMIZATION_HORIZONT = 20
NUM_OF_ACTIONS = 8

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

env.reset()

# Create the individual base abojects
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Register the individual nature #
toolbox.register("attr_bool", random.randint, 0, NUM_OF_ACTIONS - 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=OPTIMIZATION_HORIZONT*n_agents)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def evalEnv(individual, local_env):
    """ Evaluate the individual from the beginning from the local_env situation """

    # Reset conditions #
    eval_env = copy.deepcopy(local_env)
    R = 0

    for i in range(0, OPTIMIZATION_HORIZONT*n_agents, n_agents):

        # Get new state
        state, reward, done, info = eval_env.step(np.asarray(individual[i:i+n_agents]))
        R += np.mean(reward)
        if done:
            break

    del eval_env

    return R,


def cxTwoPointCopy(ind1, ind2):
    """Execute a two points crossover with copy on the input individuals. The
    copy is required because the slicing in numpy returns a view of the data,
    which leads to a self overwriting in the swap operation. It prevents
    """
    size = len(ind1)
    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else:  # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] \
        = ind2[cxpoint1:cxpoint2].copy(), ind1[cxpoint1:cxpoint2].copy()

    return ind1, ind2


toolbox.register("evaluate", evalEnv, local_env=env)
toolbox.register("mate", cxTwoPointCopy)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)


def optimize_with_budget(global_env,t):
    t0 = time.time()

    random.seed(64)

    # Generate the population
    pop = toolbox.population(n=300)

    # Fix the evaluation function with the current environment
    toolbox.register("evaluate", evalEnv, local_env=copy.deepcopy(global_env))

    hof = tools.HallOfFame(1, similar=np.array_equal)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    print(f"---------- OPTIMIZING THE MODEL. STEP {t} ----------")
    pop, log = algorithms.eaMuPlusLambda(pop, toolbox, mu=50, lambda_=50, cxpb=0.5, mutpb=0.5, ngen=20, stats=stats, halloffame=hof, verbose=True)
    print(f"-----------------------------------------------------------------------------------------------")
    return pop, log, hof


if __name__ == "__main__":

    plt.switch_backend('TkAgg')
    plt.ion()

    nav = np.genfromtxt('../Environment/example_map.csv', delimiter=',')
    n_agents = 4
    init_pos = np.array([[6, 16],
                         [14, 18],
                         [21, 18],
                         [28, 23],
                         [36, 29],
                         [45, 28],
                         [41, 21],
                         [33, 13],
                         [26, 8],
                         [15, 10]])

    init_pos = init_pos.astype(int)

    evaluator = MetricsDataCreator(metrics_names=['Mean Reward', 'Uncertainty', 'Distance', 'Collisions', 'RMSE'], algorithm_name='MPC-GA', experiment_name=f'MPC_GA_H{OPTIMIZATION_HORIZONT}_Results')
    benchmark = BenchmarkEvaluator(navigation_map=nav)
    benchmark.reset_values()


    my_env = UncertaintyReductionMA(navigation_map=nav,
                                 number_of_agents=n_agents,
                                 initial_positions=init_pos,
                                 random_initial_positions=False,
                                 movement_length=1,
                                 distance_budget=100,
                                 initial_meas_locs=None)

    done = False
    R = 0
    t=0
    A = []


    np.random.seed(0)

    for run in range(10):

        print("Run ", run)

        done, t, indx = False, 0, 0

        R = 0

        selected_positions = np.random.choice(np.arange(0, len(init_pos)), size=n_agents, replace=False)
        my_env.initial_positions = init_pos[selected_positions]
        s = my_env.reset()

        benchmark.reset_values()
        benchmark.update_rmse(positions=my_env.fleet.get_positions())
        positions = my_env.fleet.get_positions().flatten()

        while not done:

            pop, log, hof = optimize_with_budget(global_env=my_env, t=t)

            _, r, done, _ = my_env.step(hof[0][:n_agents])
            R += np.mean(r)

            print("Current real Reward: ", R)

            rmse, _ = benchmark.update_rmse(positions=my_env.fleet.get_positions())

            metrics = [R, np.mean(my_env.uncertainty),
                       np.mean(np.sum(my_env.fleet.get_distance_matrix(), axis=1) / (n_agents - 1)),
                       my_env.fleet.fleet_collisions]

            evaluator.register_step(run_num=run, step=t, metrics=[*metrics, rmse])

            positions = np.vstack((positions, my_env.fleet.get_positions().flatten()))

            indx += n_agents
            t += 1

    # plot_trajectory(nav, positions)
    # plt.show(block=True)

    evaluator.register_experiment()