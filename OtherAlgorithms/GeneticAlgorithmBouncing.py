from Environment.MultiAgentEnvironment import UncertaintyReductionMA
from Evaluation.path_plotter import plot_trajectory
from Evaluation.metrics_wrapper import MetricsDataCreator, BenchmarkEvaluator
import numpy as np
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
import random
import datetime
import multiprocessing
import matplotlib.pyplot as plt

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


evaluator = MetricsDataCreator(metrics_names=['Mean Reward', 'Uncertainty', 'Distance', 'Collisions', 'RMSE'], algorithm_name='Simple GA', experiment_name='GAResults')
benchmark = BenchmarkEvaluator(navigation_map=nav)
benchmark.reset_values()

# Maximization #
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
# Individual creator - Ind is a nparray#
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Cromosome is an action in [0,7] #
toolbox.register("attr_bool", random.randint, 0, 7)
# Each individual is a set of n_agents x 100 steps #
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=n_agents*101)
# Population is a list of individuals #
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evalEnv(individual, local_env):
    """ Evaluate the individual from the beginning """

    # Reset conditions #
    local_env.reset()
    R = 0
    done = False
    index = 0
    # Slice individual into agents actions #

    while not done:

        _, r, done, _ = local_env.step(individual[index:index+n_agents])
        index += n_agents
        R += np.mean(r)

    if local_env.fleet.fleet_collisions >= local_env.max_number_of_collisions:
        R = -5

    return R,


def cxTwoPointCopy(ind1, ind2):

    size = len(ind1)
    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else:  # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] = ind2[cxpoint1:cxpoint2].copy(), ind1[cxpoint1:cxpoint2].copy()
    return ind1, ind2


toolbox.register("evaluate", evalEnv, local_env=env)
toolbox.register("mate", cxTwoPointCopy)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=5)

pool = multiprocessing.Pool()
toolbox.register("map", pool.map)



def optimize(local_env, save=False):


    random.seed(64)

    pop = toolbox.population(n=1000)

    # np equality function (operators.eq) between two arrays returns the
    # equality element wise, which raises an exception in the if similar()
    # check of the hall of fame. Using a different equality function like
    # np.array_equal or np.allclose solve this issue.

    # Fix the evaluation function with the current environment
    toolbox.register("evaluate", evalEnv, local_env=local_env)

    hof = tools.HallOfFame(5, similar=np.array_equal)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.5, ngen=60, stats=stats, halloffame=hof)

    if save:

        with open(f"ga_simple_optimization_result_{datetime.datetime.now().strftime('%Y_%m_%d-%H:%M_%S')}.txt", "w") as solution_file:

            solution_file.write("Optimization result for the GA\n")
            solution_file.write("---------------------------------\n")
            solution_file.write(f"Starting positions: {local_env.initial_positions}\n")
            solution_file.write("---------------------------------\n")
            solution_file.write("--------- Best Individuals -----------\n")

            for idx, individual in enumerate(hof):
                str_data = ','.join(str(i) for i in individual)
                solution_file.write(f"Individual {idx}: {str_data}\n")
                solution_file.write(f"Fitness: {individual.fitness.values}\n")

            solution_file.close()

    return hof[0]


if __name__ == "__main__":

    np.random.seed(0)

    my_env = UncertaintyReductionMA(navigation_map=nav,
                                 number_of_agents=n_agents,
                                 initial_positions=init_pos[0:4, :],
                                 movement_length=1,
                                 distance_budget=100,
                                 random_initial_positions=False,
                                 initial_meas_locs=None)

    for run in range(10):

        print("Run ", run)

        done, t, indx = False, 0, 0

        R = 0

        selected_positions = np.random.choice(np.arange(0, len(init_pos)), size=n_agents, replace=False)
        my_env.initial_positions = init_pos[selected_positions]
        s = my_env.reset()

        """ OPTIMIZE THE SCENARIO """
        best = np.asarray(optimize(my_env))

        s = my_env.reset()
        benchmark.reset_values()
        benchmark.update_rmse(positions=my_env.fleet.get_positions())
        positions = my_env.fleet.get_positions().flatten()

        print(my_env.initial_positions)

        while not done:

            s, r, done, info = my_env.step(best[indx:indx+n_agents])
            R += np.mean(r[0])

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

pool.close()
