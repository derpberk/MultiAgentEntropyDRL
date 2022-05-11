from Environment.MultiAgentEnvironment import UncertaintyReductionMA
import numpy as np
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
import random
import pickle
import datetime

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
                             initial_meas_locs=None,
                             only_uncertainty=True)

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


def evalEnv(individual):
    """ Evaluate the individual from the beginning """

    # Reset conditions #
    env.reset()
    R = 0
    done = False
    index = 0
    # Slice individual into agents actions #

    while not done:
        _, r, done, _ = env.step(individual[index:index+n_agents])
        index += n_agents
        R += np.mean(r)

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


toolbox.register("evaluate", evalEnv)
toolbox.register("mate", cxTwoPointCopy)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=5)


def main():

    random.seed(64)

    pop = toolbox.population(n=300)

    # np equality function (operators.eq) between two arrays returns the
    # equality element wise, which raises an exception in the if similar()
    # check of the hall of fame. Using a different equality function like
    # np.array_equal or np.allclose solve this issue.
    hof = tools.HallOfFame(5, similar=np.array_equal)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.6, mutpb=0.4, ngen=50, stats=stats, halloffame=hof)

    cp = dict(population=pop, halloffame=hof, logbook=logbook, rndstate=random.getstate())

    with open(f"ga_optimization_result_{datetime.datetime.now().strftime('%Y_%m_%d-%H:%M_%S')}.pkl", "wb") as cp_file:
        pickle.dump(cp, cp_file)


if __name__ == "__main__":
    main()
