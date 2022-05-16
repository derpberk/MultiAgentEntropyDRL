from Environment.MultiAgentEnvironment import UncertaintyReductionMA
import numpy as np
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
import random
import gym
import pickle
import datetime


# -------- PARAMETERS -------- #
n_max_actions = 100

# Maximization #
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
# Individual creator - Ind is a nparray#
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Cromosome is an action in [0,7] #
toolbox.register("attr_bool", random.randint, 0, 7)
# Each individual is a set of n_agents x 100 steps #
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=n_max_actions)
# Population is a list of individuals #
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Create the base environment #
# This environment
baseEnvironment = gym.make("LunarLander-v2")


def evalEnv(individual):
    """ Evaluate the individual from the beginning """

    # Reset conditions #
    local_env.reset()
    R = 0
    done = False
    index = 0
    # Slice individual into agents actions #

    while not done:
        _, r, done, _ = local_env.step(individual[index])
        index += 1
        R += r

    return R,

def cxTwoPointCopy(ind1, ind2):
    """ Numpy array TwoPoint crossover operation """
    size = len(ind1)
    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else:  # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] = ind2[cxpoint1:cxpoint2].copy(), ind1[cxpoint1:cxpoint2].copy()
    return ind1, ind2

# Register the genetic operations #
toolbox.register("evaluate", evalEnv)
toolbox.register("mate", cxTwoPointCopy)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=5)


def optimize_one_episode():

    random.seed(64)

    pop = toolbox.population(n=1000)

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

    algorithms.eaSimple(pop, toolbox, cxpb=0.6, mutpb=0.4, ngen=100, stats=stats, halloffame=hof)



if __name__ == "__main__":
    main()

