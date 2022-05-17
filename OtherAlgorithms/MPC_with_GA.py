import copy
import random
import numpy as np
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
import gym
import time
import matplotlib.pyplot as plt


# --------- PARAMETERS ---------- #
OPTIMIZATION_HORIZONT = 70
NUM_OF_ACTIONS = 2

env = gym.make('CartPole-v0')

# Create the individual base abojects
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Register the individual nature #
toolbox.register("attr_bool", random.randint, 0, NUM_OF_ACTIONS - 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=OPTIMIZATION_HORIZONT)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def evalEnv(individual, local_env):
    """ Evaluate the individual from the beginning from the local_env situation """

    # Reset conditions #
    eval_env = copy.deepcopy(local_env)
    R = 0

    for action in individual:
        # Get new state
        state, reward, done, info = eval_env.step(action)
        R += reward
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

    print(f"---------- OPTIMIZING THE MODEL WITH A PREDICTION HORIZON OF {OPTIMIZATION_HORIZONT} ----------")
    pop, log = algorithms.eaMuPlusLambda(pop, toolbox, mu=50, lambda_=50, cxpb=0.5, mutpb=0.5, ngen=20, stats=stats, halloffame=hof, verbose=True)
    print(f"-----------------------------------------------------------------------------------------------")
    return pop, log, hof


if __name__ == "__main__":

    my_env = gym.make('CartPole-v0')
    my_env.reset()
    done = False
    R = [0]
    ExpRew = [0]
    t=0
    while not done and R[-1] < 195:
        pop, log, hof = optimize_with_budget(global_env=my_env, t=t)
        _, _, done, _ = my_env.step(hof[0][0])
        t+=1

    print("Best individual was: %s" % hof[0])
    print("The real score was: %s" % t)


