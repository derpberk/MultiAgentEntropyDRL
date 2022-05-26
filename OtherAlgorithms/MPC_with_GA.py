import copy
import random
import numpy as np
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
import time
from Environment.MultiAgentEnvironment import UncertaintyReductionMA



# --------- PARAMETERS ---------- #
OPTIMIZATION_HORIZONT = 20
NUM_OF_ACTIONS = 8

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
        R += np.sum(reward)
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

    my_env = UncertaintyReductionMA(navigation_map=nav,
                                 number_of_agents=n_agents,
                                 initial_positions=init_pos,
                                 random_initial_positions=False,
                                 movement_length=1,
                                 distance_budget=100,
                                 initial_meas_locs=None)

    my_env.reset()

    done = False
    R = 0
    t=0
    A = []
    while not done:
        pop, log, hof = optimize_with_budget(global_env=my_env, t=t)
        A.append(hof[0][:n_agents])
        _, r, done, _ = my_env.step(hof[0][:n_agents])
        R += np.mean(r)
        print("Current real Reward: ", R)
        t+=1
        #my_env.render()

    print("Best individual was: %s" % hof[0])
    np.savetxt("./MPC_with_GA_results_HORIZON_10.csv", A, delimiter=",")
    print("The real score was: %s" % R)


