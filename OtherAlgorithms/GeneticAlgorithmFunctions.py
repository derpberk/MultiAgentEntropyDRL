import copy
import time
import numpy as np
import random
import gym

class GAhorizontOptimizer:

	def __init__(self, number_of_actions, horizon, pop_size, cx_rate, mut_rate, mu, lambd):


		# This is the environment provided in the initial state to optimize #
		self.base_environment = None
		# Number of possible actions to perform in each step #
		self.number_of_actions = number_of_actions
		# Number of steps in the future to optimize #
		self.horizon = horizon
		# Population size #
		self.pop_size = pop_size
		# Crossover rate #
		self.cx_rate = cx_rate
		# Mutation rate #
		self.mut_rate = mut_rate
		self.mu = mu
		self.lambd = lambd
		self.n_childrens = mu // lambd

	@staticmethod
	def evaluate_individual(individual, env_arg):
		""" Evaluate the individual from the beginning """

		# Reset conditions #
		local_env = copy.deepcopy(env_arg)
		R = 0

		for index in range(len(individual)):
			_, r, done, _ = local_env.step(individual[index])
			R += r

			if done:
				break

		del local_env

		return R

	def optimize(self, env, time_budget, n_max_generations):
		""" Evolute the population until time budget is reached or n_generations_reached """

		# Get the initial time
		t0 = time.time()
		# Copy the environment to be used in the optimization #
		base_env = copy.deepcopy(env)
		# Create the population #
		population = self.create_population(self.pop_size)

		gens = 0

		while time.time() - t0 < time_budget:

			# Evaluate the population #
			fitness = self.eval_population(population, base_env)
			# Generate offspring #
			offspring = self.generate_offspring(population, self.lambd, self.cx_rate, self.mut_rate)
			# Evaluate offspring #
			offspring_fitness = self.eval_population(offspring, base_env)
			# Replace the population #
			population = self.replace_population(population, offspring, fitness, offspring_fitness, self.mu)

			self.print_results(population, fitness, gens)

			gens+=1
			if gens >= n_max_generations:
				break

		return population[np.argmax(fitness)][0]

	@staticmethod
	def print_results(population, fitness, gens):
		""" Print the results """

		print("Generation: ", gens)
		max_fitness_indx = np.argmax(fitness)
		print("Best individual: ", population[max_fitness_indx])
		print("Fitness: ", fitness[max_fitness_indx])
		print("\n")

	def eval_population(self, population, env):
		""" Evaluate the population """

		# Evaluate the population #
		fitness = []
		for individual in population:
			fitness.append(self.evaluate_individual(individual, env))

		return np.asarray(fitness)

	def create_population(self, pop_size):
		""" Create a population of individuals """

		# Create the population #
		population = []
		for _ in range(pop_size):
			individual = np.random.randint(0, self.number_of_actions, size=self.horizon)
			population.append(individual)

		return np.asarray(population)

	def generate_offspring(self, population, lambd, cx_rate, mut_rate):
		""" Generate offspring """

		# Generate offspring #
		offspring = []
		for _ in range(lambd):

			if np.random.rand() < cx_rate:
				# Crossover parents #
				new_individual = self.crossover(population)
			elif np.random.rand() < mut_rate:
				# Mutate offspring #
				new_individual = self.mutate(population)
			else:
				new_individual = self.clone(population)

			# Append offspring #
			offspring.append(new_individual)

		return np.asarray(offspring)

	def crossover(self, population):
		""" Choose randomly two individuals from the population and perform
		a Two point crossover operation. Return  the first child. """

		# Choose randomly two elements from population #

		indxs = np.random.randint(0, len(population), size=2)
		parents = population[indxs]
		# Perform Two point crossover #
		child1, child2 = self.two_point_crossover(parents[0], parents[1])
		# Return the first child #
		return child1

	@staticmethod
	def two_point_crossover(parent1, parent2):
		""" Perform a Two point crossover operation """

		# Choose randomly two points #
		point1 = np.random.randint(0, len(parent1))
		point2 = np.random.randint(point1, len(parent1))
		# Create the children #
		child1 = np.concatenate((parent1[:point1], parent2[point1:point2], parent1[point2:]))
		child2 = np.concatenate((parent2[:point1], parent1[point1:point2], parent2[point2:]))
		# Return the children #
		return child1, child2

	def mutate(self, population):
		""" Perform a mutation operation """

		# Choose randomly an individual #

		individual = population[np.random.randint(0,len(population))]
		# Mutate the individual #
		new_individual = self.mutate_individual(individual)
		# Return the mutated individual #
		return new_individual

	def mutate_individual(self, individual):
		""" Perform a mutation operation on an individual """

		new_individual = individual.copy()
		# Choose randomly a position #
		position = np.random.randint(0, len(individual))
		# Choose randomly a value #
		value = np.random.randint(0, self.number_of_actions)
		# Replace the value #
		new_individual[position] = value
		# Return the mutated individual #
		return new_individual

	@staticmethod
	def clone(population):
		""" Clone an individual """

		# Choose randomly an individual #
		individual = population[np.random.randint(0,len(population))]
		# Return the cloned individual #
		return individual.copy()

	def replace_population(self, population, offspring, fitness, offspring_fitness, mu):
		""" Replace the population """

		# Replace the population #
		population = np.concatenate((population, offspring), axis=0)
		fitness = np.concatenate((fitness, offspring_fitness), axis=0)

		new_population, new_fitness = self.select_population(population, fitness, mu)

		# Return the sorted population #
		return new_population

	def select_population(self, population, fitness, mu):
		""" Select the population """

		# Sort the population #
		sorted_population, sorted_fitness = self.sort_population(population, fitness)
		# Return the sorted population #
		return sorted_population[:mu], sorted_population[mu:]

	@staticmethod
	def sort_population(population, fitness):
		""" Sort the population """

		# Sort the population #
		sorted_indx = np.argsort(fitness)
		sorted_population = population[sorted_indx]
		sorted_fitness = fitness[sorted_indx]
		# Return the sorted population #
		return sorted_population, sorted_fitness

if __name__ == "__main__":


	env = 	gym.make("Breakout-v4")
	env.reset()
	optimizer = GAhorizontOptimizer(number_of_actions = 3, horizon = 10, pop_size = 50, cx_rate = 0.6, mut_rate=0.4, mu = 50, lambd=50)
	done = False

	while not done:

		action = optimizer.optimize(env, time_budget = 5, n_max_generations=100)
		_, _, done, _ = env.step(action)
		env.render()