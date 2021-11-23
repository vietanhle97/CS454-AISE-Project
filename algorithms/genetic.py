import sys
import os
sys.path.append(os.getcwd() + '/..')
from fitness_function import FitnessFunction
import random
from roulette_wheel_selection import RouletteWheelSelection
from tournament_selection import TournamentSelection
from local_search import LocalSearch

def create_first_generation(options, search_params):

	first_gen = {}

	for k in options:
		if k in search_params:
			first_gen[k] = random.choice(options[k])
		else:
			first_gen[k] = options[k]

	return first_gen

def init_population(options, search_params, size):
	population = []
	for i in range(size):
		individual = create_first_generation(options, search_params)
		population.append((individual, FitnessFunction.calculate_fitness(individual)))

	return population

def rank_individuals(population):

	return sorted(population, key = lambda item: item[1], reverse = True)

def mating_pool(population, selection):
	pool = []
	for i in range(len(selection)):
		index = selection[i]
		pool.append(population[index])
	return pool

def crossover(p1, p2):
	child = {}
	
	idx = random.randint(0, len(p1))

	params = list(p1[0].keys())

	for i, e in enumerate(params):
		if i < idx:
			child[e] = p1[0][e]
		else:
			child[e] = p2[0][e]

	return child

def crossover_population(mating_pool, size):
	children = []

	length = len(mating_pool) - size
	pool = random.sample(mating_pool, len(mating_pool))

	for i in range(size):
		children.append(mating_pool[i])

	for i in range(length):
		child = crossover(pool[i], pool[len(mating_pool)-i-1])
		children.append((child, FitnessFunction.calculate_fitness(child)))
	return children

def mutate(individual, options, search_params):
	
	params = list(options.keys())

	mutate_gene = random.choice(search_params)

	individual[0][mutate_gene] = random.choice(options[mutate_gene])

	return individual 

def mutate_population(population, options, search_params):
	mutated_population = []

	for i in range(len(population)):
		mutated_index = mutate(population[i], options, search_params)
		mutated_population.append(mutated_index)
	return mutated_population

def next_generation(current, size, options, strategy, search_params):\

	pop_ranked = rank_individuals(current)

	results = []
	if strategy == "Tournament":
		results = TournamentSelection.select(pop_ranked, size)
	elif strategy == "Roulette Wheel":
		results = RouletteWheelSelection.select(pop_ranked, size)
	matingpool = mating_pool(current, results)
	children = crossover_population(matingpool, size)
	next_gen = mutate_population(children, options, search_params)
	return next_gen


	# plt.show()

def memetic(options, search_params, pop_size, selection_size, generations, strategy):
	pop = init_population(options, search_params, pop_size)
	distance = rank_individuals(pop)[0][1]

	for i in range(generations):
		pop = next_generation(pop, selection_size, options, strategy, search_params)
		tmp = rank_individuals(pop)[0][1]
		if i == generations - 1:
			distance = tmp

	return distance


class GeneticAlgorithm:

	def __init__(self):
		pass

	@staticmethod
	def execute(options, search_params, strategy, figure):
		return memetic(options = options, search_params = search_params, pop_size=3, selection_size=2,  generations=10, strategy=strategy)

