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

def init_population(options, search_params, size, model):
	population = []
	for i in range(size):
		individual = create_first_generation(options, search_params)
		fitness = FitnessFunction.calculate_fitness(individual, model)
		population.append((individual, fitness))
	return population

def rank_individuals(population):

	return sorted(population, key = lambda item: item[1], reverse = True)

def mating_pool(population, selection):
	pool = []
	for i in range(len(selection)):
		pool.append(population[selection[i]])
	return pool

def crossover(p1, p2, model):

	child = {}
	
	idx = random.randint(0, len(p1))

	params = list(p1.keys())

	for i, e in enumerate(params):
		if i < idx:
			child[e] = p1[e]
		else:
			child[e] = p2[e]

	# Perform Local Search

	child, fitness = LocalSearch.search(child)

	fitness = FitnessFunction.calculate_fitness(child, model)

	return child, fitness

def crossover_population(mating_pool, size, model):

	children = []
	length = len(mating_pool) - size
	pool = random.sample(mating_pool, len(mating_pool))

	for i in range(size):
		children.append(mating_pool[i])

	for i in range(length):
		child, fitness = crossover(pool[i][0], pool[len(mating_pool)-i-1][0], model)
		
		children.append((child, fitness))

	return children

def mutate(individual, options, search_params, model):
	
	params = list(options.keys())

	mutate_gene = random.choice(search_params)

	individual[mutate_gene] = random.choice(options[mutate_gene])

	fitness = FitnessFunction.calculate_fitness(individual, model)

	return individual, fitness

def mutate_population(population, options, search_params, model):
	mutated_population = []

	for i in range(len(population)):
		mutated_index = mutate(population[i][0], options, search_params, model)
		mutated_population.append(mutated_index)
	return mutated_population

def next_generation(current, size, options, strategy, search_params, model):

	pop_ranked = rank_individuals(current)

	results = []
	if strategy == "Tournament":
		results = TournamentSelection.select(pop_ranked, size)
	elif strategy == "Roulette Wheel":
		results = RouletteWheelSelection.select(pop_ranked, size)
	matingpool = mating_pool(current, results)
	children = crossover_population(matingpool, size, model)
	next_gen = mutate_population(children, options, search_params, model)
	return next_gen


	# plt.show()

def memetic(options, search_params, pop_size, selection_size, generations, strategy, model):

	pop = init_population(options, search_params, pop_size, model)

	params, fitness = rank_individuals(pop)[0]

	for i in range(generations):
		pop = next_generation(pop, selection_size, options, strategy, search_params, model)
		curr_params, curr_fitness = rank_individuals(pop)[0]

		if i == generations-1:
			fitness = curr_fitness
			params = curr_params

	return params, fitness 


class MemeticAlgorithm:

	def __init__(self):
		pass

	@staticmethod
	def execute(options, search_params, strategy, model):
		return memetic(options = options, search_params = search_params, pop_size=3, selection_size=2,  generations=10, strategy=strategy, model = model)

