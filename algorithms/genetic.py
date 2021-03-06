import sys
import os
sys.path.append(os.getcwd() + '/..')

from fitness_function import FitnessFunction
import random
from roulette_wheel_selection import RouletteWheelSelection
from tournament_selection import TournamentSelection
from local_search import LocalSearch

def is_exist(pop, new_individual, search_params):

	if len(pop) == 0:
		return False

	for individual in pop:

		curr = individual[0]

		for k in search_params:
			if curr[k] != new_individual[k]:
				return False
	return True

def find_int(l):

	for i, e in enumerate(l):
		try:
			tmp = int(e)
			return i
		except:
			continue

def create_new_name(curr_name, is_mutate, generation, index):

	curr_model_name = curr_name.split("-")
	idx = find_int(curr_model_name)
	if is_mutate:
		return "-".join(curr_model_name[:idx]) + "-m-" + "-".join(curr_model_name[idx:])
	return "-".join(curr_model_name[:3]) + "-" + str(generation) + "-" + str(index)
	


def create_first_generation(options, search_params):

	first_gen = {}

	for k in options:
		if k in search_params:
			first_gen[k] = random.choice(options[k])
		else:
			first_gen[k] = options[k]

	return first_gen

def init_population(data, options, search_params, size, model, strategy):
	population = []
	for i in range(size):
		individual = create_first_generation(options, search_params)

		while is_exist(population, individual, search_params):
			individual = create_first_generation(options, search_params)

		if model == "SentimentAnalysisModel":
			if strategy == "Tournament":
				individual["model_name"] = "ga-sa-tournament-1" + "-" + str(i+1)
			else:
				individual["model_name"] = "ga-sa-roulette-1" + "-" + str(i+1)
		else:
			if strategy == "Tournament":
				individual["model_name"] = "ga-ic-tournament-1" + "-" + str(i+1)
			else:
				individual["model_name"] = "ga-ic-roulette-1" + "-" + str(i+1)

		fitness = FitnessFunction.calculate_fitness(individual, model, data)
		population.append((individual, fitness))

	return population

def rank_individuals(population):

	return sorted(population, key = lambda item: item[1], reverse = True)

def mating_pool(population, selection):
	pool = []
	for i in range(len(selection)):
		pool.append(population[selection[i]])
	return pool

def crossover(p1, p2, model, data, search_params, generation, index):

	child = {}
	
	idx = random.randint(0, len(search_params)-1) # avoid duplication

	for k in p1.keys():
		if k not in search_params:
			if k == "model_name":
				child[k] = create_new_name(p1["model_name"], is_mutate=False, generation=generation, index=index)
			else:
				child[k] = p1[k]

	for i, e in enumerate(search_params):
		if i < idx:
			child[e] = p1[e]
		else:
			child[e] = p2[e]

	fitness = FitnessFunction.calculate_fitness(child, model, data)

	return child, fitness

def crossover_population(mating_pool, size, model, data, search_params, generation):

	children = []
	length = len(mating_pool) - size
	pool = random.sample(mating_pool, len(mating_pool))



	for i in range(size):
		children.append(mating_pool[i])

	for idx in range(1, length+1):
		child, fitness = crossover(pool[i][0], pool[len(mating_pool)-i-1][0], model, data, search_params, generation, idx)
		
		children.append((child, fitness))

	return children

def mutate(data, individual, options, search_params, model):
	
	params = list(options.keys())

	mutate_gene = random.choice(search_params)

	new_gene = random.choice(options[mutate_gene])

	# Avoid duplication

	while new_gene == individual[mutate_gene]:

		new_gene = random.choice(options[mutate_gene])

	individual[mutate_gene] = new_gene


	individual["model_name"] = create_new_name(individual["model_name"], is_mutate=True, generation=None, index=None)

	fitness = FitnessFunction.calculate_fitness(individual, model, data)

	return individual, fitness

def mutate_population(data, population, options, search_params, model, mutate_rate):
	mutated_population = []

	to_be_mutated_individuals = random.sample([i for i in range(len(population))], int(len(population) * mutate_rate))

	for i in range(len(population)):
		if i in to_be_mutated_individuals:
			mutated_individual = mutate(data, population[i][0], options, search_params, model)
			mutated_population.append(mutated_individual)
		else:
			mutated_population.append(population[i])

	return mutated_population

def next_generation(data, current, size, options, strategy, search_params, model, mutate_rate, generation):

	pop_ranked = rank_individuals(current)

	results = []
	if strategy == "Tournament":
		results = TournamentSelection.select(pop_ranked, size)
	elif strategy == "Roulette Wheel":
		results = RouletteWheelSelection.select(pop_ranked, size)
	matingpool = mating_pool(current, results)
	children = crossover_population(matingpool, size, model, data, search_params, generation)
	next_gen = mutate_population(data, children, options, search_params, model, mutate_rate)
	return next_gen


def genetic(data, options, search_params, pop_size, selection_size, generations, strategy, model, mutate_rate):

	pop = init_population(data, options, search_params, pop_size, model, strategy)

	params, fitness = rank_individuals(pop)[0]

	print("fitness result: " + str(fitness))

	for generation in range(2, generations+2):
		pop = next_generation(data, pop, selection_size, options, strategy, search_params, model, mutate_rate, generation)
		curr_params, curr_fitness = rank_individuals(pop)[0]

		print("fitness result: " + str(curr_fitness))

		if curr_fitness > fitness:
			fitness = curr_fitness
			params = curr_params

	return params, fitness 


class GeneticAlgorithm:

	def __init__(self):
		pass

	@staticmethod
	def execute(data, options, generations, pop_size, selection_size, search_params, strategy, model, mutate_rate):

		return genetic(data = data, options = options, search_params = search_params, pop_size=pop_size, selection_size=selection_size,  generations=generations, strategy=strategy, model = model, mutate_rate = mutate_rate)

