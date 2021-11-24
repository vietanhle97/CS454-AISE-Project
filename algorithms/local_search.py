import random
import copy

from fitness_function import FitnessFunction

class LocalSearch:

	def __init__(self):
		pass

	@staticmethod
	def search(individual, options, search_params, model):

		result_individual, result_fitness = individual[0], individual[1]

		search_param = random.choice(search_params)

		values = options[search_param]

		if type(values[0]) != int and type(values[0]) !=float:
			for i in range(100):
				search_param = random.choice(search_params)
				values = options[search_param]

				if type(values[0]) == int or type(values[0]) ==float:
					break

		if type(values[0]) != int and type(values[0]) !=float:
			return individual


		curr_idx = options[search_param].index(individual[0][search_param])

		neighbor_1 = copy.deepcopy(individual[0])

		neighbor_1[search_param] = options[search_param][(curr_idx + 1 + len(values))%len(values)]

		neighbor_2 = copy.deepcopy(individual[0])

		neighbor_2[search_param] = options[search_param][(curr_idx + 1 + len(values))%len(values)]

		fitness_1 = FitnessFunction.calculate_fitness(neighbor_1, model)

		if fitness_1 > result_fitness:
			result_individual = neighbor_1
			result_fitness = fitness_1

		fitness_2 = FitnessFunction.calculate_fitness(neighbor_2, model)

		if fitness_2 > result_fitness:
			result_individual = neighbor_2
			result_fitness = fitness_2

		return result_individual, result_fitness