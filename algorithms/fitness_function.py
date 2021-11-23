import random

class FitnessFunction:

	def __init__(self):
		pass

	@staticmethod
	def calculate_fitness(parameters):
		return random.randrange(1, 100)