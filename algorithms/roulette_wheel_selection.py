import random

class RouletteWheelSelection:

	def __init__(self):
		pass

	@staticmethod
	def select(pop_ranked, roulette_wheel_size):
		results = []
		total = sum(j for i, j in pop_ranked)
		cumulative_sum = [0] * len(pop_ranked)
		cumulative_sum_percentage = [0] * len(pop_ranked)

		for i, e in enumerate(pop_ranked):
			if i - 1 >= 0:
				cumulative_sum[i] = e[1] + cumulative_sum[i-1]
			else:
				cumulative_sum[i] = e[1]

			cumulative_sum_percentage[i] = 100*cumulative_sum[i]/total

		for i in range(roulette_wheel_size):
			results.append(i)
		for i in range(len(pop_ranked) - roulette_wheel_size):
			pick = 100*random.random()
			for i in range(0, len(pop_ranked)):
				if pick <= cumulative_sum_percentage[i]:
					results.append(i)
					break
		return results