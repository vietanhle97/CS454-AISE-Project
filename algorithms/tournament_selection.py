import random

class TournamentSelection:

	def __init__(self):
		pass

	@staticmethod
	def select(pop_ranked, tournament_size):
		tmp = {}

		for i in range(len(pop_ranked)):
			tmp[i] = pop_ranked[i]
		results = []

		for i in range(len(tmp)):
			tournament = [] * tournament_size
			for j in range(tournament_size):
				random_index = int(random.random() * len(tmp))
				tournament.append((random_index, tmp[random_index]))
			best = 0
			for i in range(len(tournament)):
				if tournament[i][1][1] > tournament[best][1][1]:
					best = i
			results.append(tournament[best][0])
		return results
