import sys

import os


sys.path.append(os.getcwd() + '/algorithms')

sys.path.append(os.getcwd() + '/models')

from sentimentanalysis import SentimentAnalysisModel

from imageclassifier import ImageClassifier

from memetic import MemeticAlgorithm

from genetic import GeneticAlgorithm


def optimize(data, options, algorithm="GeneticAlgorithm", selection="Tournament", model="SentimentAnalysisModel", mutate_rate=0.4, num_local_search=3):

	search_params = [k for k in options if type(options[k]) == list]

	result = 0

	if algorithm == "GeneticAlgorithm":

		return GeneticAlgorithm.execute(data, options, search_params, selection, model, mutate_rate)

	elif algorithm == "MemeticAlgorithm":

		return MemeticAlgorithm.execute(data, options, search_params, selection, model, mutate_rate, num_local_search)

	raise NotImplementedError



def run(options, model, data):

	genetic_tournament_params,  genetic_tournament_fitness = optimize(data, options, "GeneticAlgorithm", "Tournament", model, 0.4, 0)

	print("The optimal parameters of " + model + " using Genetic Algorithm with Tournament Selection are: " )

	for i in genetic_tournament_params:
		print(i + ": " + str(genetic_tournament_params[i]))

	print("The accuracy of the " + model + " is: " + str(genetic_tournament_fitness) + "\n")

	genetic_roullete_params, genetic_roullete_fitness = optimize(data, options, "GeneticAlgorithm", "Tournament", model, 0.4, 0)

	print("The optimal parameters of " + model + " using Genetic Algorithm with Roulette Wheel Selection are: " )

	for i in genetic_roullete_params:
		print(i + ": " + str(genetic_tournament_params[i]))

	print("The accuracy of the " + model + " is: " + str(genetic_roullete_fitness) + "\n")


	memetic_tournament_params, memetic_tournament_fitness = optimize(data, options, "MemeticAlgorithm", "Tournament", model, 0.4, 3)

	print("The optimal parameters of " + model + " using Memetic Algorithm with Tournament are: " )

	for i in memetic_tournament_params:
		print(i + ": " + str(memetic_tournament_params[i]))

	print("The accuracy of the " + model + " is: " + str(memetic_tournament_fitness) + "\n")

	memetic_roullete_params, memetic_roullete_fitness = optimize(data, options, "MemeticAlgorithm", "Roulette Wheel", model, 0.4, 3)

	print("The optimal parameters of " + model + " using Memetic Algorithm with Roulette Wheel Selection are: " )

	for i in memetic_roullete_params:
		print(i + ": " + str(memetic_roullete_params[i]))

	print("The accuracy of the " + model + " is: " + str(memetic_roullete_fitness) + "\n")



if __name__ == '__main__':

	sentimentanalysis_options = {
		'model_name': 'sa-1-1', # this is just identifier first '1' means generation and second '1' is just id

		'embedding_dim': [8, 16, 32, 64, 128, 256, 512, 1024],
		# please set 1024 to be the maximum value
		'rnn_hidden_dim': [8, 16, 32, 64, 128, 256, 512, 1024], # can be modified
		# 
		# please set 4096 to be the maximum value
		'rnn_num_layers': [1,2,3,4,5,6,7,8], # can be modified

		# please set 8 to be the maximum value
		'rnn_dropout': [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85], # can be modified

		# please set 0.85 to be the maximum value
		'rnn_bidirectional': [True, False], # can be modified
		# 
		'fc_hidden_dim' : [8, 16, 32, 64, 128, 256, 512, 1024], # can be modified

		# please set 4096 to be the maximum value
		'fc_num_layers' : [1,2,3,4,5,6,7,8], # can be modified
		# 
		# please set 8 to be the maximum value
		'fc_dropout' : [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85], # can be modified
		# 
		# please set 0.85 to be the maximum value
		'learning_rate': [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001], # can be modified
		# 
		# please set 0.01 to be the maximum value and 0.00001 to the minimum value
		'batch_size': 32,

		'num_epochs': 10, 

		'device':'cuda'

	}


	imageclassifier_options = {
		'model_name': 'sa-1-1', # this is just identifier first '1' means generation and second '1' is just id

		"conv_1_out_channels": [8, 16, 32, 64, 128, 256, 512, 1024],

		"conv_1_bias": [True, False],

		"conv_2_out_channels": [8, 16, 32, 64, 128, 256, 512, 1024],

		"conv_2_bias": [True, False],

		"conv_dropout": [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85],

		"fc_hidden_dim": [8, 16, 32, 64, 128, 256, 512, 1024],

		"fc_num_layers": [1,2,3,4,5,6,7,8],

		"fc_dropout": [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85],

		"learning_rate": [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001], 
		# ============= do not modify the below hyperparameter
		"batch_size": 8, 

		"num_epochs": 10, 

		'device':'cuda'
	}

	sentimentanalysis_data = SentimentAnalysisModel.load_data()

	run(sentimentanalysis_options, "SentimentAnalysisModel", sentimentanalysis_data)


	imageclassifier_data = ImageClassifier.load_data()

	run(imageclassifier_options, "ImageClassifier", imageclassifier_data)












