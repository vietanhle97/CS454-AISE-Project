import sys

import os


sys.path.append(os.getcwd() + '/algorithms')

sys.path.append(os.getcwd() + '/models')

from sentimentanalysis import SentimentAnalysisModel

from imageclassifier import ImageClassifier

from memetic import MemeticAlgorithm

from genetic import GeneticAlgorithm

def optimize(data, options, algorithm, strategy, model, generations, pop_size, selection_size, mutate_rate, num_local_search):

	search_params = [k for k in options if type(options[k]) == list]

	result = 0

	if algorithm == "GeneticAlgorithm":

		return GeneticAlgorithm.execute(data, options, generations, pop_size, selection_size, search_params, strategy, model, mutate_rate)

	elif algorithm == "MemeticAlgorithm":

		return MemeticAlgorithm.execute(data, options, generations, pop_size, selection_size, search_params, strategy, model, mutate_rate, num_local_search)

	raise NotImplementedError



def run(options, model, data, generations, pop_size, selection_size, mutate_rate, num_local_search):

	genetic_tournament_params,  genetic_tournament_fitness = optimize(data, options, "GeneticAlgorithm", "Tournament", model, generations, pop_size, selection_size, mutate_rate, num_local_search)

	f = open("./result/" + model.lower() + "_genetic_tournament.txt", "w")

	f.write("The optimal parameters of " + model + " using Genetic Algorithm with Tournament Selection are: \n" )

	for i in genetic_tournament_params:
		f.write("    " + i + ": " + str(genetic_tournament_params[i]) + "\n")

	f.write("The accuracy of the " + model + " is: " + str(genetic_tournament_fitness) + "\n")

	f.close()

	genetic_roullete_params, genetic_roullete_fitness = optimize(data, options, "GeneticAlgorithm", "Roulette Wheel", model, generations, pop_size, selection_size, mutate_rate, num_local_search)

	f = open("./result/" + model.lower() + "_genetic_roullete.txt", "w")

	f.write("The optimal parameters of " + model + " using Genetic Algorithm with Roulette Wheel Selection are: \n" )

	for i in genetic_roullete_params:
		f.write("    " + i + ": " + str(genetic_roullete_params[i]) + "\n")

	f.write("The accuracy of the " + model + " is: " + str(genetic_roullete_fitness) + "\n")

	f.close()

	memetic_tournament_params, memetic_tournament_fitness = optimize(data, options, "MemeticAlgorithm", "Tournament", model, generations, pop_size, selection_size, mutate_rate, num_local_search)


	f = open("./result/" + model.lower() + "_memetic_tournament.txt", "w")

	f.write("The optimal parameters of " + model + " using Memetic Algorithm with Tournament Selection are: \n" )

	for i in memetic_tournament_params:
		f.write("    " + i + ": " + str(memetic_tournament_params[i]) + "\n")

	f.write("The accuracy of the " + model + " is: " + str(memetic_tournament_fitness) + "\n")

	f.close()

	memetic_roullete_params, memetic_roullete_fitness = optimize(data, options, "MemeticAlgorithm", "Roulette Wheel", model, generations, pop_size, selection_size, mutate_rate, num_local_search)

	f = open("./result/" + model.lower() + "_memetic_roullete.txt", "w")

	f.write("The optimal parameters of " + model + " using Memetic Algorithm with Roulette Wheel Selection are: \n" )

	for i in memetic_roullete_params:
		f.write("    " + i + ": " + str(memetic_roullete_params[i]) + "\n")

	f.write("The accuracy of the " + model + " is: " + str(memetic_roullete_fitness) + "\n")


	f.close()



if __name__ == '__main__':

	if not os.path.exists(os.getcwd()+"/result"):
		os.makedirs(os.getcwd()+"/result")

	sentimentanalysis_options = {

		'embedding_dim': [8, 16, 32, 64, 128, 256, 512, 1024],

		'rnn_hidden_dim': [8, 16, 32, 64, 128, 256, 512, 1024],

		'rnn_num_layers': [1,2,3,4,5,6,7,8],

		'rnn_dropout': [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85],

		'rnn_bidirectional': [True, False], 
	
		'fc_hidden_dim' : [8, 16, 32, 64, 128, 256, 512, 1024], 

		'fc_num_layers' : [1,2,3,4,5,6,7,8], 
		
		'fc_dropout' : [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85], 
		
		'learning_rate': [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001],
		
		'batch_size': 32,

		'num_epochs': 10, 

		'device':'cuda'

	}


	imageclassifier_options = {

		"conv_1_out_channels": [8, 16, 32, 64, 128, 256, 512, 1024],

		"conv_1_bias": [True, False],

		"conv_2_out_channels": [8, 16, 32, 64, 128, 256, 512, 1024],

		"conv_2_bias": [True, False],

		"conv_dropout": [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85],

		"fc_hidden_dim": [8, 16, 32, 64, 128, 256, 512, 1024],

		"fc_num_layers": [1,2,3,4,5,6,7,8],

		"fc_dropout": [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85],

		"learning_rate": [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001],
		
		"batch_size": 8, 

		"num_epochs": 10, 

		'device':'cuda'
	}

	# config manually
	generations = 10 
	pop_size = 4 
	selection_size = 2 
	mutate_rate = 0.5
	num_local_search = 3

	sentimentanalysis_data = SentimentAnalysisModel.load_data()

	run(sentimentanalysis_options, "SentimentAnalysisModel", data, generations, pop_size, selection_size, mutate_rate, num_local_search)


	imageclassifier_data = ImageClassifier.load_data()

	run(imageclassifier_options, "ImageClassifier", data, generations, pop_size, selection_size, mutate_rate, num_local_search)












