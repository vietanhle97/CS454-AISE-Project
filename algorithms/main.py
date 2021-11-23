from memetic import MemeticAlgorithm


if __name__ == '__main__':
	
	options = {
		'model_name': 'sa-1-1', # this is just identifier first '1' means generation and second '1' is just id

		'embedding_dim': [8, 16, 32, 64, 128, 256, 512, 1024],
		# please set 1024 to be the maximum value
		'rnn_hidden_dim': [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096], # can be modified
		# 
		# please set 4096 to be the maximum value
		'rnn_num_layers': [1,2,3,4,5,6,7,8], # can be modified

		# please set 8 to be the maximum value
		'rnn_dropout': [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85], # can be modified

		# please set 0.85 to be the maximum value
		'rnn_bidirectional': [True, False], # can be modified
		# 
		'fc_hidden_dim' : [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096], # can be modified

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

		'num_epochs': 1, 

		'device':'cuda'

	}


	search_params = [k for k in options if type(options[k]) == list]

	res = MemeticAlgorithm.execute(options, search_params, "Tournament", [])

	print(res)