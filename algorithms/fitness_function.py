import random
import sys
import os

sys.path.append(os.getcwd() + '/models')

from sentimentanalysis import SentimentAnalysisModel
from imageclassifier import ImageClassifier

class FitnessFunction:

	def __init__(self):
		pass

	@staticmethod
	def calculate_fitness(parameters, model, data):

		if model == "SentimentAnalysisModel":
			return SentimentAnalysisModel.build(parameters, data)
		elif model == "ImageClassifier":
			return ImageClassifier.build(parameters)

		return random.randrange(1, 100)