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
	def calculate_fitness(parameters, model):

		if model == "SentimentAnalysisModel":
			return SentimentAnalysisModel.build()
		elif model == "ImageClassifier":
			return ImageClassifier.build()

		return random.randrange(1, 100)