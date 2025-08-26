import numpy 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

class LogisticRegressionNN:
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
        # learning_rate = Step size for gradient descent algorithm
        # max_iterations = maximum number of iterations for training
        # tolerance = convergence tolerance / stop early

        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.weights = None
        self.bias = None
        self.cost_history = []

    