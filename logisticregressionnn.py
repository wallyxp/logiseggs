import numpy as np 
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

    def sigmoid(self, z):
        # sigmoid function with value stability
        z = np.clip(z,-500,500)
        return 1/(1+np.exp(-z))

    def initialize_parameters(self, n_features):
        # initializing the weights and bias
        self.weights = np.random.randn(n_features) * np.sqrt(1/n_features)
        self.bias = 0.00
    

    def forward_propagation(self, X):
        # forward propagation -> X : input features of m x n 
        # m samples containing n features
        # the function will return : 
        #   Z -> linear combination before activation
        #   A -> sigmoid(Z) : linear combination after activation

        Z = np.dot(X, self.weights) + self.bias
        A = self.sigmoid(Z)

        return A, Z
    
    def compute_cost(self, A, y):

        m =y.shape[0]

        epsilon = 1e-15
        A = np.clip(A, epsilon, 1 - epsilon)

        cost = -(1/m) * np.sum(y * np.log(A) + (1-y) * np.log(1-A))

        return cost

    def backward_propagation(self, X, A, y):
        pass