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
        # epsilon clipping is needed to prevent the calculation of log(0)
        A = np.clip(A, epsilon, 1 - epsilon)
        cost = -(1/m) * np.sum(y * np.log(A) + (1-y) * np.log(1-A))

        return cost

    def backward_propagation(self, X, A, y):
        # what we have
        # X : input features
        # A : predicted probabilities
        # y : true labels
        
        # what we want to calculate
        # dw : gradient of the weights
        # db : gradient of the bias

        m = X.shape[0]

        # computing the gradients
        dz = A - y
        dw = (1/m) * np.dot(X.T, dz)
        db = (1/m) * np.sum(dz)

        return dw, db

    def fit(self, X, y, verbose=False):
        # Train the logistic regression model
        n_features = X.shape[1]
        self.initialize_parameters(n_features)

        #Training loop
        for i in range(n_features):
            A,Z = self.forward_propagation(X)

            # cost 
            cost = self.compute_cost(A,y)
            self.cost_history.append(cost)

            # backward_propagation
            dw, db = self.backward_propagation(X, A, y)

            # update parameters 
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db 

            # check for convergence
            if i > 0 and abs(self.cost_history[-2] - self.cost_history[-1]) < self.tolerance:
                if verbose:
                    print(f"Converged at iteration {i}")
                break

            if verbose and i % 100 == 0:
                print(f"Iteration {i}, cost : {cost:.6f}")

    def predict_proba(self, X):
        # predict probabilities for input samples
        A, _ = self.forward_propagation(X)
        return A
    
    def predict(self, X, threshold=0.5):
        # make binary predictions
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)

    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
