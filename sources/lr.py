#linear regression class using python
import numpy as np 

class LinearRegression:
    def __init__(self, learning_rate: float = 0.01, iterations: int = 1000) -> None:
        """
        Initializes the LinearRegression model with learning rate and number of iterations.

        :param learning_rate: Step size for gradient descent, default is 0.01 
        :param iterations: Number of iterations for the optimization loop, default is 1000
        """
        self.lr = learning_rate 
        self.n_iters = iterations 
        self.weights, self.bias = None, None 

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fits the LinearRegression model to the training data using gradient descent.

        :param X: Input features, shape (n_samples, n_features)
        :param y: Target values, shape (n_samples,)
        """
        n_samples, n_features = X.shape
        # self.weights = np.random.rand(n_features)
        self.weights = np.zeros(n_features)
        self.bias = 0 
        
        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias
            error = y_pred - y
            
            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, error)
            db = (1 / n_samples) * np.sum(error)
            
            # Update weights and bias
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the target values for a given set of input features.

        :param X: Input features, shape (n_samples, n_features)
        :return: Predicted values, shape (n_samples,)
        """
        return np.dot(X, self.weights) + self.bias 

# Example usage:
# lr_model = LinearRegression()
# lr_model.fit(X_train, y_train)
# y_pred = lr_model.predict(X_test)

