import numpy as np


class Perceptron:
    def __init__(self, n_inputs, learning_rate):
        """
        :param n_inputs: the number of inputs of the perceptron
        :param learning_rate: the learning rate
        """
        self.n_inputs = n_inputs
        self.learning_rate = learning_rate

        # Will be defiend after training
        self.threshold = None
        self.weights = None

        self._weights = np.zeros(n_inputs + 1)  # +1 for the threshold

    def fit(self, X, Y):
        """
        Train the perceptron against the data
        """
        X = np.insert(X, X.shape[1], 1, axis=1)  # We insert a fake feature always equal to one for the threshold
        while self.error(X, Y) != 0:  # We want the error to be 0. Only works we linearly separable data
            for x, y in zip(X, Y):
                predicted_y = self._predict([x])[0]
                for i, (xi, wi) in enumerate(zip(x, self._weights)):
                    if predicted_y == y:
                        continue
                    diff_weight = self.learning_rate * (y - predicted_y) * xi
                    self._weights[i] += diff_weight

        # The training is done, extract the threshold from the weights
        self.threshold = -self._weights[-1]
        self.weights = self._weights[:-1]

    def error(self, test_X, test_Y):
        """
        Return the sum of squared errors (SSE) of the perceptron
        :return: the SSE
        """
        predicted_Y = self._predict(test_X)
        return np.linalg.norm(predicted_Y - test_Y)

    def _predict(self, X):
        """
        Predict without explicit threshold (the threshold is embedded in the weights)
        """
        return [1 if np.dot(x, self._weights) >= 0 else 0 for x in X]

    def predict(self, X):
        """
        Predict after the percreptron has been trained (when threshold is known)
        """
        return [1 if np.dot(x, self.weights) >= self.threshold else 0 for x in X]


# We generate training data for OR perceptron
train_X = np.array([
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1],
])
train_Y = np.array([
    0,
    1,
    1,
    1,
])

p = Perceptron(n_inputs=2, learning_rate=0.1)
p.fit(train_X, train_Y)
print("Weights: {}\nThreshold: {}".format(p.weights, p.threshold))
print(p.predict(train_X))
