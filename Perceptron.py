# basic    Perceptron algorithm with a binary approach
# Each price category is treated as one class and all other categories are treated as another class

class Perceptron:
    def __init__(self, num_features):
        # Initialize weights and bias
        self.weights = [0] * num_features
        self.bias = 0

    def predict(self, features):
        # Compute the dot product of features and weights, and add bias
        score = sum(x*w for x, w in zip(features, self.weights)) + self.bias
        # Return 1 if score is positive, else 0
        return 1 if score > 0 else 0

    def update(self, features, y_true):
        # Compute the prediction
        y_pred = self.predict(features)
        # Update weights and bias if prediction is incorrect
        if y_pred != y_true:
            for i in range(len(self.weights)):
                self.weights[i] += (y_true - y_pred) * features[i]
            self.bias += (y_true - y_pred)
            