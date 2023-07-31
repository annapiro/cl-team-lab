# Perceptron algorithm with a binary approach
# Each price category is treated as one class and all other categories are treated as another class
import random


class Perceptron:
    def __init__(self, num_features: int, tar_label: int, lr: float = 0.1):
        # Initialize weights and bias
        self.weights = self._initialize_weights(num_features)
        self.bias = 0
        self.LR = lr  # learning rate
        self.tar_label = tar_label  # target label that this perceptron is trained to predict

    def train(self, train_data: list, epochs: int):
        """
        Trains the perceptron for a specified number of epochs
        :param train_data: Training data
        :param epochs: Number of epochs
        """
        for epoch in range(1, epochs + 1):
            random.shuffle(train_data)
            for feats, label in train_data:
                self.update(feats, label)

    def predict(self, features: list, activate=False) -> int | float:
        """
        Predicts the class of a single instance based on its features
        :param features: List of encoded features
        :param activate: Whether the output should be activated
        :return: Binary result (1 or 0) if activated, raw score otherwise
        """
        # Compute the dot product of features and weights, and add bias
        score = sum(x*w for x, w in zip(features, self.weights)) + self.bias
        return self._activation(score) if activate else score

    def update(self, features: list, y_true: int):
        """
        Updates the weights based on the prediction for one instance
        :param features: List of encoded features
        :param y_true: True label in the original format (not binarized)
        """
        # Compute the prediction
        y_pred = self.predict(features, activate=True)
        y_true = self._binarize_label(y_true)
        # Update weights and bias if prediction is incorrect
        if y_pred != y_true:
            error = self._error(y_true, y_pred)
            for i in range(len(self.weights)):
                self.weights[i] += error * features[i]
            self.bias += error

    def _activation(self, score: float) -> int:
        """
        Activation function that converts raw score to class prediction
        :param score: Raw prediction score
        :return: 1 if score is positive, else 0
        """
        return 1 if score > 0 else 0

    def _error(self, y_true: int | float, y_pred: int | float) -> int | float:
        """
        Error function calculates the error between true label and predicted label/score
        :param y_true: True label
        :param y_pred: Predicted label or score
        :return: Error
        """
        return y_true - y_pred

    def _initialize_weights(self, num_features: int) -> list:
        """
        :param num_features: Number of features per instance
        :return: List of initialized weights
        """
        return [0] * num_features

    def _binarize_label(self, label: int) -> int:
        """
        Convert class label from multiclass representation to a binary label,
        in accordance with the class that this perceptron is trained to predict
        :param label: Original label
        :return: Binary label
        """
        return int(label == self.tar_label)
