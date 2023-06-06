# Perceptron algorithm with a binary approach
# Each price category is treated as one class and all other categories are treated as another class
# TODO should this class contain a method to print the predictions to a file?
import random
import math


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
        TODO random weight initialization
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


if __name__ == "__main__":
    # very fake data just to test if the perceptron runs without bugs
    fake_corpus = [([random.randint(0, 10) for _ in range(100)], random.randint(1, 4)) for _ in range(50)]
    N = len(fake_corpus[0][0])  # number of features
    EPOCHS = 10
    test1 = Perceptron(N, 1)  # predict label 1, default learning rate
    test2 = Perceptron(N, 2)
    test3 = Perceptron(N, 3)
    test4 = Perceptron(N, 4)

    # training loop
    def training(model, train_data, epochs):
        for epoch in range(1, epochs + 1):
            # shuffle the training data every epoch
            random.shuffle(train_data)
            for feats, label in train_data:
                model.update(feats, label)

    # softmax function converts raw scores to a probability distribution
    def softmax(z: list) -> list:
        out = []
        max_num = max(z)
        z = [y - max_num for y in z]  # clip the numbers to avoid numerical overflow
        for y in z:
            out.append(math.exp(y) / sum([math.exp(y_i) for y_i in z]))
        return out

    # train a model for each class
    training(test1, fake_corpus, EPOCHS)
    training(test2, fake_corpus, EPOCHS)
    training(test3, fake_corpus, EPOCHS)
    training(test4, fake_corpus, EPOCHS)

    sample = fake_corpus[0]
    predictions = [test1.predict(fake_corpus[0][0], activate=False),
                   test2.predict(fake_corpus[0][0], activate=False),
                   test3.predict(fake_corpus[0][0], activate=False),
                   test4.predict(fake_corpus[0][0], activate=False)]
    print(f'True label: {sample[1]}')
    print(f'Predictions: {predictions}')
    print(f'Normalized predictions: {softmax(predictions)}')
