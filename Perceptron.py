# basic    Perceptron algorithm with a binary approach
# Each price category is treated as one class and all other categories are treated as another class
# TODO perceptron should receive already decoded features, probably implement decoding in Corpus
# TODO handle OOV tokens
# TODO note: no need to binarize

class Perceptron:
    def __init__(self, num_features, tar_label, lr=0.1):
        # Initialize weights and bias
        self.weights = self._initialize_weights(num_features)
        self.bias = 0
        self.LR = lr  # learning rate
        self.tar_label = tar_label  # target label that this perceptron is be trained to predict

    def predict(self, features, activate=True):
        """
        TODO docs
        :param features:
        :param activate:
        :return:
        """
        # Compute the dot product of features and weights, and add bias
        score = sum(x*w for x, w in zip(features, self.weights)) + self.bias
        return self._activation(score) if activate else score

    def update(self, features, y_true):
        """
        TODO docs
        :param features:
        :param y_true: True label in the original format (not binarized)
        :return:
        """
        # Compute the prediction
        y_pred = self.predict(features, activate=True)
        y_true = self._binarize_label(y_true)
        # Update weights and bias if prediction is incorrect
        if y_pred != y_true:
            for i in range(len(self.weights)):
                self.weights[i] += self._error(y_true, y_pred) * features[i]
            self.bias += self._error(y_true, y_pred)

    def _activation(self, score):
        """
        Return 1 if score is positive, else 0
        :param score: TODO docs
        :return:
        """
        return 1 if score > 0 else 0

    def _error(self, y_true, y_pred):
        """
        TODO docs
        :param y_true:
        :param y_pred:
        :return:
        """
        return y_true - y_pred

    def _initialize_weights(self, num_features):
        """
        TODO random weight initialization
        :param num_features: TODO docs
        :return:
        """
        return [0] * num_features

    def _binarize_label(self, label):
        """
        TODO docs
        :param label:
        :return:
        """
        return label == self.tar_label


if __name__ == "__main__":
    N = 3
    test = Perceptron(N, 2)
