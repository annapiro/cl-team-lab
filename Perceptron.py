from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
import numpy as np

class MultiClassPerceptron:
    def __init__(self):
        self.scaler = StandardScaler()
        self.clf = SGDClassifier(loss='perceptron', eta0=1, learning_rate='constant', penalty='l1')

    def train(self, train_data):
        # Prepare the training data
        X_train = [data[0] for data in train_data]
        y_train = [data[1] for data in train_data]

        # Scale the features
        X_train = self.scaler.fit_transform(X_train)

        # Train the classifier
        self.clf.partial_fit(X_train, y_train, classes=np.arange(1, 5))

    def predict(self, features):
        # Scale the features using the same scaler used in training
        features = self.scaler.transform([features])
        
        return self.clf.predict(features)[0]
