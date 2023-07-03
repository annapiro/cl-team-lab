from Perceptron import MultiClassPerceptron
from Corpus import Corpus
from Evaluator import Evaluator
from tqdm import tqdm
import numpy as np
import random

if __name__ == "__main__":
    EPOCHS = 2  # Define the number of training iterations

    # Load corpus
    data = Corpus.read_file("data/menu_sample.txt")
    dev = Corpus.read_file("data/menu_sample.txt")

    corpus = Corpus(data, test_data=dev, exclude_feats=[])

    # Initialize multi-class perceptron
    perceptron = MultiClassPerceptron()

    # Prepare the training data
    train_data = [(corpus.get_dense_features(restaurant), restaurant.gold_label) for restaurant in corpus.train_data]
    
    # Train the perceptron
    perceptron.train(train_data)

    # Make predictions
    for restaurant in corpus.test_data:
        dense_features = corpus.get_dense_features(restaurant)
        predicted_class = perceptron.predict(dense_features)
        # Set the predicted label
        restaurant.set_predicted_label(predicted_class)

    # Cross-validation
    K = 5  # number of splits for cross-validation
    fold_size = len(data) // K
    f1_scores = []
    correlations = []

    random.shuffle(data)

    for i in range(K):
        dev = data[i*fold_size: (i+1)*fold_size]
        train = data[:i*fold_size] + data[(i+1)*fold_size:]
        corpus = Corpus(train, exclude_feats=['name'])
        corpus.set_test_data(dev)

        # Get the number of features
        num_features = corpus.num_feats

        # Initialize perceptron for each class
        perceptron = MultiClassPerceptron() 

        # Train perceptrons
        train_data = [(corpus.get_dense_features(restaurant), restaurant.gold_label) for restaurant in corpus.train_data]
        perceptron.train(train_data)

        # Make predictions
        for restaurant in corpus.test_data:
            dense_features = corpus.get_dense_features(restaurant)
            predicted_class = perceptron.predict(dense_features)
            # Set the predicted label
            restaurant.set_predicted_label(predicted_class)

    # For testing if the labels are assigned correctly
    corpus.print_labels()

    # Set up evaluator
    evaluator = Evaluator(corpus)
    
    if len(set(y_true)) == 1 or len(set(y_pred)) == 1:
        print("Insufficient variation in the data to calculate correlation.")
    else:
        correlation = evaluator.evaluate_correlation()
        correlations.append(correlation)

    f1_score = evaluator.evaluate_f1_score()
    f1_scores.append(f1_score)

average_f1_score = sum(f1_scores) / len(f1_scores)
average_correlation = sum(correlations) / len(correlations)

print(f"Average F1 Score across all folds: {average_f1_score:.2f}")
print(f"Average Correlation across all folds: {average_correlation:.2f}")
