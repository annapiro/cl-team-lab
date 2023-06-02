import itertools
import random
from Perceptron import Perceptron
from Corpus import Corpus
from Restaurant import Restaurant
from Evaluator import Evaluator
from tqdm import tqdm

if __name__ == "__main__":
    EPOCHS = 2  # Define the number of training iterations

    # Load corpus
    data = Corpus.read_file("data/menu_train_500.txt")
    # Cross-validation
    K = 5  # number of splits for cross-validation
    fold_size = len(data) // K
    f1_scores = []
    correlations = []

    for i in range(K):
        dev = data[i*fold_size: (i+1)*fold_size]
        train = data[:i*fold_size] + data[(i+1)*fold_size:]
        corpus = Corpus(train, exclude_feats=['name'])
        corpus.set_test_data(dev)

        # Get the number of features
        num_features = corpus.num_feats

        # Initialize perceptron for each class
        perceptrons = [Perceptron(num_features, i) for i in range(1, 5)]  # Assuming 4 price categories

        # Train perceptrons
        train_data = [(corpus.get_dense_features(restaurant), restaurant.gold_label) for restaurant in corpus.train_data]
        for perceptron in tqdm(perceptrons):
            perceptron.train(train_data, EPOCHS)

        # Make predictions
        for restaurant in corpus.test_data:
            dense_features = corpus.get_dense_features(restaurant)
            predictions = [perceptron.predict(dense_features) for perceptron in perceptrons]
            # Get the predicted class (1-indexed)
            predicted_class = predictions.index(max(predictions)) + 1
            # Set the predicted label
            restaurant.set_predicted_label(predicted_class)

    # Evaluate perceptrons
    y_true = []
    y_pred = []
    for restaurant in corpus.test_data:
        # Check if gold_label and predicted_label are not None
        if restaurant.gold_label is None:
            print(f"Gold label for restaurant {restaurant.name} is None.")
            continue
        if restaurant.pred_label is None:
            print(f"Predicted label for restaurant {restaurant.name} is None.")
            continue

        y_true.append(restaurant.gold_label)
        y_pred.append(restaurant.pred_label)

    # For testing if the labels are assigned correctly
    # corpus.print_labels()

    # Set up evaluator
    evaluator = Evaluator(corpus)
    
    if len(set(y_true)) == 1 or len(set(y_pred)) == 1:
        print("Insufficient variation in the data to calculate correlation.")
    else:
        correlation = evaluator.evaluate_correlation()
        print(f"Spearman's Rank Correlation Coefficient: {correlation:.2f}")

    f1_score = evaluator.evaluate_f1_score()
    print(f"Macro Average F1 Score: {f1_score:.2f}")
