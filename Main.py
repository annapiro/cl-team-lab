import itertools
import random
from Perceptron import Perceptron
from Corpus import Corpus
from Restaurant import Restaurant
from Evaluator import Evaluator
from tqdm import tqdm

if __name__ == "__main__":
    EPOCHS = 10  # Define the number of training iterations

    # Load corpus
    data = Corpus.read_file("data/menu_train.txt")
    split = int(len(data) / 5)
    dev, train = data[:split], data[split:]
    corpus = Corpus(train)
    corpus.set_test_data(dev)

    # Extract features
    # corpus.extract_features()

    # Build the dictionaries
    # corpus.build_dictionaries()

    # Get the number of features
    num_features = len(corpus.menu_tokens) + len(corpus.food_types) + len(corpus.locations) + len(corpus.name_tokens)

    # Initialize perceptron for each class
    perceptrons = [Perceptron(num_features, i) for i in range(1, 5)]  # Assuming 4 price categories

    # Train perceptrons and make predictions
    for epoch in tqdm(range(EPOCHS)):
        # Shuffle instances
        random.shuffle(corpus.train_data)
        # Train each perceptron
        for perceptron in tqdm(perceptrons):
            for restaurant in corpus.train_data:
                # Get dense features of the restaurant
                dense_features = corpus.get_dense_features(restaurant)
                perceptron.update(dense_features, restaurant.gold_label)

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
