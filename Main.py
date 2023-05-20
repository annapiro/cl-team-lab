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
    corpus = Corpus("data/menu_train.txt")

    # Extract features
    # corpus.extract_features()

    # Build the dictionaries
    # corpus.build_dictionaries()

    # Get the number of features
    num_features = len(corpus.menu_tokens) + len(corpus.food_types) + len(corpus.locations) + len(corpus.name_tokens)

    # Initialize perceptron for each class
    perceptrons = [Perceptron(num_features, i) for i in range(1, 5)]  # Assuming 4 price categories

    # Train perceptrons
    for epoch in tqdm(range(EPOCHS)):
        # Shuffle instances
        random.shuffle(corpus.instances)
        # Train each perceptron
        for perceptron in tqdm(perceptrons):
            for restaurant in corpus.instances:
                # Get the combined feature vector from all feature dictionaries
                # combined_features = list(itertools.chain(*restaurant.features.values()))
                # perceptron.update(combined_features, restaurant.gold_label)

                # Get dense features of the restaurant
                dense_features = corpus.get_dense_features(restaurant)
                perceptron.update(dense_features, restaurant.gold_label)
                
    # Evaluate perceptrons
    y_true = []
    y_pred = []
    for restaurant in corpus.instances:
        combined_features = list(itertools.chain(*restaurant.features.values()))
        # Get predictions from all perceptrons
        predictions = [perceptron.predict(combined_features) for perceptron in perceptrons]
        # Get the predicted class (1-indexed)
        predicted_class = predictions.index(max(predictions)) + 1
        y_true.append(restaurant.gold_label)
        y_pred.append(predicted_class)
        
    # Set up evaluator
    evaluator = Evaluator(corpus)
    f1_score = evaluator.evaluate_f1_score()
    correlation = evaluator.evaluate_correlation()

    print(f"Macro Average F1 Score: {f1_score:.2f}")
    print(f"Spearman's Rank Correlation Coefficient: {correlation:.2f}")
