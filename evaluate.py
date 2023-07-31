import argparse
from Corpus import Corpus
from Evaluator import Evaluator
from model_utils import load_model


def main(model: str, file: str):
    dev = Corpus.read_file(f'data/{file}')

    corpus, perceptrons = load_model(model)
    corpus.set_test_data(dev)

    # Make predictions
    for restaurant in corpus.test_data:
        dense_features = corpus.get_dense_features(restaurant)
        predictions = [perceptron.predict(dense_features) for perceptron in perceptrons]
        # Get the predicted class (1-indexed)
        predicted_class = predictions.index(max(predictions)) + 1
        # Set the predicted label
        restaurant.set_predicted_label(predicted_class)

    f1_scores = []
    correlations = []

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

    print()
    print(f"F1 Score: {average_f1_score:.2f}")
    print(f"Correlation: {average_correlation:.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='Model name')
    parser.add_argument('file', help='Test file')
    args = parser.parse_args()

    main(args.model, args.file)
