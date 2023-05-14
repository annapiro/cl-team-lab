# class that implements various evaluation methods for a corpus
from Corpus import Corpus


class Evaluator:
    # The constructor initializes the Evaluator with a Corpus object.
    def __init__(self, corpus):
        self.corpus = corpus

    def create_confusion_matrix(self, y_true, y_pred):
        # Get the number of unique labels in the true labels
        n_labels = len(set(y_true))
        # Initialize an n_labels x n_labels matrix with all elements set to 0
        matrix = [[0] * n_labels for _ in range(n_labels)]

        # Iterate through the true and predicted labels
        for true, pred in zip(y_true, y_pred):
            # Increment the corresponding cell in the confusion matrix
            matrix[true - 1][pred - 1] += 1

        return matrix

    def precision_recall(self, confusion_matrix):
        # Get the number of unique labels in the confusion matrix
        n_labels = len(confusion_matrix)
        # Initialize precision and recall lists with n_labels elements set to 0
        precision = [0] * n_labels
        recall = [0] * n_labels

        # Iterate through the labels
        for i in range(n_labels):
            # Calculate true positives, false positives, and false negatives
            true_positives = confusion_matrix[i][i]
            false_positives = sum(row[i] for row in confusion_matrix) - true_positives
            false_negatives = sum(confusion_matrix[i]) - true_positives

            # Calculate precision and recall for the current label
            if true_positives + false_positives > 0:
                precision[i] = true_positives / (true_positives + false_positives)
            else:
                precision[i] = 0

            if true_positives + false_negatives > 0:
                recall[i] = true_positives / (true_positives + false_negatives)
            else:
                recall[i] = 0

        return precision, recall

    def f1_score(self, precision, recall):
        # Get the number of labels from the precision list
        n_labels = len(precision)
        # Initialize the F1-score list with n_labels elements set to 0
        f1 = [0] * n_labels

        # Iterate through the labels
        for i in range(n_labels):
            # Calculate F1-score for the current label
            if precision[i] + recall[i] > 0:
                f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
            else:
                f1[i] = 0

        # Calculate the macro-averaged F1-score
        macro_avg_f1 = sum(f1) / n_labels

        return f1, macro_avg_f1

    def rank_data(self, data):
        # Sort the data in ascending order
        sorted_data = sorted(data)
        # Initialize a dictionary to store the ranks
        ranks = {}
        # Iterate through the sorted data and assign ranks
        for i, val in enumerate(sorted_data):
            if val not in ranks:
                ranks[val] = i + 1
        # Return the ranks for the original data
        return [ranks[val] for val in data]

    def squared_rank_differences(self, y_true_ranked, y_pred_ranked):
        # Calculate the squared differences between the true and predicted ranks
        return [(y_true_ranked[i] - y_pred_ranked[i]) ** 2 for i in range(len(y_true_ranked))]


    def spearman_correlation(self, y_true_ranked, y_pred_ranked, squared_differences):
        # Calculate the number of elements in the ranked data
        n = len(y_true_ranked)
        # Calculate the numerator of the Spearman correlation formula
        numerator = 6 * sum(squared_differences)
        # Calculate the denominator of the Spearman correlation formula
        denominator = n * (n ** 2 - 1)
        # Calculate the Spearman correlation
        return 1 - (numerator / denominator)

    # This method calculates the macro-average F1 score for the model.
    def evaluate_f1_score(self):
        # Extract the gold (true) labels and predicted labels from the Corpus instances.
        y_true = [instance.gold_label for instance in self.corpus.instances]
        y_pred = [instance.pred_label for instance in self.corpus.instances]

        # Compute the confusion matrix, precision, recall, and F1 scores.
        confusion_matrix = self.create_confusion_matrix(y_true, y_pred)
        precision, recall = self.precision_recall(confusion_matrix)
        f1, macro_avg_f1 = f1_score(precision, recall)

        # Return the macro-average F1 score.
        return macro_avg_f1

    # This method calculates Spearman's rank correlation coefficient for the model.
    def evaluate_correlation(self):
        # Extract the gold (true) labels and predicted labels from the Corpus instances.
        y_true = [instance.gold_label for instance in self.corpus.instances]
        y_pred = [instance.pred_label for instance in self.corpus.instances]

        # Compute the ranked gold (true) labels and predicted labels.
        y_true_ranked = self.rank_data(y_true)
        y_pred_ranked = self.rank_data(y_pred)

        # Calculate the squared rank differences between the ranked gold (true) labels and predicted labels.
        squared_differences = self.squared_rank_differences(y_true_ranked, y_pred_ranked)

        # Compute Spearman's rank correlation coefficient using the ranked gold (true) labels, predicted labels, and squared rank differences.
        correlation = self.spearman_correlation(y_true_ranked, y_pred_ranked, squared_differences)

        # Return Spearman's rank correlation coefficient.
        return correlation


# For testing
if __name__ == "__main__":
    test_corpus = Corpus("data/menu_train.txt")

    # Set predicted labels for the instances in the corpus
    # TODO: This should be replaced  with the  actual prediction code
    for instance in test_corpus.instances:
        instance.set_predicted_label(instance.gold_label)  # For now, set the predicted label to the gold label

    evaluator = Evaluator(test_corpus)

    f1_score = evaluator.evaluate_f1_score()
    correlation = evaluator.evaluate_correlation()

    print(f"Macro Average F1 Score: {f1_score:.2f}")
    print(f"Spearman's Rank Correlation Coefficient: {correlation:.2f}")
