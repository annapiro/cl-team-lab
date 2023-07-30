import argparse
import pickle
from Perceptron import Perceptron
from Corpus import Corpus
from tqdm import tqdm
from typing import Any


def save_model(model: Any, filepath: str):
    """
    Wrapper to save the model in a pickle
    :param model: Model as an object
    :param filepath: Path to the file it should be saved in
    """
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)


def main(model: str, file: str, method: str, epochs: int, exclude_feats: list):
    # Load corpus
    data = Corpus.read_file(f'data/{file}')  # comment out this line if loading from json

    # create corpus from training data
    # choose which features to exclude: ['name', 'type', 'loc', 'menu'] TODO move to docs
    # chose feature extraction method: 'bow' for bag of words or 'emb' for embeddings
    corpus = Corpus(train_data=data, exclude_feats=exclude_feats, method=method)

    # initialize a perceptron for each price category
    perceptrons = [Perceptron(corpus.num_feats, i) for i in range(1, 5)]  # Assuming 4 price categories

    # collect training data as a list of tuples (feature_vector, gold_label)
    train_data = [(corpus.get_dense_features(restaurant), restaurant.gold_label) for restaurant in corpus.train_data]

    print("\nTraining the model...")
    for perceptron in tqdm(perceptrons):
        perceptron.train(train_data, epochs)

    # TODO rewrite for new file structure
    for p in perceptrons:
        save_model(p, "out/emb_allfeats_3ep_perc" + str(p.tar_label))
        print(f"Perceptron {p.tar_label} saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='Model name')
    parser.add_argument('file', help='Training file')
    parser.add_argument('method', help='bow for bag of words, emb for embeddings')
    parser.add_argument('epochs', type=int, help='Number of epochs')
    parser.add_argument('--exclude', nargs='*', help='Features to exclude from training (optional). '
                                                     'Possible values: name, type, loc, menu')
    args = parser.parse_args()

    main(args.model, args.file, args.method, args.epochs, args.exclude)
