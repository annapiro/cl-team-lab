import argparse
from Perceptron import Perceptron
from Corpus import Corpus
from tqdm import tqdm
from model_utils import save_model


def main(model: str, file: str, method: str, epochs: int, exclude_feats: list = None):
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

    save_model(corpus, perceptrons, model)


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
