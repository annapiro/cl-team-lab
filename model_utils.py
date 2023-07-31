"""
model_utils.py

Utility functions to save and load trained models
"""
import os
import pickle
from Corpus import Corpus


def save_model(corpus: Corpus, perceptrons: list, model: str):
    """
    Save perceptrons and the feature mappings for the corpus
    :param corpus: Corpus that the perceptrons were trained on
    :param perceptrons: List of perceptrons
    :param model: Name for the model and the folder where it will be stored
    """
    folder_path = f"models/{model}"

    if os.path.exists(folder_path):
        raise FileExistsError(f"The model '{model}' already exists.")

    os.makedirs(folder_path)

    for perceptron in perceptrons:
        with open(f"{folder_path}/perc{perceptron.tar_label}", "wb") as f:
            pickle.dump(perceptron, f)

    corpus.save_feature_mapping(f"{folder_path}/map.json")
    print(f"Model saved as {folder_path}")


def load_model(model: str) -> (Corpus, list):
    """
    Load perceptrons and feature mappings from a folder
    :param model: Name of the model and the folder where it is stored
    :return: Corpus object and a list of perceptrons
    """
    folder_path = f"models/{model}"
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The model '{model}' does not exist.")

    corpus = Corpus(load_mapping=f"{folder_path}/map.json")

    file_list = os.listdir(folder_path)
    if not file_list:
        raise FileNotFoundError(f"The folder '{model}' is empty.")

    perceptrons = []
    for filename in file_list:
        if filename.startswith("perc"):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, "rb") as f:
                perc = pickle.load(f)
            perceptrons.append(perc)

    # sort the perceptron list ascending by target label
    perceptrons.sort(key=lambda perc: perc.tar_label)

    print(f"Model loaded from {folder_path}")
    return corpus, perceptrons
