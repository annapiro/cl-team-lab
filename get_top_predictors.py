"""
get_top_predictors.py

Date: 10.07.2023

Script that loads the perceptron models and corresponding feature mapping
and outputs a text file per each feature per perceptron
with predictors sorted by their weights, from most positive to most negative
"""
import argparse
import os
from Perceptron import Perceptron
from Corpus import Corpus
from model_utils import load_model


def split_weights(c: Corpus, p: Perceptron) -> (list, list, list, list):
    """
    TODO
    :param c:
    :param p:
    :return: feature weights in order: name, type, location, menu
    """
    # get the length of each feature in the corpus
    len_name = len(c.map_names)
    len_type = len(c.map_types)
    len_loc = len(c.map_locs)

    w = p.weights
    w_name = w[:len_name]
    w_type = w[len_name:len_name + len_type]
    w_loc = w[len_name + len_type:len_name + len_type + len_loc]
    w_menu = w[len_name + len_type + len_loc:]

    return w_name, w_type, w_loc, w_menu


def sort_by_weight(map: dict, weights: list) -> list:
    """
    Sorts list of tokens by weights in descending order
    :param map:
    :param weights:
    :return: TODO
    """
    # kw = {key:weight for key in map.keys() for weight in weights}
    kw = dict(zip(map.keys(), weights))
    return sorted(kw.items(), key=lambda item: item[1], reverse=True)


def save_to_file(folder_path: str, fname: str, obj: list):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    with open(f"{folder_path}/{fname}.txt", "w") as f:
        for item in obj:
            f.write(f"{item[0]} : {item[1]}\n")


def weights_to_file(c: Corpus, p: Perceptron, model: str):
    w_name, w_type, w_loc, w_menu = split_weights(c, p)
    p_loc = sort_by_weight(c.map_locs, w_loc)
    p_name = sort_by_weight(c.map_names, w_name)
    p_type = sort_by_weight(c.map_types, w_type)
    p_menu = sort_by_weight(c.map_menu, w_menu)

    price_cat = p.tar_label
    folder_path = f"models/{model}/weights"
    save_to_file(folder_path, f"perc{price_cat}_loc", p_loc)
    save_to_file(folder_path, f"perc{price_cat}_name", p_name)
    save_to_file(folder_path, f"perc{price_cat}_type", p_type)
    save_to_file(folder_path, f"perc{price_cat}_menu", p_menu)


def main(model: str):
    corpus, perceptrons = load_model(model)

    for p in perceptrons:
        weights_to_file(corpus, p, model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='Model name')
    args = parser.parse_args()

    main(args.model)
