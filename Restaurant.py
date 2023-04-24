# class that represents a single instance of a restaurant
# has attributes name, gold label, predicted label, category, location, and menu
# TODO implement the possibility to read/set predicted labels

class Restaurant:
    def __init__(self, features: list):
        self.name = features[1]
        self.gold_label = Restaurant.encode_label(features[0])
        self.pred_label = ''  # TODO
        self.category = features[2]
        self.location = features[3]
        self.menu = features[4].split(";")

    def __repr__(self):
        """
        Defines string representation of the object, so that a print() method can be called on it
        Note: to save screen space, the number of menu items is listed instead of the items themselves
        :return: string representation of the Restaurant instance
        """
        return f'[{self.gold_label};{self.name};{self.category};{self.location};{len(self.menu)} menu items]'

    @staticmethod
    def encode_label(label: str) -> int:
        """
        Simple method to map string-based labels to number-based ones
        TODO not sure if this encoding is good, maybe revisit it
        :param label: label in the form of a string $, $$, $$$, or $$$$
        :return: integer representing the label
        """
        return len(label)
