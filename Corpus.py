# class that represents the whole corpus of restaurants
from Restaurant import Restaurant
import string


class Corpus:
    def __init__(self, filepath: str):
        # this is for storing the list of restaurants
        self.instances = Corpus.read_file(filepath)

        # initialize feature dictionaries
        self.name_tokens = dict()
        self.food_types = dict()
        self.locations = dict()
        self.menu_tokens = dict()

        # extract features
        self.build_dictionaries()
        self.extract_features()

    @staticmethod
    def read_file(filepath: str) -> list:
        """
        Read a file containing one restaurant instance per line
        :param filepath: path to the data file
        :return: List of Restaurant instances
        """
        out = []
        with open(filepath) as f:
            for line in f:
                line = line.strip().split("\t")
                # append a new instance of Restaurant to the list
                out.append(Restaurant(line))
        return out

    def pred_from_file(self, filepath: str):
        """
        Read predicted labels for the corpus from file
        :param filepath: path to the file containing one predicted label per line
        """
        with open(filepath) as f:
            labels = [Restaurant.encode_label(line.strip()) for line in f]
        if len(labels) != len(self.instances):
            print("Number of predicted labels doesn't match the number of instances in the corpus. No labels were set")
            return
        for inst, pred in zip(self.instances, labels):
            inst.set_predicted_label(pred)

    @staticmethod
    def tokenize(text: str) -> list:
        """
        Split a string into tokens. Make it case-insensitive and able to handle punctuation.
        :param text: string to be tokenized
        :return: list of strings where each element is a token
        """
        text = text.lower()  # make it case-insensitive
        text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
        return text.split()

    def build_dictionaries(self):
        """
        Create dictionaries that map each unique category, location, menu item, and restaurant name token to a unique integer.
        """
        self.food_types = {category: i for i, category in enumerate(set([restaurant.category for restaurant in self.instances]))}
        self.locations = {location: i for i, location in enumerate(set([restaurant.location for restaurant in self.instances]))}
        self.menu_tokens = {item: i for i, item in enumerate(set([item for restaurant in self.instances for item in restaurant.menu]))}
        self.name_tokens = {token: i for i, token in enumerate(set([token for restaurant in self.instances for token in self.tokenize(restaurant.name)]))}

    def extract_features(self):
        """
        Store resulting features as an instance variable.
        Each restaurant will be represented as a dictionary where the keys are the feature names (location, food type, menu items, restaurant name)
        and the values are dictionaries where the keys are the indices of non-zero elements and the values are the non-zero values.
        """
        for restaurant in self.instances:
            features = {}
            # One-hot encoding for location
            features['location'] = {self.locations[restaurant.location]: 1}
            # One-hot encoding for food type
            features['food_type'] = {self.food_types[restaurant.category]: 1}
            # Bag of words for menu items
            features['menu'] = {self.menu_tokens[item]: 1 for item in restaurant.menu if item in self.menu_tokens}
            # Bag of words for restaurant name
            name_tokens = self.tokenize(restaurant.name)
            features['name'] = {self.name_tokens[token]: 1 for token in name_tokens if token in self.name_tokens}
            restaurant.features = features

    def get_dense_features(self, instance: Restaurant) -> list:
        """
        Converts sparse feature representation of a single Restaurant instance
        to decoded dense representation
        :param instance: Restaurant instance
        :return: Dense feature representation as a list
        """
        enc_name = instance.features['name']
        enc_food_type = instance.features['food_type']
        enc_location = instance.features['location']
        enc_menu = instance.features['menu']

        def _decode(feat_dict: dict, reference: dict):
            out = [0 for _ in range(len(reference))]
            for idx in feat_dict:
                out[idx] = feat_dict[idx]
            return out

        dec_name = _decode(enc_name, self.name_tokens)
        dec_food_type = _decode(enc_food_type, self.food_types)
        dec_location = _decode(enc_location, self.locations)
        dec_menu = _decode(enc_menu, self.menu_tokens)

        return dec_name + dec_food_type + dec_location + dec_menu


# for testing
if __name__ == "__main__":
    test = Corpus("data/menu_train.txt")
    test_inst = test.instances[10]
    decoded = test.get_dense_features(test_inst)
    print('done!')
