# class that represents the whole corpus of restaurants
from Restaurant import Restaurant
import string


class Corpus:
    def __init__(self, train_data: list, test_data: list = None):
        # list of instances that will be used for training
        self.train_data = train_data
        # list of instances that will be used for testing
        # can be added later via set_test_data
        self.test_data = test_data if test_data else None

        # initialize feature dictionaries
        self.name_tokens = dict()
        self.food_types = dict()
        self.locations = dict()
        self.menu_tokens = dict()

        # extract features
        self.build_dictionaries()
        self.extract_features(self.train_data)
        if self.test_data:
            self.extract_features(self.test_data)

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
                instance = Restaurant(line)
                if instance.gold_label is None:
                    print(f"Warning: Invalid gold label for restaurant {instance.name}. Skipping this instance.")
                    continue  # Skip this instance
                out.append(instance)
        return out

    def set_test_data(self, test_data: list):
        """
        Set test instances for the corpus and extract their features
        :param test_data: List of test instances
        """
        self.test_data = test_data
        self.extract_features(self.test_data)

    def pred_from_file(self, filepath: str):
        """
        Read predicted labels for the corpus from file
        :param filepath: path to the file containing one predicted label per line
        """
        with open(filepath) as f:
            labels = [Restaurant.encode_label(line.strip()) for line in f]
        if len(labels) != len(self.train_data):
            print("Number of predicted labels doesn't match the number of instances in the corpus. No labels were set")
            return
        for inst, pred in zip(self.train_data, labels):
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
        Dictionaries are built only based on the features in the training data
        """
        self.food_types = {category: i for i, category in enumerate(set([restaurant.category for restaurant in self.train_data]))}
        self.locations = {location: i for i, location in enumerate(set([restaurant.location for restaurant in self.train_data]))}
        self.menu_tokens = {token: i for i, token in enumerate(set([token for restaurant in self.train_data for item in restaurant.menu for token in self.tokenize(item)]))}
        self.name_tokens = {token: i for i, token in enumerate(set([token for restaurant in self.train_data for token in self.tokenize(restaurant.name)]))}

    def extract_features(self, instances: list):
        """
        Store resulting features as an instance variable.
        Each restaurant will be represented as a dictionary where the keys are the feature names (location, food type, menu items, restaurant name)
        and the values are dictionaries where the keys are the indices of non-zero elements and the values are the non-zero values.
        :param instances: Features will be extracted for each instance on the list
        and stored in the corresponding object
        """
        for restaurant in instances:
            features = {}
            # One-hot encoding for location
            features['location'] = {self.locations[restaurant.location]: 1} \
                if restaurant.location in self.locations \
                else {}
            # One-hot encoding for food type
            features['food_type'] = {self.food_types[restaurant.category]: 1} \
                if restaurant.category in self.food_types \
                else {}
            # Bag of words for menu items, now tokenizing each item
            features['menu'] = {self.menu_tokens[token]: 1 for item in restaurant.menu for token in self.tokenize(item) if token in self.menu_tokens}
            # Bag of words for restaurant name
            # name_tokens = self.tokenize(restaurant.name)
            # features['name'] = {self.name_tokens[token]: 1 for token in name_tokens if token in self.name_tokens}
            restaurant.features = features

    def get_dense_features(self, instance: Restaurant) -> list:
        """
        Converts sparse feature representation of a single Restaurant instance
        to decoded dense representation
        :param instance: Restaurant instance
        :return: Dense feature representation as a list
        """
        # enc_name = instance.features['name']
        enc_food_type = instance.features['food_type']
        enc_location = instance.features['location']
        enc_menu = instance.features['menu']

        def _decode(feat_dict: dict, reference: dict):
            out = [0 for _ in range(len(reference))]
            for idx in feat_dict:
                out[idx] = feat_dict[idx]
            return out

        # dec_name = _decode(enc_name, self.name_tokens)
        dec_food_type = _decode(enc_food_type, self.food_types)
        dec_location = _decode(enc_location, self.locations)
        dec_menu = _decode(enc_menu, self.menu_tokens)

        return dec_food_type + dec_location + dec_menu#  + dec_name

    def print_labels(self):
        """
        Prints the gold (true) label and predicted label for each instance in the corpus.
        """
        for instance in self.train_data:
            print(f"Restaurant: {instance.name}")
            print(f"Gold Label: {instance.gold_label}")
            print(f"Predicted Label: {instance.pred_label}")
            print("-"*30)  # prints a divider for clarity


# for testing
if __name__ == "__main__":
    data = Corpus.read_file("data/menu_train.txt")
    test = Corpus(data)
    test_inst = test.train_data[10]
    decoded = test.get_dense_features(test_inst)
    print('done!')
