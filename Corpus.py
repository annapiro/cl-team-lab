"""
Corpus

Represents the whole corpus of restaurants.

Date: 09.07.2023

Provides functionality to store training and test sets, read datasets from file,
extract features from the dataset, and save/load the resulting feature mappings

Available methods of feature extraction:
- bag of words (method='bow')
- sentence embeddings (method='emb')
"""

from Restaurant import Restaurant
import string
import json
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class Corpus:
    def __init__(self,
                 train_data: list = None,
                 test_data: list = None,
                 exclude_feats: list = None,
                 load_mapping: str = None,
                 method: str = 'bow'):
        # current version of Corpus for tracking compatibility with external feature mappings
        self.version = 'v2.2'

        if method != 'bow' and method != 'emb':
            raise ValueError("Invalid method chosen. "
                             "Please set either 'bow' for bag of words or 'emb' for embeddings")

        # either method of populating the feature dictionaries should be provided
        if not train_data and not load_mapping:
            raise ValueError("Please provide either training data as tsv file or feature mapping as json. "
                             "No corpus was created")
        if train_data and load_mapping:
            print("New training data was provided. Old feature mapping will be ignored")

        # list of instances that will be used for training
        self.train_data = train_data
        # list of instances that will be used for testing
        # can be added later via set_test_data
        self.test_data = test_data

        # set flag whether bag of words or embeddings will be used
        self.method = method
        # set the model for extracting embeddings and its fixed output length
        self.emb_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.emb_len = 384

        # default features that will be extracted in the corpus
        self.toggle_feats = {'name': 1, 'type': 1, 'loc': 1, 'menu': 1}
        if exclude_feats:
            for toggle in exclude_feats:
                if toggle not in self.toggle_feats:
                    print(f"Warning: '{toggle}' feature does not exist, ignoring the command")
                    continue
                self.toggle_feats[toggle] = 0

        # initialize feature dictionaries
        self.map_names = dict()
        self.map_types = dict()
        self.map_locs = dict()
        self.map_menu = dict()

        # initialize maximum values in the training data (used in normalization)
        self.max_menu_count = 1
        self.max_name_count = 1

        if train_data:
            self._init_from_data()
        elif load_mapping:
            self.load_feature_mapping(load_mapping)

        if self.test_data:
            self.extract_features(self.test_data)

        # store the length of the feature vector of each instance in the corpus
        if self.method == 'bow':
            self.num_feats = len(self.map_menu) + len(self.map_types) + len(self.map_locs) + len(self.map_names)
        else:
            self.num_feats = self.emb_len * sum(self.toggle_feats.values())

    def _init_from_data(self):
        """
        Creates feature dictionaries from scratch in cases where training data is provided
        """
        # extract features
        if self.method == 'bow':
            self.build_dictionaries()
        self.extract_features(self.train_data)
        # update max values only if bag of words is used
        if self.method == 'bow':
            if self.toggle_feats['menu']:
                self.max_menu_count = max({x for rest in self.train_data for x in rest.features['menu'].values()})
            if self.toggle_feats['name']:
                self.max_name_count = max({x for rest in self.train_data for x in rest.features['name'].values()})

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
        based on pre-existing feature dictionaries
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
        text = text.lower().replace(";", " ")  # make it case-insensitive and remove separators
        text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
        return text.split()

    def build_dictionaries(self):
        """
        Create dictionaries that map each unique category, location, menu item, and restaurant name token to a unique integer.
        Dictionaries are built only based on the features in the training data
        """
        if self.toggle_feats['type']:
            self.map_types = {category: i for i, category in enumerate(
                set([restaurant.category for restaurant in self.train_data]))}
        if self.toggle_feats['loc']:
            self.map_locs = {location: i for i, location in enumerate(
                set([restaurant.location for restaurant in self.train_data]))}
        if self.toggle_feats['menu']:
            self.map_menu = {token: i for i, token in enumerate(
                set([token for restaurant in self.train_data for token in self.tokenize(restaurant.menu)]))}
        if self.toggle_feats['name']:
            self.map_names = {token: i for i, token in enumerate(
                set([token for restaurant in self.train_data for token in self.tokenize(restaurant.name)]))}

    def extract_features(self, instances: list):
        """
        Store resulting features as an instance variable.
        Each restaurant will be represented as a dictionary where the keys are the feature names (location, food type, menu items, restaurant name)
        and the values are vectorized features depending on the chosen method of feature extraction.
        :param instances: Features will be extracted for each instance on the list
        and stored in the corresponding object
        """
        def extract_emb() -> (list, list, list, list):
            """
            Generates fixed-length sentence-level embeddings for each feature
            :return: Embedding vectors for each feature in order: name, food type, location, menu
            """
            name = self.emb_model.encode(restaurant.name).tolist() \
                if self.toggle_feats['name'] else []
            food_type = self.emb_model.encode(restaurant.category).tolist() \
                if self.toggle_feats['type'] else []
            location = self.emb_model.encode(restaurant.location).tolist() \
                if self.toggle_feats['loc'] else []
            menu = self.emb_model.encode(restaurant.menu).tolist() \
                if self.toggle_feats['menu'] else []

            return name, food_type, location, menu

        def extract_bow(counted: bool = False) -> (dict, dict, dict, dict):
            """
            For each feature, creates a dictionary where the keys are the indices of non-zero elements
            and the values are the non-zero values.
            :param counted: set to True to use counted bag of words instead of binary
            :return: Sparse feature dictionaries in order: name, food type, location, menu
            """
            # One-hot encoding for location
            # create empty dictionary if location is OOV
            location = {self.map_locs[restaurant.location]: 1}\
                if self.toggle_feats['loc'] and restaurant.location in self.map_locs \
                else {}

            # One-hot encoding for food type
            # create empty dictionary if restaurant category (food type) is OOV
            food_type = {self.map_types[restaurant.category]: 1} \
                if self.toggle_feats['type'] and restaurant.category in self.map_types \
                else {}

            # bag of words for menu items
            menu = {}
            if self.toggle_feats['menu']:
                tokenized_menu = self.tokenize(restaurant.menu)
                # counted
                if counted:
                    for token in tokenized_menu:
                        if token in self.map_menu:
                            if self.map_menu[token] not in menu:
                                menu[self.map_menu[token]] = 0
                            menu[self.map_menu[token]] += 1
                # binary
                else:
                    menu = {self.map_menu[token]: 1 for token in tokenized_menu
                            if token in self.map_menu}

            # bag of words for restaurant names
            name = {}
            if self.toggle_feats['name']:
                tokenized_name = self.tokenize(restaurant.name)
                # counted
                if counted:
                    for token in tokenized_name:
                        if token in self.map_names:
                            if self.map_names[token] not in name:
                                name[self.map_names[token]] = 0
                            name[self.map_names[token]] += 1
                # binary
                else:
                    name = {self.map_names[token]: 1 for token in tokenized_name
                            if token in self.map_names}

            return name, food_type, location, menu

        print("\nExtracting features...")
        for restaurant in tqdm(instances):
            features = {}
            if self.method == 'bow':
                features['name'], features['food_type'], features['location'], features['menu'] = extract_bow()
            elif self.method == 'emb':
                features['name'], features['food_type'], features['location'], features['menu'] = extract_emb()
            restaurant.features = features

    def get_dense_features(self, instance: Restaurant, normalize=False) -> list:
        """
        Converts sparse feature representation of a single Restaurant instance
        to decoded dense representation
        :param normalize: Choose whether the bag of words counts should be normalized
        (doesn't apply to binary BOW or embeddings)
        :param instance: Restaurant instance
        :return: Dense feature representation as a list
        """
        # embeddings can be concatenated and returned as-is
        if self.method == 'emb':
            return instance.features['name'] + instance.features['food_type'] + \
                instance.features['location'] + instance.features['menu']

        # BOW requires decoding
        enc_name = instance.features['name'] if self.toggle_feats['name'] else {}
        enc_food_type = instance.features['food_type'] if self.toggle_feats['type'] else {}
        enc_location = instance.features['location'] if self.toggle_feats['loc'] else {}
        enc_menu = instance.features['menu'] if self.toggle_feats['menu'] else {}

        def _decode(feat_dict: dict, reference: dict):
            out = [0 for _ in range(len(reference))]
            for idx in feat_dict:
                out[idx] = feat_dict[idx]
            return out

        dec_name = _decode(enc_name, self.map_names)
        dec_food_type = _decode(enc_food_type, self.map_types)
        dec_location = _decode(enc_location, self.map_locs)
        dec_menu = _decode(enc_menu, self.map_menu)

        if normalize:
            dec_name = [round(x/self.max_name_count, 5) for x in dec_name]
            dec_menu = [round(x/self.max_menu_count, 5) for x in dec_menu]

        return dec_name + dec_food_type + dec_location + dec_menu

    def print_labels(self):
        """
        Prints the gold (true) label and predicted label for each instance in the corpus.
        """
        for instance in self.train_data:
            print(f"Restaurant: {instance.name}")
            print(f"Gold Label: {instance.gold_label}")
            print(f"Predicted Label: {instance.pred_label}")
            print("-"*30)  # prints a divider for clarity

    def save_feature_mapping(self, filepath: str):
        """
        Saves feature mappings and other settings to a json file
        The file is always called last_feature_mapping.json and saved in current directory
        """
        feature_mapping = {
            "version": self.version,
            "toggle_feats": self.toggle_feats,
            "method": self.method,
            "max_values": (self.max_name_count, self.max_menu_count),
            "map_locs": self.map_locs,
            "map_types": self.map_types,
            "map_names": self.map_names,
            "map_menu": self.map_menu
        }
        with open(filepath, "w") as f:
            json.dump(feature_mapping, f, indent=4)

    def load_feature_mapping(self, filepath: str):
        """
        Loads saved feature mappings from a json file
        :param filepath: path to the json file with the settings
        """
        try:
            with open(filepath) as f:
                feature_mapping = json.load(f)
            if feature_mapping["version"] != self.version:
                raise ValueError("JSON file is incompatible with the current version of Corpus")
            self.toggle_feats = feature_mapping["toggle_feats"]
            self.method = feature_mapping["method"]
            self.map_names = feature_mapping["map_names"]
            self.map_types = feature_mapping["map_types"]
            self.map_locs = feature_mapping["map_locs"]
            self.map_menu = feature_mapping["map_menu"]
            self.max_name_count = feature_mapping["max_values"][0]
            self.max_menu_count = feature_mapping["max_values"][1]
        except Exception as e:
            raise Exception("Something went wrong while loading the feature mapping. Likely the JSON file "
                            "was created with an older version of Corpus and is no longer compatible") from e
