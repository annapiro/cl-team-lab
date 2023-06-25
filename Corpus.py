# class that represents the whole corpus of restaurants
from Restaurant import Restaurant
import string
import torch
import nltk
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import normalize

# Initialize BERT model and tokenizer
bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Use NLTK library for advanced tokenization and text processing
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

class Corpus:
    def __init__(self, train_data: list, test_data: list = None, exclude_feats: list = None):
        # list of instances that will be used for training
        self.train_data = train_data
        # list of instances that will be used for testing
        # can be added later via set_test_data
        self.test_data = test_data if test_data else None

        # default features that will be extracted in the corpus
        self.toggle_feats = {'name': 1, 'type': 1, 'loc': 1, 'menu': 1}
        if exclude_feats:
            for toggle in exclude_feats:
                self.toggle_feats[toggle] = 0

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

        # find maximum values in the training data to use in normalization
        self.max_menu_count = max({x for rest in self.train_data for x in rest.features['menu'].values()}) \
            if self.toggle_feats['menu'] else 1
        self.max_name_count = max({x for rest in self.train_data for x in rest.features['name'].values()}) \
            if self.toggle_feats['name'] else 1

        # store the length of the feature vector of each instance in the corpus
        self.num_feats = len(self.menu_tokens) + len(self.food_types) + len(self.locations) + len(self.name_tokens)

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
        25/06: Marina added NLTK improvements.
        Split a string into tokens. Make it case-insensitive and able to handle punctuation.
        Improved tokenizer with lemmatization and stop word removal.
        :param text: string to be tokenized
        :return: list of strings where each element is a token
        """
        text = text.lower()  # make it case-insensitive
        text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
        tokens = nltk.word_tokenize(text)
        return [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]

    def build_dictionaries(self):
        """
        Create dictionaries that map each unique category, location, menu item, and restaurant name token to a unique integer.
        Dictionaries are built only based on the features in the training data
        """
        if self.toggle_feats['type']:
            self.food_types = {category: i for i, category in enumerate(set([restaurant.category for restaurant in self.train_data]))}
        if self.toggle_feats['loc']:
            self.locations = {location: i for i, location in enumerate(set([restaurant.location for restaurant in self.train_data]))}
        if self.toggle_feats['menu']:
            self.menu_tokens = {token: i for i, token in enumerate(set([token for restaurant in self.train_data for item in restaurant.menu for token in self.tokenize(item)]))}
        if self.toggle_feats['name']:
            self.name_tokens = {token: i for i, token in enumerate(set([token for restaurant in self.train_data for token in self.tokenize(restaurant.name)]))}

    def generate_bert_embeddings(self, tokens):
        """
        25/06: Marina
        Generates BERT embeddings for a given list of tokens.
        """
        input_ids = bert_tokenizer.encode(tokens, add_special_tokens=True)  # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
        input_ids = input_ids[:512]  # Truncate to the maximum length BERT can handle

        with torch.no_grad():
            last_hidden_states = bert_model(torch.tensor([input_ids]))  # Models outputs are now tuples

        # Get embeddings of [CLS] token
        sentence_embedding = last_hidden_states[0][:, 0, :].numpy()
        return sentence_embedding.tolist()[0]  # Convert the numpy array to a Python list

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
            # create empty dictionary if location is OOV
            features['location'] = {self.locations[restaurant.location]: 1} \
                if self.toggle_feats['loc'] and restaurant.location in self.locations \
                else {}
            # One-hot encoding for food type
            # create empty dictionary if restaurant category (food type) is OOV
            features['food_type'] = {self.food_types[restaurant.category]: 1} \
                if self.toggle_feats['type'] and restaurant.category in self.food_types \
                else {}
            """
            # Bag of words for menu items, now counting each item
            if self.toggle_feats['menu']:
                features['menu'] = {}
                for item in restaurant.menu:
                    for token in self.tokenize(item):
                        if token in self.menu_tokens:
                            if self.menu_tokens[token] in features['menu']:
                                features['menu'][self.menu_tokens[token]] += 1
                            else:
                                features['menu'][self.menu_tokens[token]] = 1
            else:
                features['menu'] = {}
            """
            """
            # Bag of words for restaurant name, now counting each token
            if self.toggle_feats['name']:
                features['name'] = {}
                for token in self.tokenize(restaurant.name):
                    if token in self.name_tokens:
                        if self.name_tokens[token] in features['name']:
                            features['name'][self.name_tokens[token]] += 1
                        else:
                            features['name'][self.name_tokens[token]] = 1
            else:
                features['name'] = {}
            """
            """ 
            # Bag of words for menu items, now tokenizing each item
            features['menu'] = {self.menu_tokens[token]: 1 for item in restaurant.menu for token in self.tokenize(item) if token in self.menu_tokens} \
                if self.toggle_feats['menu'] \
                else {}
            # Bag of words for restaurant name
            if self.toggle_feats['name']:
                name_tokens = self.tokenize(restaurant.name)
                features['name'] = {self.name_tokens[token]: 1 for token in name_tokens if token in self.name_tokens}
            else:
                features['name'] = {}
            """
            # BERT embeddings for restaurant name
            if self.toggle_feats['name']:
                name_tokens = self.tokenize(restaurant.name)
                features['name'] = self.generate_bert_embeddings(name_tokens)
            # BERT embeddings for menu items
            if self.toggle_feats['menu']:
                menu_tokens = []
                for item in restaurant.menu:
                    menu_tokens.extend(self.tokenize(item))
                features['menu'] = self.generate_bert_embeddings(menu_tokens)
            restaurant.features = features


    # def get_dense_features(self, instance: Restaurant, normalize=False) -> list:
        """
        Converts sparse feature representation of a single Restaurant instance
        to decoded dense representation
        :param normalize: Choose whether the bag of words counts should be normalized (doesn't apply to binary BOW)
        :param instance: Restaurant instance
        :return: Dense feature representation as a list
        """
        """
        enc_name = instance.features['name'] if self.toggle_feats['name'] else {}
        enc_food_type = instance.features['food_type'] if self.toggle_feats['type'] else {}
        enc_location = instance.features['location'] if self.toggle_feats['loc'] else {}
        enc_menu = instance.features['menu'] if self.toggle_feats['menu'] else {}

        def _decode(feat_dict: dict, reference: dict):
            out = [0 for _ in range(len(reference))]
            for idx in feat_dict:
                out[idx] = feat_dict[idx]
            return out

        dec_name = _decode(enc_name, self.name_tokens)
        dec_food_type = _decode(enc_food_type, self.food_types)
        dec_location = _decode(enc_location, self.locations)
        dec_menu = _decode(enc_menu, self.menu_tokens)

        if normalize:
            dec_name = [round(x/self.max_name_count, 5) for x in dec_name]
            dec_menu = [round(x/self.max_menu_count, 5) for x in dec_menu]

        return dec_food_type + dec_location + dec_menu + dec_name
"""

    def get_dense_features(self, instance: Restaurant, normalize_embeddings=False) -> list:
        """
        25/06: Marina
        Converts sparse feature representation of a single Restaurant instance
        to decoded dense representation
        :param normalize_embeddings: Choose whether the BERT embeddings should be normalized (L2 normalization)
        :param instance: Restaurant instance
        :return: Dense feature representation as a list
        """
        dec_name = instance.features['name'] if self.toggle_feats['name'] else [0]*768
        dec_food_type = instance.features['food_type'] if self.toggle_feats['type'] else [0]*len(self.food_types)
        dec_location = instance.features['location'] if self.toggle_feats['loc'] else [0]*len(self.locations)
        dec_menu = instance.features['menu'] if self.toggle_feats['menu'] else [0]*768

        if normalize_embeddings:
            # Normalize BERT embeddings for 'name' and 'menu'
            dec_name = normalize([dec_name])[0].tolist()
            dec_menu = normalize([dec_menu])[0].tolist()

        return dec_food_type + dec_location + dec_menu + dec_name

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
