# class that represents the whole corpus of restaurants
from Restaurant import Restaurant


class Corpus:
    def __init__(self, filepath: str):
        # this is for storing the list of restaurants
        self.instances = Corpus.read_file(filepath)

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


# for testing
if __name__ == "__main__":
    test = Corpus("data/menu_train.txt")
    print(test.instances)
