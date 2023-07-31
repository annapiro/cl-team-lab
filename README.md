# Introduction

This classifier was developed to explore the role of different features as predictors of the price level of a restaurant. The project is built upon the dataset provided by Jurafsky, Chahuneau, Routledge, and Smith in their 2016 paper "Linguistic Markers of Status in Food Culture: Bourdieu's Distinction in a Menu Corpus".

# Prerequisites

The required modules are listed in `requirements.txt`

The data should be saved in the folder root/data in the txt format and follow the same structure as the dataset by Jurafsky et al.

Two pre-trained example models are provided in this repository as a starting point.

Note: The example models were not trained on the full dataset, so their performance might differ from the results outlined in the report.

# Training

Train the model by running the following command:

```
py train.py model_name training_file method epochs
```

- `model_name`: Choose the name your model will be saved under
- `training_file`: Name of the training file in the `data` folder
- `method`: Feature extraction method, choose either `bow` for bag of words or `emb` for BERT embeddings
- `epochs`: Number of epochs
- `--exclude`: Optional argument that lets you exclude certain features from the training process. Possible values:
  - `name` - restaurant name
  - `type` - food type (cuisine)
  - `loc` - the restaurant's location
  - `menu` - the menu

Example usage:

```
py train.py my_model menu_train.txt bow 5 --exclude loc name 
```

# Evaluation
## F1-score and correlation
After the model is trained in the previous step and saved in the `models` folder, run the evaluation script:

```
py evaluate.py model_name test_file
```

- `model_name`: Name of your model that it is saved under in `models`
- `test_file`: Name of the test file, located in the `data` folder

Example usage:

```
py evaluate.py my_model menu_test.txt
```

The evaluation results will be displayed on the command line.

## Identify top predictors
Another way to evaluate your model is to find out which individual tokens it identified as top positive and negative predictors for each price category. This can be done by running the following script:

```
py get_top_predictors.py model_name
```

The script will create a text file for each feature (name, food type, location, menu) of each perceptron (one per each price category) with tokens sorted in descending order by their respective weights. The files are saved in the model's directory under `weights`. 
