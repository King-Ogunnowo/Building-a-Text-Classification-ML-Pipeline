"""
Function to test trained model

Author: Oluwaseyi E. Ogunnowo
Date: July 8th, 2022
"""

import argparse
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import constants

parser = argparse.ArgumentParser(description = "Script to visualize data properties.\
NB: data passed must be saved in .csv format")

parser.add_argument("--test_set",
                    type = str,
                    help = 'path to training sample')

parser.add_argument("--independent_feature",
                    type = str,
                    help = 'column in dataframe containing texts')

parser.add_argument("--dependent_feature",
                    type = str,
                    help = 'column containing encoded target values,\
                    i.e. the values you want to predict')

args = parser.parse_args()

def load_pickle_obj(path: str):
    """
    Function to unpickle/ deserialize pickled object
    parameters
    ----------
    path (string): path to the pickled object.
    This function can unpickle models and dictionaries
    returns
    ----------
    unpicked object
    """
    with open(path, 'rb') as file:
        return pkl.load(file)
    print(f"object from {path} loaded successfully")

def map_id_to_interest(id_:int, dictionary:dict):
    """
    Function to convert predicted integers to target_values
    parameters
    ----------
    id_ (integer): predicted integer
    dictionary (dict): containing key, value pair of integer and corresponding string value
    returns
    ----------
    string, essentially a decoded integer
    """
    return dictionary[id_]


    
model = load_pickle_obj("model/model_pipeline.pkl")
id_to_value_dict = load_pickle_obj("dictionary/id_to_value.pkl")
test_set = pd.read_csv(args.test_set)
test_set['actual'] = test_set[args.dependent_feature]\
.apply(lambda x: map_id_to_interest(x, id_to_value_dict))

x_test = test_set[args.independent_feature]
y_test = test_set[args.dependent_feature]

constants.logger.info("validating model")
test_set['predicted'] = model.predict(x_test)
test_set['predicted'] = test_set['predicted']\
.apply(lambda x: map_id_to_interest(x, id_to_value_dict))

constants.logger.info("generating classification report and storing in results folder")
plt.figure(figsize=(6, 5))
plt.text(0.01, 1.05, str("Classification Report"),{'fontsize': 10}, fontproperties='monospace')
plt.text(0.01, 0.05, str(classification_report(test_set['actual'], test_set['predicted'])), {'fontsize': 10}, fontproperties='monospace')
plt.axis('off')
plt.savefig('results/classification_reports.png')

constants.logger.info("generating confusion matrix and storing in results folder")
cm = pd.DataFrame(confusion_matrix(test_set['actual'], test_set['predicted']), 
                  index = id_to_value_dict.values(), columns = id_to_value_dict.values())
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='g')
plt.title('Confusion Matrix')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
plt.savefig("results/confusion_matrix.png")




