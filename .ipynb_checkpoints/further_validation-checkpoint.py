import re
import argparse
import pickle as pkl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import constants
from wordcloud import WordCloud
from sklearn.metrics import classification_report, confusion_matrix

parser = argparse.ArgumentParser(description = "path to validation data")
parser.add_argument("path", type = str, help = "path to validation data, must be in csv format")
args = parser.parse_args()
val_set = pd.read_csv(args.path)
file_name = args.path.split('/')[2]

stop_words = constants.stop_words

def load_pickle_obj(path):
    with open(path, 'rb') as file:
        return pkl.load(file)
    
def map_id_to_interest(id_, dictionary):
    return dictionary[id_]

def clean_text(text):
    
    text = text.lower()
    text = re.sub("[0-9]", " ", text)
    text = re.sub("[^a-z]", " ", text)
    
    text = [i for i in text.split() if i not in stop_words and len(i) > 2]
    text = ' '.join(text)
    
    return text


constants.logger.info("loading model and other dependencies")
model = load_pickle_obj("model/model_pipeline.pkl")
id_to_interest_dict = load_pickle_obj("dictionary/id_to_interest.pkl")
interest_to_id_dict = load_pickle_obj("dictionary/interest_to_id.pkl")

constants.logger.info("encoding interests")
val_set['short_message_cleaned'] = val_set['short_message'].apply(lambda x: clean_text(x))
val_set['interest_id'] = val_set['interest'].apply(lambda x: map_id_to_interest(x, interest_to_id_dict))

constants.logger.info("generating predictions")
val_set['predicted_id'] = model.predict(val_set['short_message_cleaned'])
val_set['predicted_interest'] = val_set['predicted_id'].apply(lambda x: map_id_to_interest(x, id_to_interest_dict))

constants.logger.info("generating classification report and storing in results folder")
plt.figure(figsize=(6, 5))
plt.text(0.01, 1.05, str("Classification Report"),{'fontsize': 10}, fontproperties='monospace')
plt.text(0.01, 0.05, str(classification_report(val_set['interest'], val_set['predicted_interest'])), {'fontsize': 10}, fontproperties='monospace')
plt.axis('off')
plt.savefig(f'further_validation/results/classification_report_for_{file_name}.png')


constants.logger.info("generating confusion matrix and storing in results folder")
cm = pd.DataFrame(confusion_matrix(val_set['interest'], val_set['predicted_interest']), 
                  index = id_to_interest_dict.values(), columns = id_to_interest_dict.values())
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='g')
plt.title(f'confusion matrix')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
plt.savefig(f"further_validation/results/confusion_matrix_for_{file_name}.png")



