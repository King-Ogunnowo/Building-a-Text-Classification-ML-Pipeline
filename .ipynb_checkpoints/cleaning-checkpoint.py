"""
Cleaning script to clean and process text. 1st stage in the DVC pipeline.

Author: Oluwaseyi E. Ogunnowo
Date: 8th July 2022
"""

# --- importing dependencies
import re
import argparse
import pickle as pkl
import pandas as pd
import constants


def clean_text(text: str):
    """
    Function to clean texts:
    - Changes text case to lowercase
    - Removes stopwords
    - Removes unwanted characters
    Parameters
    ----------
    text: (string) unstructured texts
    returns
    ----------
    cleaned_text: (string) cleaned unstructured texts
    """
    cleaned_text = text.lower()
    cleaned_text = re.sub("[0-9]", " ", cleaned_text)
    cleaned_text = re.sub("[^a-z]", " ", cleaned_text)
    cleaned_text = [i for i in cleaned_text.split() if i not in stop_words and len(i) > 2]
    cleaned_text = ' '.join(cleaned_text)
    return cleaned_text

def export_object(object_to_pickle: dict, file_name: str):
    """
    Function to pickle/ serialize objects
    Parameters
    ----------
    - data: Object to pickle. This function can pickle dictionary objects as well as ML models
    - file_name: (string) File name to save object as in working directory.
    """
    with open(file_name, 'wb') as _object:
        pkl.dump(object_to_pickle, _object)

        
# --- setting argparse configurations 
parser = argparse.ArgumentParser(description = "Script to clean and process training data.\
NB: data passed must be saved in .csv format")

parser.add_argument("--data_path", 
                    type = str, 
                    help = 'path to training sample')

parser.add_argument("--text_column", 
                    type = str, 
                    help = 'column in dataframe containing texts')

parser.add_argument("--target_column", 
                    type = str,
                    help = 'column containing target values, i.e. the values you want to predict')

args = parser.parse_args()


# --- reading data
constants.logger.info("reading training data")
df = pd.read_csv(args.data_path)
df = df.sample(frac = 1)
df = df.reset_index(drop = True)


# --- cleaning text
stop_words = constants.stop_words


constants.logger.info("cleaning training data, removing stopwords and unwanted characters")
df["cleaned_" + args.text_column] = df[args.text_column].apply(lambda x: clean_text(x))


constants.logger.info("encoding interests and creating dictionary objects")
df[args.target_column + "_id"] = df[args.target_column].factorize()[0]
mapped_interest_df = df[[args.target_column + "_id", args.target_column]]\
.drop_duplicates().sort_values(args.target_column + "_id")
mapped_interest_df = mapped_interest_df.reset_index(drop = True)
interest_to_id = dict(mapped_interest_df.values)
id_to_interest = dict(mapped_interest_df[[args.target_column + "_id", args.target_column]].values)


constants.logger.info("saving dictionary objects")
export_object(interest_to_id, file_name = 'dictionary/value_to_id.pkl')
export_object(id_to_interest, file_name = 'dictionary/id_to_value.pkl')


# --- exporting cleaned dataframe as csv file
constants.logger.info("exporting cleaned data as csv file")
df.to_csv("data_artifacts/cleaned_data.csv", index = False)







