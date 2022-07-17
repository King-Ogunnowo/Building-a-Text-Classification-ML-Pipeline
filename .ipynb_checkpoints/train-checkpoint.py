"""
Function to train scikit-learn algorithm

Author: Oluwaseyi E. Ogunnowo
Date: 8th July 2022
"""

import argparse
import pickle as pkl
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import constants


parser = argparse.ArgumentParser(description = "Script to visualize data properties.\
NB: data passed must be saved in .csv format")

parser.add_argument("--data_path",
                    type = str,
                    help = 'path to training sample')

parser.add_argument("--independent_feature",
                    type = str,
                    help = 'column in dataframe containing texts')

parser.add_argument("--dependent_feature",
                    type = str,
                    help = 'column containing encoded target values,\
                    i.e. the values you want to predict')

parser.add_argument("--test_size",
                   type = float,
                   help = "size of test set")

args = parser.parse_args()

stop_words = constants.stop_words
constants.logger.info("reading cleaned training data")

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

sgd_clf = SGDClassifier(loss = 'hinge', random_state = 42)
tfidf = TfidfVectorizer(sublinear_tf=True,
                        min_df=2, norm='l2',
                        encoding='latin-1',
                        ngram_range=(1, 2),
                        stop_words=stop_words,
                        lowercase = False)


df = pd.read_csv(args.data_path)

x = df[args.independent_feature]
y = df[args.dependent_feature]

constants.logger.info("splitting data into train and test sets")
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = args.test_size, stratify = y)

x_train_trans = tfidf.fit_transform(x_train)

constants.logger.info("training classifier")
sgd_clf.fit(x_train_trans, y_train)

constants.logger.info("compiling trained classifier and vectorizer into model pipeline")
model_pipeline = Pipeline(steps=[('vectorizer', tfidf),
                                 ('classifier', sgd_clf)])
model_pipeline.fit(x_train, y_train)


constants.logger.info("exporting model pipeline")
export_object(model_pipeline, 'model/model_pipeline.pkl')

constants.logger.info("exporting test sets")
x_test = pd.DataFrame(x_test)
x_test[args.dependent_feature] = y_test
x_test.to_csv("data_artifacts/test_set.csv")

