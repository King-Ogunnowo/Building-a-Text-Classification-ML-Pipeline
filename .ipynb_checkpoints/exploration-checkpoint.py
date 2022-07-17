"""
Data Exploration script to visualize text data, 2nd stage in the pipeline 

Author: Oluwaseyi E. Ogunnowo
Date: 8th July 2022
"""

# --- importing dependencies 
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from tqdm import tqdm
import constants

plt.rcParams.update({'figure.max_open_warning': 0})

def create_word_cloud(df: pd.DataFrame, text_column: str, interest_column:str):
    """
    Function to visualize frequent words in wordcloud for each unique value in target column
    Parameters
    ----------
    df: (Pandas DataFrame) DataFrame object contained cleaned data
    text_column: (string) Name of column containing cleaned texts
    target_column: (string) Name of column containing target values
    """
    for interest in tqdm(df[interest_column].unique(), desc = 'wordcloud progress'):
        data_subset = df.loc[df[interest_column] == interest]
        all_texts = ' '.join([text for text in data_subset[text_column]])
        plt.figure(figsize = (15, 12))
        wordcloud = WordCloud(background_color="white").generate(all_texts)
        plt.title(f"{interest}")
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.savefig(f"eda_results/wordclouds/{interest}_wordcloud.png")
        
parser = argparse.ArgumentParser(description = "Script to visualize data properties.\
NB: data passed must be saved in .csv format")

parser.add_argument("--data_path",
                    type = str,
                    help = 'path to training sample')

parser.add_argument("--text_column",
                    type = str,
                    help = 'column in dataframe containing texts')

parser.add_argument("--target_column",
                    type = str,
                    help = 'column containing target values,\
                    i.e. the values you want to predict')

args = parser.parse_args()

constants.logger.info("reading cleaned data")
df = pd.read_csv(args.data_path)

constants.logger.info("plotting class distribution in training dataset")
plt.figure(figsize = (15,10))
sns.countplot(y = args.target_column, data = df)
plt.savefig("eda_results/class_distribution_chart.png")

constants.logger.info(f"plotting wordcloud for each unique value in {args.target_column}")
create_word_cloud(df, args.text_column, args.target_column)


    
