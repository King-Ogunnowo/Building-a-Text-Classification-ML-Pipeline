# Building-a-Financial-Text-Classifier
## 1. Introduction 
This repository contains scripts that can be used to build a financial text classifier. 
<p>Financial text classification is an application of text classification that can amonth other things be used to identify the intent/ purpose of a transaction<br>
<p>Financial text classification can take a binary, multi class, multi label classification process. In this case, a multi class classification approach is used.<br>
 
 
## 2. Requirements
To use these scripts, the following must be installed on your local machine:
 * scikit-learn
 * pandas
 * numpy
 * seaborn
 * matplotlib
 * argparse
 * wordcloud
 * logging

 Optional packages include:
 * DVC (Data Version Control, in case you wish to build an ML pipeline and run scripts like DAGs)
 You can easily install these by running this in your terminal ```pip install -r requirements.txt```
 
 3. How to use
 * install requirements
 * run scripts from Terminal with command line statements. With the cleaning script as an example, it can be run with this line of command 
 ```python cleaning.py --data_path "data_artifacts/training_set_new.csv" --text_column "column_name" --target_column "column_name"```
 * if unsure about the kind of arguements to pass in command, run ```python {script_name} --help```
