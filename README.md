# Building-a-Text-Classification-Pipeline
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
 
 
 ## 3. How to use
 * install requirements
 * run scripts from Terminal with command line statements. With the cleaning script as an example, it can be run with this line of command 
 ```python cleaning.py --data_path "data_artifacts/training_set_new.csv" --text_column "column_name" --target_column "column_name"```
 * if unsure about the kind of arguements to pass in command, run ```python {script_name} --help```

 
 ## 4. Building reproducible pipeline with DVC 
 The following steps can help you build a reproducible pipeline of these scripts with DVC 
 * run ```git init```
 * run ```dvc init```
 * run ```dvc remote add {name of remote folder} {path to remote folder}``` (optional step, but advised in case you want to experiment with pipeline versions). See [here](https://dvc.org/doc/command-reference/remote#:~:text=What%20is%20a%20%22local%20remote,for%20DVC%20projectsDVC%20projects.) for more details
 * add scripts as stages to the pipeline with either of the following syntax: 
 ```dvc stage add -n <name of stage> -d <dependency(ies)> -o <output(s)> command <python script.py --parameters "parameters">```
 <center> -or- </center>
 ```dvc run -n <name of stage> -d <dependency(ies)> -o <output(s)> command <python script.py --parameters "parameters">``` While running each stage, it is automatically added by DVC into a dvc.yaml file
 
 
 
