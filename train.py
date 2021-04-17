
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import re
import os
from sklearn.metrics import recall_score

import argparse
import joblib
from azureml.core.run import Run

from feature_preprocessing import *

# We loaded the dataset into Azure and we are grabbing it here.
from azureml.core import Workspace, Dataset

run = Run.get_context()
ws = run.experiment.workspace

key = 'Pump-it-Up-dataset'

if key in ws.datasets.keys():
      dataset = ws.datasets[key]
      print('dataset found!')

else:
      url = 'https://raw.githubusercontent.com/JayThibs/hyperdrive-vs-automl-plus-deployment/main/Pump-it-Up-dataset.csv'
      dataset = Dataset.Tabular.from_delimited_files(url)
      datatset = dataset.register(ws, key)

dataset.to_pandas_dataframe()
X = dataset.to_pandas_dataframe()
y = X[['status_group']]
del X['status_group']

# Cleaning up the features of our dataset
X = bools(X)
X = locs(X)
X = construction(X)
X = removal(X)
X = dummy(X)
X = dates(X)
x = dates2(X)
X = small_n(X)

# Removing ">", "[" and "]" from the headers to make the data compatible with different algorithms (namely, xgboost)
regex = re.compile(r"\[|\]|<", re.IGNORECASE)
X.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in X.columns.values]

# Converting the population values to log
X['population'] = np.log(X['population'])

# Splitting the dataset into a training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def main():
    # Adds arguments to script
    parser = argparse.ArgumentParser()

    # Setting the hyperparameters we will be optimizing for you Random Forest model
    parser.add_argument('--max_depth', type=int, default=6, help='The maximum depth of the trees.')
    parser.add_argument('--min_samples_split', type=int, default=4, help='The minimum number of samples required to split an internal node.')
    parser.add_argument('--n_estimators', type=int, default=750, help='The number of trees in the forest.')

    args = parser.parse_args()

    run.log("Max depth of the trees:", np.int(args.max_depth))
    run.log("Minimum number of samples required to split:", np.int(args.min_samples_split))
    run.log("Number of trees:", np.int(args.n_estimators))

    # Fitting a Random Forest model to our data. 
    # Sidenote: I also tried XGBoost on my local machine, but it did not perform as well.
    # RF has a score of 0.811, XGBoost has a score of 0.745
    # Since I did not use a validation set, it's possible that I'm just overfitting with RF.
    # But I wanted to focus on the end-to-end process for this project so I didn't bother with 
    # a validation set.
    rf = RandomForestClassifier(max_depth=args.max_depth,
                                min_samples_split=args.min_samples_split,
                                n_estimators=args.n_estimators,
                                criterion='gini',
                                oob_score=True,
                                random_state=42,
                                n_jobs=-1).fit(X_train, y_train.values.ravel())
    
    # Predicting on the test set
    predictions = rf.predict(X_test)
    pred = pd.DataFrame(predictions)

    # Calculate recall to test how well we do on True Positives
    # We can imagine a real scenario where we want to build a model
    # that does not miss the non-functioning water pumps, and we
    # care much less functioning water pumps that are incorrectly
    # predicted as non-functional. 
    recall_micro = recall_score(y_test, pred, average='micro')
    run.log("Recall_Micro", np.float(recall_micro))

    os.makedirs('outputs', exist_ok=True)
    joblib.dump(rf, 'outputs/rf_model.pkl')

if __name__ == '__main__':
    main()