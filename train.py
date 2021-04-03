import argparse
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from azureml.data.dataset_factory import TabularDatasetFactory
import pandas as pd
import joblib
from azureml.core.run import Run
run = Run.get_context()
url = 'https://azproject30213612965.blob.core.windows.net/train-data/train_data.csv'
data = TabularDatasetFactory.from_delimited_files(url)
df = data.to_pandas_dataframe()
# get the cleaned data
x = df.drop('class-label',axis=1).values
y = df['class-label'].values
# split the dataset
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2,random_state=1)
# create machine learning model
def main():
    # Add arguments to script for HyperDrive
    parser = argparse.ArgumentParser()
    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")
    args = parser.parse_args()
    run.log('Regularization Strength: ',args.C)
    run.log("Max iterations:", np.int(args.max_iter))

    # create model
    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(xtrain, ytrain)
    accuracy = model.score(xtest,ytest)
    run.log('Accuracy',np.float(accuracy))
    os.makedirs('./outputs', exist_ok=True)
    joblib.dump(value=model, filename='./outputs/model.joblib')
if __name__ == '__main__':
    main()