import mlflow
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the Iris dataset
X, y = datasets.load_iris(return_X_y = True)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model hyperparameters
params = {
    "solver": "lbfgs",
    "max_iter": 1000,
    "random_state": 8888,
}

#Setting the MLflow Experiment
mlflow.set_experiment("iris")

# This is manual logging 
#with mlflow.start_run(run_name = "run_iris"):
#    mlflow.log_params(params)
#    lr = LogisticRegression(**params)
#    lr.fit(X_train, y_train)
#
#    mlflow.sklearn.log_model(sk_model = lr, name = "simple_model")

# This is auto logging 
mlflow.sklearn.autolog()

with mlflow.start_run(run_name = "run_iris_auto"):
    lr = LogisticRegression(**params)
    lr.fit(X_train, y_train)

    mlflow.sklearn.log_model(sk_model=lr, name="adcb", registered_model_name="best-production-model")

    # If you have a model with same name like "best-production-model" in registered_model_name then
    # it will add more versions to it instead of making new model

# Modle URI (Uniform Resource Identifier)
# This is used to reference a registerd model 

models: /best-production-model/3  #This is used when there is need of deploying a model 