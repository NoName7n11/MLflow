# In this we will lurn how to deploy a registered model 
import mlflow.sklearn

model_uri = "models:/best-production-model/2"

mlflow_model = mlflow.sklearn.load_model(model_uri)
# This is mlflow_model will be used for "Batch Infrencing"

# Batch Inference: The process of running a trained machine learning model 
# or large language model on a large, pre-collected set of data all at once, 
# rather than generating single predictions in real time.

print(mlflow_model)
# Output: LogisticRegression(max_iter=1000, random_state=8888)
# The above method is used to load an already "registed model or logged model" for "Batch Inference"


# For real-time inferencing we set up a server like this:
# mlflow models serve -m "models:/best-production-model/2" -p 4560 --env-manager local
# mlflow models serve -m "models:/<model_name>/version" -p <port> --env-manager local

