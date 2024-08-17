import pandas as pd
from sklearn.metrics import recall_score
import mlflow 

mlflow.set_tracking_uri("http://127.0.0.1:5000")

MLFLOW_EXPERIMENT_NAME = "NHTSA FARS Injury prediction"
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

# Define a function to log the model and metrics to mlflow
def log_pipeline_with_mlflow(model_name, model, model_params, preprocess_params, artifacts, X_train, X_val, y_train, y_val, model_artifact_path = 'model', is_validation_set_test = False):
    with mlflow.start_run():
        # Log the model type as a tag for easy filtering
        mlflow.set_tag(key = 'model_name', value = model_name)

        # Log preprocessing parameters
        mlflow.log_params(preprocess_params)
        
        # Log model parameters
        mlflow.log_params(model_params)

        # Set the parameters
        model.set_params(**model_params)
        
        # Fit the model
        model.fit(X_train, y_train)
        
        # Predict and calculate the wighted recall score for the training set
        y_train_pred = model.predict(X_train)
        train_weighted_recall = weighted_recall_score(y_train, y_train_pred)
        mlflow.log_metric('train_weighted_recall', train_weighted_recall)

        # Predict and calculate the wighted recall score for the validation set
        y_val_pred = model.predict(X_val)
        val_weighted_recall = weighted_recall_score(y_val, y_val_pred)

        if not is_validation_set_test:
            mlflow.log_metric('val_weighted_recall', val_weighted_recall)
        else:
            mlflow.log_metric('test_weighted_recall', val_weighted_recall)
        
        # Log the model
        if model_name == "XGBClassifier":
            mlflow.xgboost.log_model(model, model_artifact_path)
        else:
            mlflow.sklearn.log_model(model, model_artifact_path)

        # Log the artifacts
        for artifact in artifacts:
            if artifact['type'] == 'dict':
                mlflow.log_dict(dictionary = artifact['object'], artifact_file = artifact['file_name'])
            elif artifact['type'] == 'file':
                mlflow.log_artifact(local_path = artifact['local_path'], artifact_path = artifact['artifact_path'])


# We want to get the best models according tot he validation set metrics </br>
# And train them from scratch on the train and validation set so that we can evaluate them on the validation set
# 
# For this purpose we created several helper functions

