import os
import pandas as pd
import mlflow
from flask import jsonify


INDEX_COLUMNS = [
    'ST_CASE',
    'VEH_NO',
    'PER_NO'
]

def get_model_artifacts_location(run_id):
    model_location = os.getenv('MODEL_LOCATION')
    model_labels_dict_location = os.getenv('MODEL_LABELS_DICT_LOCATION')

    if model_location is not None and model_labels_dict_location is not None:
        return model_location, model_labels_dict_location

    bucket_path = os.getenv('MLFLOW_BUCKET_PATH')
    experiment_id = os.getenv('MLFLOW_EXPERIMENT_ID')
    model_artifact_path = os.getenv('MODEL_ARTIFACT_PATH')
    model_labels_dict_file = os.getenv('MODEL_LABELS_DICT_FILE')

    artifacts_location = f'{bucket_path}/{experiment_id}/{run_id}/artifacts'

    model_labels_dict_location = f'{artifacts_location}/{model_labels_dict_file}'
    model_location = f'{artifacts_location}/{model_artifact_path}'
    return model_location, model_labels_dict_location


def load_model_artifacts(run_id):
    model_path, model_labels_dict_path = get_model_artifacts_location(run_id)
    
    model = mlflow.pyfunc.load_model(model_path)
    labels_for_predictions = mlflow.artifacts.load_dict(model_labels_dict_path)
    return model, labels_for_predictions


class ModelService:
    def __init__(self, model, labels_for_predictions, model_version=None):
        self.model = model
        self.labels_for_predictions = labels_for_predictions
        self.model_version = model_version

    def prepare_features(self, features):
        features_df = pd.DataFrame([features])

        features_IDs_df = features_df[INDEX_COLUMNS]
        features_df = features_df.drop(INDEX_COLUMNS, axis=1)

        features_IDs = features_IDs_df.iloc[0, :].to_dict()

        return features_df, features_IDs

    def predict(self, features):
        prediction = self.model.predict(features)
        prediction = str(prediction[0])

        text_prediction = self.labels_for_predictions[prediction]
        
        return text_prediction

    def request_handler(self, passenger):
        passenger, passenger_IDs = self.prepare_features(passenger)

        prediction = self.predict(passenger)

        result = {
            'prediction': prediction,
            'passenger_identifiers': passenger_IDs,
            'model_version': self.model_version,
            'model': 'accident_injury_prediction_model',
        }

        return jsonify(result)



def init(run_id: str):
    model, labels_for_prediction = load_model_artifacts(run_id)

    model_service = ModelService(model=model, labels_for_predictions=labels_for_prediction, model_version=run_id)

    return model_service