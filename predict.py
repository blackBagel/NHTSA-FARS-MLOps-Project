import os
import pandas as pd
import mlflow
from flask import Flask, request, jsonify


REGISTERED_MODEL_NAME = os.getenv('MODEL_NAME')
ALIAS = os.getenv('MODEL_ALIAS')

BUCKET_PATH = os.getenv('BUCKET_PATH', 'gs://mlops_zoomcamp-mlflow-artifacts/artifacts')
EXPERIMENT_NAME = os.getenv('EXPERIMENT_NAME')
EXPERIMENT_ID = os.getenv('EXPERIMENT_ID')
RUN_ID = os.getenv('RUN_ID')

MODEL_ARTIFACT_PATH = os.getenv('MODEL_ARTIFACT_PATH')
MODEL_LABELS_DICT_FILE = os.getenv('MODEL_LABELS_DICT_FILE', 'NHTSA_FARS_labels_for_target.json')

INDEX_COLUMNS = [
    'ST_CASE',
    'VEH_NO',
    'PER_NO'
]

# This is actually preferable because it takes out the dependency on the tracking server
if RUN_ID and BUCKET_PATH and EXPERIMENT_NAME:
    # mlflow.set_tracking_uri("http://127.0.0.1:5000")
    # client = mlflow.tracking.MlflowClient()
    # experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    # experiment_id = experiment.experiment_id
    
    artifacts_uri = f'{BUCKET_PATH}/{EXPERIMENT_ID}/{RUN_ID}/artifacts'
    model_uri = f'{artifacts_uri}/{MODEL_ARTIFACT_PATH}'
# WIP: transfer this to a separate component that periodically updates the run_id when necessary
# elif REGISTERED_MODEL_NAME:
#     model_uri = f"models:/{REGISTERED_MODEL_NAME}@{ALIAS}"
else:
    raise Exception("There are no model name or run id env variables in this evironment. Exiting")

model = mlflow.pyfunc.load_model(model_uri)
labels_for_predictions = mlflow.artifacts.load_dict(f'{artifacts_uri}/{MODEL_LABELS_DICT_FILE}')
print(labels_for_predictions)

def prepare_features(features):
    features_df = pd.DataFrame([features])
    # features_df_idx = features_df[INDEX_COLUMNS]
    # features_df = features_df.drop(INDEX_COLUMNS, axis=1)

    return features_df # , features_df_idx

def predict(features):
    prediction = model.predict(features)
    text_preditcion = labels_for_predictions[str(prediction[0])]
    
    return text_preditcion


app = Flask('injury-prediction')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    passenger = request.get_json()
    print(passenger)

    passenger = prepare_features(passenger)

    prediction = predict(passenger)

    result = {
        'injury': prediction,
        'model_version': RUN_ID
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)