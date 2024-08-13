import os
import pandas as pd
from sklearn.metrics import recall_score
import mlflow
from sklearn_pandas import DataFrameMapper, gen_features
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
import re
from utils.data_preprocess import make_id_label_dict, add_accident_result_columns, prep_training_datasets

def get_models_datasets_dir(datasets_dir_relative_path = 'data/datasets', models_datasets_dir = 'for_models'):
    current_path = os.path.realpath(__file__)
    parent_dir = os.path.dirname(os.path.dirname(current_path))
    models_datasets_dir = os.path.join(parent_dir, datasets_dir_relative_path, models_datasets_dir)

    return models_datasets_dir


mlflow.set_tracking_uri("http://127.0.0.1:5000")

MLFLOW_EXPERIMENT_NAME = "NHTSA FARS Injury prediction"
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
