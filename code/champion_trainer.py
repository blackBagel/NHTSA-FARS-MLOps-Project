import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
import utils.data_preprocess as data_prep
from utils.training import get_train_val_test_dfs 
from utils.training import retrain_pipeline, get_best_n_runs, ml
from utils.evaluation import EVALUATION_METRIC

mlflow_client = MlflowClient()
MLFLOW_EXPERIMENT_NAME = 'NHTSA FARS Injury prediction'

def get_registered_model_run_id(model_name, model_alias = None, model_version = None):
    if model_alias:
        model_version_info = mlflow_client.get_model_version_by_alias(model_name, model_alias)
    elif model_version:
        model_version_info = mlflow_client.get_model_version(model_name, model_version)
    else:
        return None
    
    return model_version_info.run_id

def get_run_object(run_id):
    return mlflow_client.search_runs(filter_string=f"attributes.run_id = '{run_id}'")[0]

def alert_score_difference(challenger_run, champion_score):
    return None

def monitor_champion_performance(champion_run):
    champion_score = champion_run.data.metrics[f'test_{EVALUATION_METRIC}']
    
    top_runs = get_best_n_runs(experiment_name = MLFLOW_EXPERIMENT_NAME,
                               metric_name = EVALUATION_METRIC,
                               top_n = 3,
                               is_higher_better = True,)

    for run in top_runs:
        run_score = run.data.metrics[f'test_{EVALUATION_METRIC}']
        score_difference = champion_score - run_score

        best_challenger = None
        max_score = champion_score

        if score_difference > 0.05:
            alert_score_difference(run, champion_score)

            if run_score > max_score:
                max_score = run_score
                best_challenger = run

    return best_challenger

def register_model(run_id, name, alias, labels_dict):
    model_version = mlflow.register_model(
        model_uri=f"runs:/{run_id}/model",
        name=name
    )

    # Add tags to the registered model
    mlflow_client.set_registered_model_tag(name, "task", "classification")

    accident_result_unique_values_count = len(labels_dict)
    mlflow_client.set_registered_model_tag(name, "accident_result_unique_values_count", accident_result_unique_values_count)

    # Add Champion alias since it's the only model
    mlflow_client.set_registered_model_alias(name, alias, model_version.version)

def train_champion(champion_model_name, model_alias):
    train_df, val_df, test_df = get_train_val_test_dfs()

    train_df = pd.concat([train_df, val_df], ignore_index=True)

    train_df, _, train_target_df, train_labels_dict, train_artifacts = data_prep.prep_training_datasets(train_df)
    test_df, _, test_target_df, _, _ = data_prep.prep_training_datasets(test_df)
    
    champion_run_id = get_registered_model_run_id(champion_model_name, model_alias = model_alias)
    champion_run = get_run_object(champion_run_id)

    new_run_id = retrain_pipeline(champion_run, train_df, train_target_df, test_df, test_target_df, train_artifacts)
    
    register_model(new_run_id, champion_model_name, model_alias, train_labels_dict)
    # retrained_champion_run = get_run_object(new_run_id)

if __name__ == "__main__":
    train_champion('starter_notebook_model', "Champion")
