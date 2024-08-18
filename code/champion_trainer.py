import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
import utils.data_preprocess as data_prep
from utils.training import get_train_val_test_dfs, MLFLOW_EXPERIMENT_ID
from utils.training import retrain_pipeline, get_best_n_runs
import utils.evaluation as model_eval
from datetime import datetime, timedelta
import time

mlflow_client = MlflowClient()

def get_registered_model_run_id(model_name, model_alias = None, model_version = None):
    if model_alias:
        model_version_info = mlflow_client.get_model_version_by_alias(model_name, model_alias)
    elif model_version:
        model_version_info = mlflow_client.get_model_version(model_name, model_version)
    else:
        return None
    
    return model_version_info.run_id

def get_run_object(run_id):
    return mlflow_client.search_runs(experiment_ids=[MLFLOW_EXPERIMENT_ID],
                                     filter_string=f"attributes.run_id = '{run_id}'",)[0]

def get_yesterday_epoch():
    # Get the current time
    now = datetime.now()

    # Subtract one day (timedelta of 1 day)
    one_day_ago = now - timedelta(days=1)

    # Convert to epoch time (Unix timestamp)
    epoch_time_one_day_ago = int(time.mktime(one_day_ago.timetuple()))

    return epoch_time_one_day_ago * 1000

def alert_score_difference(challenger_run, eval_metric, eval_metric_diff, champion_score):
    challenger_run_id = challenger_run.info.run_id
    
    print(f'Warning! A model with the run id: {challenger_run_id} Got a {eval_metric} score {eval_metric_diff:.3f} higher than the current Champion Model\'s score of {champion_score:.3f}')

    return None

def monitor_champion_performance(champion_run_id: str):
    evaluation_metric = f'val_{model_eval.EVALUATION_METRIC}'
    
    champion_run = get_run_object(champion_run_id)
    champion_score = champion_run.data.metrics[evaluation_metric]
    
    top_runs = get_best_n_runs(experiment_id = MLFLOW_EXPERIMENT_ID,
                               metric_name = evaluation_metric,
                               n = 3,
                               is_higher_better = True,
                               filter_string = f"attributes.created > {get_yesterday_epoch()}",)

    best_challenger = None
    max_score = champion_score

    for run in top_runs:
        run_score = run.data.metrics[evaluation_metric]
        score_difference = champion_score - run_score

        if score_difference > model_eval.EVALUATION_METRIC_SIGNIFICANT_DIFF:
            alert_score_difference(run, model_eval.EVALUATION_METRIC, score_difference)

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

def retrain_model_by_run(model_run, train_df, val_df, is_validation_set_test):
    train_df, _, train_target_df, train_labels_dict, train_artifacts = data_prep.prep_training_datasets(train_df)
    val_df, _, val_target_df, _, _ = data_prep.prep_training_datasets(val_df)

    retrained_model_run_id = retrain_pipeline(model_run,
                                              train_df,
                                              train_target_df,
                                              val_df,
                                              val_target_df,
                                              train_artifacts,
                                              is_validation_set_test = is_validation_set_test)

    return retrained_model_run_id, train_labels_dict

def train_champion(champion_model_name, model_alias):
    train_df, val_df, test_df = get_train_val_test_dfs()
    train_val_df = pd.concat([train_df, val_df], ignore_index=True)

    champion_run_id = get_registered_model_run_id(champion_model_name, model_alias = model_alias)
    champion_run = get_run_object(champion_run_id)

    retrained_on_train_champion_run_id, _ = retrain_model_by_run(champion_run, train_df, val_df, is_validation_set_test = False)

    challenger_run = monitor_champion_performance(retrained_on_train_champion_run_id)

    if challenger_run:
        trained_challenger_run_id, train_val_labels_dict = retrain_model_by_run(challenger_run, train_val_df, test_df, is_validation_set_test = True)
        register_model(trained_challenger_run_id, champion_model_name, 'challenger', train_val_labels_dict)
    
    retrained_champion_run_id, train_val_labels_dict = retrain_model_by_run(champion_run, train_val_df, test_df, is_validation_set_test = True)
    register_model(retrained_champion_run_id, champion_model_name, model_alias, train_val_labels_dict)

if __name__ == "__main__":
    train_champion('starter_notebook_model', "Champion")
