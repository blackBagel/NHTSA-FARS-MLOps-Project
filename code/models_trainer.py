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


def get_best_n_runs(experiment_name, metric_name, n=3, is_higher_better = True):
    """
    Retrieve the top n runs based on a specified metric.

    Args:
    experiment_name (str): The name of the MLflow experiment.
    metric_name (str): The name of the metric to sort by.
    n (int): The number of top runs to retrieve.

    Returns:
    List of run info of the top n runs.
    """
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    sort_order = 'DESC' if is_higher_better else 'ASC'

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric_name} {sort_order}"],
        max_results=n,
        filter_string="attributes.status = 'FINISHED'"
    )
    return runs

def load_pipeline_from_run(run_id, model_name, model_artifact_path = 'model'):
    """
    Load a pipeline artifact from an MLflow run.

    Args:
    run_id (str): The run ID to load the pipeline from.
    model_name (str): The name of the pipeline artifact.
    model_artifact_path (str): The artifact path for the model. Default Value='models'

    Returns:
    The loaded pipeline.
    """
    model_uri = f"runs:/{run_id}/{model_artifact_path}"

    if model_name == "XGBClassifier":
        pipeline = mlflow.xgboost.load_model(model_uri)
    else:
        pipeline = mlflow.sklearn.load_model(model_uri)
    
    return pipeline

def recreate_dataframe_mapper(original_mapper):
    """
    Recreate a DataFrameMapper with the same parameters and settings.

    Args:
    original_mapper (DataFrameMapper): The original DataFrameMapper.

    Returns:
    The new DataFrameMapper with the same parameters and settings.
    """
    return DataFrameMapper(
        original_mapper.features,
        input_df=original_mapper.input_df,
        df_out=original_mapper.df_out
    )

def recreate_pipeline_steps(original_pipeline):
    """
    Recreate the pipeline steps with untrained instances.

    Args:
    original_pipeline (Pipeline): The original pipeline.

    Returns:
    A list of steps with untrained instances.
    """
    new_steps = []
    for name, step in original_pipeline.steps:
        if isinstance(step, DataFrameMapper):
            new_step = recreate_dataframe_mapper(step)
        else:
            new_step = step.__class__()
        new_steps.append((name, new_step))
    return new_steps

def create_fresh_pipeline(original_pipeline):
    """
    Create a new pipeline with the same steps and given parameters.

    Args:
    original_pipeline (Pipeline): The original pipeline.

    Returns:
    A new pipeline with the same exact steps.
    """
    steps = recreate_pipeline_steps(original_pipeline)
    new_pipeline = Pipeline(steps)
    return new_pipeline

def infer_param_types(pipeline):
    """
    Infer parameter types based on an existing pipeline's parameter values.
    """
    original_params = pipeline.get_params()
    param_types = {key: type(value) for key, value in original_params.items()}
    return param_types

def convert_params_to_original_types(params, pipeline):
    """
    Convert parameter values to their original types based on the model class.
    """
    param_types = infer_param_types(pipeline)
    converted_params = {}
    
    for param_name, value in params.items():
        # param_name = key.split("__")[-1]
        if param_name in param_types:
            param_type = param_types[param_name]
            converted_params[param_name] = param_type(value)
        else:
            converted_params[param_name] = value
    return converted_params

def filter_dict_by_regex(input_dict, pattern):
    """
    Filters a dictionary by keys that match a given regex pattern.

    Args:
    input_dict (dict): The input dictionary to be filtered.
    pattern (str): The regex pattern to match the keys.

    Returns:
    tuple: Two dictionaries, the first with key-value pairs where the keys match the regex pattern,
           and the second with key-value pairs where the keys do not match the regex pattern.
    """
    regex = re.compile(pattern)
    matching_dict = {}
    non_matching_dict = {}
    
    for key, value in input_dict.items():
        if regex.match(key):
            matching_dict[key] = value
        else:
            non_matching_dict[key] = value
    
    return matching_dict, non_matching_dict


def retrain_top_pipelines(experiment_name, metric_name, top_n, is_higher_better, X_train, y_train, X_test, y_test, artifacts):
    """
    Retrain the top n pipelines based on a specified metric with new data.

    Args:
    experiment_name (str): The name of the MLflow experiment.
    metric_name (str): The name of the metric to sort by.
    top_n (int): The number of top pipelines to retrain.
    X_train (pd.DataFrame): The new training features.
    y_train (pd.Series): The new training labels.
    X_test (pd.DataFrame): The test set for comparing our models.
    y_test (pd.Series): The test set labels.

    Returns:
    A list of retrained pipelines.
    """
    top_runs = get_best_n_runs(experiment_name, metric_name, top_n, is_higher_better = is_higher_better)

    for run in top_runs:
        run_id = run.info.run_id
        model_name = run.data.tags['model_name']
        original_pipeline = load_pipeline_from_run(run_id, model_name)
        
        # Create a new pipeline with untrained steps and parameters
        new_pipeline = create_fresh_pipeline(original_pipeline)

        # Retrieve the model parameters
        pipeline_params, non_pipeline_params = filter_dict_by_regex(run.data.params, f'^\w+__')
        
        # Convert model params to their original types
        pipeline_params = convert_params_to_original_types(pipeline_params, original_pipeline)
        
        # Retrain the new pipeline with the new data
        log_pipeline_with_mlflow(model_name = model_name,
                        model = new_pipeline,
                        model_params = pipeline_params,
                        preprocess_params = non_pipeline_params,
                        artifacts = artifacts,
                        X_train = X_train,
                        X_val = X_test,
                        y_train = y_train,
                        y_val = y_test,
                        is_validation_set_test=True,)


# Retraining top models on train+val single dataset
train_df = pd.read_csv(TRAIN_PATH, encoding='Windows-1252')
val_df = pd.read_csv(VALIDATION_PATH, encoding='Windows-1252')
test_df = pd.read_csv(TEST_PATH, encoding='Windows-1252')


train_df['year'] = 2019
val_df['year'] = 2021
final_train_index_columns = INDEX_COLUMNS + ['year']

final_train_df = pd.concat([train_df, val_df], ignore_index=True)

# Seperating the target column    
final_train_df, final_train_indices_df, final_train_target_df, final_train_label_for_target = prep_training_datasets(final_train_df, final_train_index_columns)
test_df, test_indices_df, test_target_df, test_label_for_target = prep_training_datasets(test_df, INDEX_COLUMNS)

# Make sure the training and validation possible model labels are the same
# assert train_label_for_target == val_label_for_target


experiment_name = MLFLOW_EXPERIMENT_NAME
metric_name = "val_weighted_recall"
artifacts = [
    { 
        'type': 'dict',
        'object': final_train_label_for_target,
        'file_name': 'NHTSA_FARS_labels_for_target.json',
    },
]

retrain_top_pipelines(experiment_name = experiment_name,
                      metric_name = metric_name,
                      top_n=3,
                      is_higher_better = True,
                      X_train = final_train_df,
                      y_train = final_train_target_df,
                      X_test = test_df,
                      y_test = test_target_df,
                      artifacts = artifacts,)


# #### Register the best model and set it as the champion 

# In[43]:


# Select the model with the highest test weighted_recall
client = MlflowClient()

# Retrieve the top_n model runs and log the models
experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)

best_run = client.search_runs(
    experiment_ids=experiment.experiment_id,
    run_view_type=ViewType.ACTIVE_ONLY,
    max_results=1,
    order_by=["metrics.test_weighted_recall DESC"],
    filter_string="attributes.status = 'FINISHED'"
)[0]

# Register the best model
registered_model_name = 'starter_notebook_model'

model_version = mlflow.register_model(
    model_uri=f"runs:/{best_run.info.run_id}/model",
    name=registered_model_name
)

# Add tags to the registered model
client.set_registered_model_tag(registered_model_name, "task", "classification")

accident_result_unique_values_count = len(final_train_label_for_target)
client.set_registered_model_tag(registered_model_name, "accident_result_unique_values_count", accident_result_unique_values_count)

# Add Champion alias since it's the only model
client.set_registered_model_alias(registered_model_name, "Champion", model_version.version) 

# train_and_log_pipelines(
#     models = models,
#     data_prep_steps = data_prep_steps,
#     preprocessing_params = preprocessing_params,
#     artifacts = artifacts,
#     X_train = train_df,
#     X_val = val_df,
#     y_train = train_target_df,
#     y_val = val_target_df,
#     is_validation_set_test = False)


val_df = pd.read_csv(VALIDATION_PATH, encoding='Windows-1252')
train_df, train_indices_df, train_target_df, train_label_for_target = prep_training_datasets(df, INDEX_COLUMNS)
val_df, val_indices_df, val_target_df, val_label_for_target = prep_training_datasets(val_df, INDEX_COLUMNS)

train_df_with_idx = train_indices_df.join(train_df, how='inner')
val_df_with_idx = val_indices_df.join(val_df, how='inner')


