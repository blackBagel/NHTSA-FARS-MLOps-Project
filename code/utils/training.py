import os
import pandas as pd
import mlflow
from sklearn_pandas import DataFrameMapper
from sklearn.pipeline import Pipeline
import re
from utils.evaluation import weighted_recall_score

PROJECT_PATH = os.getenv('PROJECT_PATH')
MODELS_DATASETS_PATH = os.getenv('MODELS_DATASETS_PATH',
                                 os.path.join(PROJECT_PATH, 'data', 'datasets', 'for_models'),)
TRAIN_FILENAME = os.getenv('TRAIN_FILENAME', 'train.csv')
VALIDATION_FILENAME = os.getenv('VALIDATION_FILENAME', 'validation.csv')
TEST_FILENAME = os.getenv('TEST_FILENAME', 'test.csv')

# Will be set using the MLFLOW_TRACKING_URI 
# mlflow.set_tracking_uri("http://127.0.0.1:5000")

if os.getenv('MLFLOW_TRACKING_URI'):
    MLFLOW_EXPERIMENT_ID = os.getenv('MLFLOW_EXPERIMENT_ID')
    mlflow.set_experiment(experiment_id = MLFLOW_EXPERIMENT_ID)
else:
    raise Exception("No mlflow tracking uri set!")

def get_train_val_test_dfs():
    train_path = os.path.join(MODELS_DATASETS_PATH, TRAIN_FILENAME)
    val_path = os.path.join(MODELS_DATASETS_PATH, VALIDATION_FILENAME)
    test_path = os.path.join(MODELS_DATASETS_PATH, TEST_FILENAME)

    train_df = pd.read_csv(train_path, encoding='Windows-1252')
    val_df = pd.read_csv(val_path, encoding='Windows-1252')
    test_df = pd.read_csv(test_path, encoding='Windows-1252')

    return train_df, val_df, test_df

# Define a function to log the model and metrics to mlflow
def log_pipeline_with_mlflow(model_name, model, model_params, preprocess_params, artifacts, X_train, X_val, y_train, y_val, model_artifact_path = 'model', is_validation_set_test = False):
    """
    Logs a machine learning pipeline to MLflow, including model parameters, metrics, and artifacts.

    Args:
    model_name (str): Name of the model being logged (used as a tag in MLflow).
    model (Pipeline): The machine learning pipeline containing preprocessing and model steps.
    model_params (dict): Dictionary of hyperparameters to set in the model before training.
    preprocess_params (dict): Dictionary of parameters related to the preprocessing steps in the pipeline.
    artifacts (list of dicts): List of artifacts to log with the model in MLflow, where each artifact dictionary contains:
        - 'type': Type of artifact ('dict' or 'file').
        - 'object': The dictionary object to log (if type is 'dict').
        - 'file_name': The file name for the logged dictionary (if type is 'dict').
        - 'local_path': Path to the file to log (if type is 'file').
        - 'artifact_path': Path within the artifact store where the file will be stored (if type is 'file').
    X_train (DataFrame or array): Training feature data.
    X_val (DataFrame or array): Validation feature data.
    y_train (Series or array): Training labels.
    y_val (Series or array): Validation labels.
    model_artifact_path (str, default='model'): Path within the MLflow artifact store to save the model.
    is_validation_set_test (bool, default=False): Flag indicating whether the validation set should be used as a test set for final evaluation.

    Returns:
    Logged mlflow run_id
    """
    
    with mlflow.start_run() as run:
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

        run_id = run.info.run_id
    
    return run_id


# Train and log each model using a pipeline
def train_and_log_pipelines(models, model_pipe_step_name, data_prep_steps, preprocessing_params, artifacts, X_train, X_val, y_train, y_val, is_validation_set_test = False):
    """
    Trains multiple machine learning models with different hyperparameter combinations, constructs pipelines, and logs the results to MLflow.

    Args:
    models (dict): Dictionary where keys are model names and values are tuples containing a model instance and a list of hyperparameter dictionaries.
    model_pipe_step_name (str): Name to assign to the model step in the pipeline.
    data_prep_steps (list of tuples): List of tuples representing the preprocessing steps in the pipeline.
    preprocessing_params (dict): Dictionary containing parameters specific to preprocessing steps.
    artifacts (dict): Dictionary of artifacts (e.g., feature names, preprocessing objects) to log with the model in MLflow.
    X_train (DataFrame or array): Training feature data.
    X_val (DataFrame or array): Validation feature data.
    y_train (Series or array): Training labels.
    y_val (Series or array): Validation labels.
    is_validation_set_test (bool, default=False): Flag indicating whether the validation set should also be used as a test set for final evaluation.

    Returns:
    None
    """
    
    # Iterate over each model in the models dictionary
    for model_name, (model, model_params_combinations) in models.items():
        # Iterate over each hyperparameter combination for the current model
        for model_params_combination in model_params_combinations:
            # Copy the data preprocessing steps to avoid mutating the original list
            pipeline_steps = data_prep_steps.copy()
            
            # Append the model and its step name to the pipeline steps
            pipeline_steps.append((model_pipe_step_name, model))
            
            # Create the pipeline with the combined preprocessing and model steps
            pipeline = Pipeline(pipeline_steps)

            # Log the pipeline, model, and training details with MLflow
            log_pipeline_with_mlflow(model_name = model_name,
                                    model = pipeline,
                                    model_params = model_params_combination,
                                    preprocess_params = preprocessing_params,
                                    artifacts = artifacts,
                                    X_train = X_train,
                                    X_val = X_val,
                                    y_train = y_train,
                                    y_val = y_val,
                                    is_validation_set_test=is_validation_set_test)

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

def retrain_pipeline(run, X_train, y_train, X_test, y_test, artifacts):
    run_id = run.info.run_id
    model_name = run.data.tags['model_name']
    original_pipeline = load_pipeline_from_run(run_id, model_name)
    
    # Create a new pipeline with untrained steps and parameters
    # new_pipeline = create_fresh_pipeline(original_pipeline)

    # Retrieve the model parameters
    pipeline_params, non_pipeline_params = filter_dict_by_regex(run.data.params, f'^\w+__')
    
    # Convert model params to their original types
    pipeline_params = convert_params_to_original_types(pipeline_params, original_pipeline)
    
    # Retrain the new pipeline with the new data
    new_run_id = log_pipeline_with_mlflow(model_name = model_name,
                                            model = original_pipeline,
                                            model_params = pipeline_params,
                                            preprocess_params = non_pipeline_params,
                                            artifacts = artifacts,
                                            X_train = X_train,
                                            X_val = X_test,
                                            y_train = y_train,
                                            y_val = y_test,
                                            is_validation_set_test=True,)
    
    return new_run_id

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
    None
    """
    top_runs = get_best_n_runs(experiment_name, metric_name, top_n, is_higher_better = is_higher_better)

    for run in top_runs:
        _ = retrain_pipeline(run, X_train, y_train, X_test, y_test, artifacts)