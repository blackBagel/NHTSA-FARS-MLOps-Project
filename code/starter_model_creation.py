#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import recall_score
import mlflow
from sklearn_pandas import DataFrameMapper, gen_features
from sklearn.base import TransformerMixin
# from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
import re

# In[3]:


DATASET_DIR = '../data/datasets/'
data_file_name = 'person.csv'

TRAIN_PATH = f'{DATASET_DIR}/2018/{data_file_name}'
VALIDATION_PATH = f'{DATASET_DIR}/2019/{data_file_name}'
TEST_PATH = f'{DATASET_DIR}/2021/{data_file_name}'

INVESTIGATED_COLUMNS = [
    'ST_CASE', 
    'STATE',
    'STATENAME',
    'VEH_NO',
    'VE_FORMS',
    'PER_NO',
    'COUNTY',
    'DAY',
    'MONTH',
    'HOUR',
    'AGE',
    'SEX',
    'INJ_SEV',
    'INJ_SEVNAME',
    'DOA',
    'DOANAME',
    'SEAT_POS',
    'REST_USE',
    'REST_MIS',
    'HELM_USE',
    'HELM_MIS',
    'AIR_BAG',
    'EJECTION',
]

INDEX_COLUMNS = [
    'ST_CASE',
    'VEH_NO',
    'PER_NO'
]

CATEGORY_LABEL_COLUMNS = ['STATENAME', 'INJ_SEV', 'INJ_SEVNAME',  'DOA', 'DOANAME']
COLUMNS_NOT_FOR_MODEL = CATEGORY_LABEL_COLUMNS + INDEX_COLUMNS
COLUMNS_FOR_MODEL = [column for column in INVESTIGATED_COLUMNS if column not in COLUMNS_NOT_FOR_MODEL]

# #### Dropping irrelevant injury categories
# 
# The dataset user manual states these are the possible values for the injury severity field (`INJ_SEV`):
# 
# - 0 - No Apparent Injury
# - 1 - Possible Injury
# - 2 - Suspected Minor Injury
# - 3 - Suspected Serious Injury
# - 4 - Fatal Injury 
# - 5 - Injured, Severity Unknown
# - 6 - Died Prior to Crash
# - 9 - Unknown/Not Reported 
# 
# Since we want to teach our model to predict a specific injury severity, we'll only use categories 0-4
# 
# In addition, we'll consider death as another type of injury. </br>
# The user manual describes the column detailing death (`DOA`) as such:
# 
# - 0 Not Applicable 
# - 7 Died at Scene
# - 8 Died En Route (to a hospital)
# - 9 Unknown
# 
# Again, we'll ignore cases where death is unknown and focus on categories 0,7 and 8
# We barely have any "Died En Route" cases
# So we'll combine values 7 & 8 value to a single `Died in accident` value with a key of 7

# In[13]:


DIED_VALUE = 7
DIED_LABEL = 'Died in accident'


# In[14]:

# We'll create a new `accident_result` field that will be our prediction target and will be a combination of both `INJ_SEV` AND `DOA`
def add_accident_result_columns(df, accident_result_column, accident_result_name_column):
    """
    Creates a df of accident results for the inputted df

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the raw data
    accident_result_column (str): The name of the new accident result column
    accident_result_name_column (str): The name of the new accident result label column 
    
    Returns:
    result_df (pd.DataFrame): A df containing the actual accident result 
    per the the input df's rows
    """

    # Creating a copy df containing only the columns necessary for inferring the accident result
    result_df = df.filter(regex=r'^(INJ_SEV|DOA)', axis=1).copy()
    
    # Initialize accident_result_column and accident_result_name_column based on 'INJ_SEV' and 'INJ_SEVNAME'
    result_df[accident_result_column] = result_df['INJ_SEV']
    result_df[accident_result_name_column] = result_df['INJ_SEVNAME']
    
    # If the person died, update accident_result_column and accident_result_name_column as the corrsponding death type
    global DIED_VALUE 
    global DIED_LABEL
    mask_doa = result_df['DOA'] > 0
    result_df.loc[mask_doa, accident_result_column] = DIED_VALUE
    result_df.loc[mask_doa, accident_result_name_column] = DIED_LABEL
    
    # If the result was a fatal injury without death, 
    # we'll tag an accident result as a fatal injury only when the person did not die.
    mask_fatal = result_df[accident_result_column] == 4
    result_df.loc[mask_fatal, accident_result_name_column] = 'Fatal Injury without Death'
    
    # Clean accident_result_name_column labels
    result_df[accident_result_name_column] = result_df[accident_result_name_column].str.extract(r'^(?P<accident_result_name>[^()]+)')
    result_df[accident_result_name_column] = result_df[accident_result_name_column].str.strip()

    result_df = result_df[[accident_result_column, accident_result_name_column]]

    return result_df

def make_id_label_dict(df, id_column, label_column):
    """
    Creates a dictionary from a pandas DataFrame with IDs as keys and labels as values.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame containing ID and label columns.
    id_column (str): The name of the column containing the IDs.
    label_column (str): The name of the column containing the labels.
    
    Returns:
    id_label_dict (dict): A dictionary with IDs as keys and labels as values.
    """
    # Drop duplicates to ensure unique ID-label pairs
    unique_pairs = df[[id_column, label_column]].drop_duplicates()
    
    # Convert the unique pairs to a dictionary
    id_label_dict = dict(zip(unique_pairs[id_column], unique_pairs[label_column]))
    
    return id_label_dict

# In[23]:




# ### Running models
# For starters we'll try to create models that learn on the data as is.</br>
# This makes sense because not having specific information about a person in a motor accident could be valuable information by itself in predicting the person's injury

# The evaluation metric we'll use for our model will be a weighted average of recall per class.
# 
# The more severe an injury gets, the more important it is to decrease the amount of False Negative predictions of it, since the price of an error becomes more severe. 
# Therefore, it makes sense to calculate the recall score of each injury class separately and then calculate an overall weighted average that gives higher importance to more severe injuries.
# 
# For the sake of this exercise, we'll focus for now only on this metric as the only maximising metric and not take into account other satisfising factors like the predition speed of our model

# In[25]:





# ### Trying out some simple models

# #### Prep for modeling

# In[26]:





# In[28]:



val_df = pd.read_csv(VALIDATION_PATH, encoding='Windows-1252')
train_df, train_indices_df, train_target_df, train_label_for_target = prep_training_datasets(df, INDEX_COLUMNS)
val_df, val_indices_df, val_target_df, val_label_for_target = prep_training_datasets(val_df, INDEX_COLUMNS)


# In[34]:
assert train_label_for_target == val_label_for_target


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


# In[38]:


# Train and log each model using a pipeline
def train_and_log_pipelines(models, data_prep_steps, preprocessing_params, artifacts, X_train, X_val, y_train, y_val, is_validation_set_test = False):
    for model_name, (model, model_params_combinations) in models.items():
        for model_params_combination in model_params_combinations:
            pipeline_steps = data_prep_steps.copy()
            pipeline_steps.append((model_pipe_step_name, model))
            pipeline = Pipeline(pipeline_steps)

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


# In[ ]:


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


# In[39]:


train_df_with_idx = train_indices_df.join(train_df, how='inner')
val_df_with_idx = val_indices_df.join(val_df, how='inner')


# In[40]:


data_preparation_mapper.fit_transform(train_df_with_idx).iloc[4, :].to_dict()


# ## Productionizing the best model
# 
# The results for our models are honestly pretty bad. </br>
# But since the purpose of our project is productionizing our model, this will suffice for now 

# We want to get the best models according tot he validation set metrics </br>
# And train them from scratch on the train and validation set so that we can evaluate them on the validation set
# 
# For this purpose we created several helper functions

# In[41]:


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


# In[42]:


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


# ### Confusion matrix code for later exploring

# In[66]:


import mlflow.pyfunc

alias = "Champion"
model_path = f'models/{registered_model_name}@Champion'
champion_model = mlflow.pyfunc.load_model(model_path)


# In[85]:


import numpy as np
from sklearn import metrics

def plot_confusion_matrix(test_true_labels, test_predictions, textual_labels):

    translated_test_target_df = test_true_labels.map(textual_labels)

    # Create a vectorized function that applies the dictionary mapping
    vectorized_map = np.vectorize(textual_labels.get)

    # Translate the numbers in the array to their corresponding strings using the vectorized function
    translated_test_predictions = vectorized_map(test_predictions)

    # Define the order of labels according to their numerical values
    label_order = [textual_labels[key] for key in sorted(textual_labels.keys())]

    confusion_matrix = metrics.confusion_matrix(translated_test_target_df, translated_test_predictions, labels=label_order)


    plt.figure(figsize=(10, 7))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_order, 
                yticklabels=label_order)
    plt.xlabel('Predicted Labels')
    plt.ylabel('Actual Labels')
    plt.title('Confusion Matrix')
    plt.show()

test_predictions = champion_model.predict(test_df)
plot_confusion_matrix(test_target_df, test_predictions, test_label_for_target)

