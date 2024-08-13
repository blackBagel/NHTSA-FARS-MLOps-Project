# from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline

# In[25]:

def prefix_model_params(model_pipe_step_name, models):
    """
    Prefixes the model parameter keys with the given model pipeline step name.
    This prefixing is necessary for the params to be added to the sklearn.pipeline model step

    Args:
    model_pipe_step_name (str): The model pipeline step name to prefix the parameter keys with.
    models (dict): A dictionary containing model definitions and parameters.

    Returns:
    dict: A new dictionary with the prefixed parameter keys.
    """
    prefixed_models = {}
    
    for model_name, (model, params_list) in models.items():
        prefixed_params_list = []
        for params in params_list:
            prefixed_params = {f'{model_pipe_step_name}__{key}': value for key, value in params.items()}
            prefixed_params_list.append(prefixed_params)
        prefixed_models[model_name] = (model, prefixed_params_list)
    
    return prefixed_models

# Define the models and parameter grids to be tested
MODELS = {
    'RandomForest': (RandomForestClassifier(), [
        {'n_estimators': 100, 'max_depth': 10},
        {'n_estimators': 200, 'max_depth': 20},
    ]),
    # # 'SVC': (SVC(probability=True), [
    # #     {'C': 1, 'kernel': 'linear'},
    # #     {'C': 1, 'kernel': 'rbf'},
    # # ]),
    # # 'LogisticRegression': (LogisticRegression(), [
    # #     {'penalty': 'l2', 'C': 1},
    # #     {'penalty': 'l2', 'C': 0.1},
    # # ]),
    'GradientBoosting': (GradientBoostingClassifier(), [
        {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3},
        {'n_estimators': 200, 'learning_rate': 0.1, 'max_depth': 3},
    ]),
    # 'XGBClassifier': (XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'), [
    #     {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3},
    #     {'n_estimators': 200, 'learning_rate': 0.1, 'max_depth': 3},
    # ]),
}

def get_possible_models_for_pipeline(model_pipe_step_name = 'classifier'):
    models = prefix_model_params(model_pipe_step_name, MODELS)

    return models