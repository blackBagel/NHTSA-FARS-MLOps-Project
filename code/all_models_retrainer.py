import os
import pandas as pd
import utils.data_preprocess as data_prep
from utils.training import get_train_val_test_dfs, train_and_log_pipelines
from utils import ready_pipelines

# def get_models_datasets_dir(datasets_dir_relative_path = 'data/datasets', models_datasets_dir = 'for_models'):
#     current_path = os.path.realpath(__file__)
#     parent_dir = os.path.dirname(os.path.dirname(current_path))
#     models_datasets_dir = os.path.join(parent_dir, datasets_dir_relative_path, models_datasets_dir)

#     return models_datasets_dir

def retrain_all_models():
    train_df, val_df, _ = get_train_val_test_dfs()

    train_df, _, train_target_df, _, train_artifacts = data_prep.prep_training_datasets(train_df)
    val_df, _, val_target_df, _, _ = data_prep.prep_training_datasets(val_df)

    data_prep_steps, preprocessing_params = data_prep.get_data_preprocessor_step()

    models = ready_pipelines.get_possible_models_for_pipeline()

    train_and_log_pipelines(models = models,
                            data_prep_steps = data_prep_steps,
                            preprocessing_params = preprocessing_params,
                            artifacts = train_artifacts,
                            X_train = train_df,
                            X_val = val_df,
                            y_train = train_target_df,
                            y_val = val_target_df,
                            is_validation_set_val = False)

