from update_train_val_test_data import update_model_datsets
from all_models_retrainer import retrain_all_models
from champion_trainer import train_champion
from prefect import serve

if __name__ == "__main__":
    model_datsets_update_deployment = update_model_datsets.to_deployment(
        name="model_datasets_updater",
        cron="0 1 * * 1", # At 01:00 on Mondays
        tags=["data", "scheduled"],
        description="Updates the datasets models will be trained on",
    )
    
    all_models_train_deployment = retrain_all_models.to_deployment(
        name="candidate_models_train_deployment",
        cron="0 3 * * 1", # At 03:00 AM on Mondays
        tags=["models", "scheduled"],
        description="Responsible for training all the possible candidate models on the training data",
    )

    champion_train_deployment = train_champion.to_deployment(
        name="champion_model_retrain_deployment",
        cron="0 4 * * 1", # At 03:00 AM on Mondays
        tags=["models", "scheduled"],
        description="Responsible for retraining the champion model on updated data",
    )

    serve(model_datsets_update_deployment, all_models_train_deployment, champion_train_deployment)