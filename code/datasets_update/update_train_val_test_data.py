import os
import pandas as pd
from datetime import datetime, timedelta
from prefect import flow, task, get_run_logger
from prefect.tasks import task_input_hash

DATA_FILE_NAME = os.getenv('DATA_FILE_NAME')
DATASETS_DIR_RELATIVE_PATH = os.getenv('DATASETS_DIR_RELATIVE_PATH')

def get_datasets_dir():
    current_path = os.path.realpath(__file__)
    parent_dir = os.path.dirname(os.path.dirname(current_path))
    datasets_dir = os.path.join(parent_dir, DATASETS_DIR_RELATIVE_PATH)

    return datasets_dir

def load_and_process_csv(year, day, month):
    date_str = f"{year}-{month:02d}-{day:02d}"
    return datetime.strptime(date_str, "%Y-%m-%d")

def filter_data(df, start_date, end_date):
    df_copy = df.copy()
    df_copy.loc[:, 'DATE'] = df_copy.apply(lambda row: load_and_process_csv(row['YEAR'], row['DAY'], row['MONTH']), axis=1)
    return df_copy.loc[(df_copy['DATE'] >= start_date) & (df_copy['DATE'] < end_date), :].copy()

def process_files(datasets_dir, year_dirs):
    dfs = []
    file = DATA_FILE_NAME

    for year_dir in year_dirs:
        file_path = os.path.join(datasets_dir, str(year_dir), file)

        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['YEAR'] = year_dir  # Add the year directory as a column to infer the year
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

@task(cache_key_fn=task_input_hash, 
      cache_expiration=timedelta(hours=1),
      )
def save_data(df, start_date, end_date, filename, logger, dataset_name):
    filtered_data = filter_data(df.copy(), start_date, end_date)
    filtered_data = filtered_data.drop(columns=['DATE'])  # Drop auxiliary columns
    filtered_data.to_csv(filename, index=False)

    logger.info(f"Saved the {dataset_name} dataset file")


@flow(retries=3, retry_delay_seconds=60)
def update_model_datsets(): 
    # Date ranges
    today = datetime.today()
    train_start_weeks_delta = 156 # 3 years + 2 weeks ago
    train_end_weeks_delta = 104 + 2 # 2 years + 2 weeks ago

    train_start = today - timedelta(weeks=train_start_weeks_delta) 
    train_end = today - timedelta(weeks=train_end_weeks_delta)  
    
    validation_end_weeks_delta = 104 + 1 # 2 years + 1 weeks ago
    validation_start = train_end
    validation_end = today - timedelta(weeks=validation_end_weeks_delta)
    
    test_end_weeks_delta = 104 # 2 years
    test_start = validation_end
    test_end = today - timedelta(weeks=test_end_weeks_delta)
    
    # Load and combine data from all directories
    datasets_dir = get_datasets_dir()
    year_dirs = set([train_start.year, train_end.year, validation_start.year, validation_end.year, test_start.year, test_end.year])
    combined_data = process_files(datasets_dir=datasets_dir, year_dirs=year_dirs)
    
    # Save to files
    model_datasets_dir = os.path.join(datasets_dir, 'for_models')

    train_path = os.path.join(model_datasets_dir, 'train.csv')
    validation_path = os.path.join(model_datasets_dir, 'validation.csv')
    test_path = os.path.join(model_datasets_dir, 'test.csv')
    logger = get_run_logger()

    save_data.submit(combined_data, train_start, train_end, train_path, logger, 'train')
    save_data.submit(combined_data, validation_start, validation_end, validation_path, logger, 'validation')
    save_data.submit(combined_data, test_start, test_end, test_path, logger, 'test')

if __name__ == "__main__":
    update_model_datsets.serve(
        name="model_datasets_updater",
        cron="0 1 * * 1", # At 01:00 on Mondays
        tags=["data", "scheduled"],
        description="Updates the datasets models will be trained on",
    )
