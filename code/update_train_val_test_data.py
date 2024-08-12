import os
import pandas as pd
from datetime import datetime, timedelta

DATA_FILE_NAME = 'person.csv'
YEAR_DIRS = ['2021', '2022']
DATASETS_DIR = '../data/datasets'
MODEL_DATASETS_DIR = os.path.join(DATASETS_DIR, 'for_model')

def load_and_process_csv(year, day, month):
    date_str = f"{year}-{month:02d}-{day:02d}"
    return datetime.strptime(date_str, "%Y-%m-%d")

def filter_data(df, start_date, end_date):
    df_copy = df.copy()
    df_copy.loc[:, 'DATE'] = df_copy.apply(lambda row: load_and_process_csv(row['YEAR'], row['DAY'], row['MONTH']), axis=1)
    return df_copy.loc[(df_copy['DATE'] >= start_date) & (df_copy['DATE'] < end_date), :].copy()

def process_files(year_dirs):
    dfs = []
    file = DATA_FILE_NAME
    for year_dir in year_dirs:
        file_path = os.path.join(DATASETS_DIR, year_dir, file)

        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['YEAR'] = year_dir  # Add the year directory as a column to infer the year
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def save_data(df, start_date, end_date, filename):
    filtered_data = filter_data(df, start_date, end_date)
    filtered_data.drop(columns=['DATE'], inplace=True)  # Drop auxiliary columns
    filtered_data.to_csv(filename, index=False)

def main(): 
    # Load and combine data from all directories
    combined_data = process_files(YEAR_DIRS)
    
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
    
    # Save to files

    train_path = os.path.join(MODEL_DATASETS_DIR, 'train.csv')
    validation_path = os.path.join(MODEL_DATASETS_DIR, 'validation.csv')
    test_path = os.path.join(MODEL_DATASETS_DIR, 'test.csv')
    save_data(combined_data, train_start, train_end, train_path)
    save_data(combined_data, validation_start, validation_end, validation_path)
    save_data(combined_data, test_start, test_end, test_path)

if __name__ == "__main__":
    main()
