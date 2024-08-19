import os
from sklearn_pandas import DataFrameMapper, gen_features
from sklearn.base import TransformerMixin

MODEL_LABELS_DICT_FILE=os.getenv('MODEL_LABELS_DICT_FILE')

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
    'YEAR',
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
    'PER_NO',
    'YEAR'
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

# We'll create a new `accident_result` field that will be our prediction target and will be a combination of both `INJ_SEV` AND `DOA`
def add_accident_result_columns(df, accident_result_column, accident_result_name_column, death_value = 7, death_label = 'Dies in accident'):
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
    mask_doa = result_df['DOA'] > 0
    result_df.loc[mask_doa, accident_result_column] = death_value
    result_df.loc[mask_doa, accident_result_name_column] = death_label
    
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

def prep_training_datasets(df, index_columns = INDEX_COLUMNS):
    """
    Creates a df containing only rows our model can train on,
    a corresponding target Series for supervised learning,
    And a label dictionary for inferring the target series values

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the raw data
    including the columns necessary for creating the target column
    index_columns (list): a list of the columns creating together 
    a unique row identifier

    Returns:
    train_df (pd.DataFrame): A df containing only rows our model can train on
    indices_df (pd.DataFrame): A df containing the columns that make up a unique row identifier
    target_df (pd.DataFrame): A pandas Series containing the target the model will learn on
    accident_result_names (dict): A dictionary containing target series textual labels
    artifacts (list[dict[str, Any]]): A list of artifacts to be saved with the model in mlflow

    """
    # As mentioned above, we'll train our model only on cases where 
    # the outcome of the person was known
    training_mask = (df['INJ_SEV'] <= 4) & (df['DOA'] != 9)
    
    # Create a copy of the DataFrame to avoid modifying the original
    trainable_df = df[training_mask].copy()

    # Creating a df containing the index columns on which we shouldn't train
    indices_df = trainable_df[index_columns]
    trainable_df = trainable_df.drop(indices_df, axis=1)
    
    # Names of the target column and its' corresponding textual label
    accident_result_column = 'accident_result'
    accident_result_name_column = accident_result_column + '_name'

    # Creating a df containing only the target column and its' label
    result_df = add_accident_result_columns(trainable_df, accident_result_column, accident_result_name_column)

    # We'll keep a dict mapping the target ordinal value to its' label
    accident_result_names = make_id_label_dict(result_df, accident_result_column, accident_result_name_column)

    # Since we have the label dict there's no need to return the label column
    target_df = result_df[accident_result_column]

    artifacts = [
        { 
            'type': 'dict',
            'object': accident_result_names,
            'file_name': MODEL_LABELS_DICT_FILE,
        },
        # { 
        #     'type': 'file',
        #     'local_path': label_for_target_file,
        #     'artifact_path': 'label_for_target',
        # },
    ]

    return trainable_df, indices_df, target_df, accident_result_names, artifacts


# In[23]:
class AccidentPreModelVeFormsProcessor(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Keep only the specified columns
        X_transformed = X.copy()

        # Here we're actually thinking of the number of cars
        # as more of an ordinal variable rather than a numerical one
        # Meaning accidents with 7 or more cars is a "category 7" accident
        X_transformed = X_transformed.apply(lambda value: value if value < 7 else 7)
        
        return X_transformed

def get_data_preprocessor_step():
    processed_columns = ['VE_FORMS']
    columns_without_processing = [col for col in COLUMNS_FOR_MODEL if col not in processed_columns]

    columns_without_processing_definition = gen_features(
        columns = columns_without_processing,
        classes = [None],
    )

    columns_preprocessing_definition = gen_features(
        columns = ['VE_FORMS'],
        classes= [AccidentPreModelVeFormsProcessor],
    )

    dataframe_preprocessing_definition = columns_preprocessing_definition + columns_without_processing_definition

    data_preparation_mapper = DataFrameMapper(
        dataframe_preprocessing_definition,
        input_df=True,
        df_out=True,
    )

    data_prep_steps = [('general_preprocessor', data_preparation_mapper)]

    preprocessing_params = {
        'VE_FORMS_outlier_strategy': '> 6 becomes category 7',
        'train_columns' : COLUMNS_FOR_MODEL
    }

    return data_prep_steps, preprocessing_params