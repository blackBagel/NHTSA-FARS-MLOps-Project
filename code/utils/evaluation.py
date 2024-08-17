import pandas as pd
from sklearn.metrics import recall_score
import mlflow.pyfunc
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

EVALUATION_METRIC = 'weighted_recall'

def weighted_recall_score(true_labels: pd.Series, predicted_labels: pd.Series) -> float:
    """
    Calculate the weighted average recall score of each possible value of the true labels,
    where the weight of each group is the value of the true label.

    Args:
    - true_labels (pd.Series): Series of true labels.
    - predicted_labels (pd.Series): Series of predicted labels.

    Returns:
    - float: Weighted average recall score.
    """
    # Ensure the input series have the same length
    assert len(true_labels) == len(predicted_labels), "True and predicted labels must have the same length."
    
    # Get unique values in true labels
    unique_labels = true_labels.unique()
    
    # Initialize variables to calculate weighted recall
    total_weight = 0
    weighted_recall_sum = 0
    
    # Calculate recall for each unique label and compute weighted sum
    for label in unique_labels:
        # Create boolean masks for the current label
        true_mask = (true_labels == label)
        evaluated_true_labels = true_labels[true_mask]
        evaluated_pred_labels = predicted_labels[true_mask]
        
        # Calculate recall for the current label
        recall = recall_score(evaluated_true_labels, evaluated_pred_labels, average="micro")
        
        # Calculate the weight for the current label
        weight = int(label)
        
        # Accumulate weighted recall and total weight
        weighted_recall_sum += recall * weight
        total_weight += weight
    
    # Calculate weighted average recall
    weighted_avg_recall = weighted_recall_sum / total_weight if total_weight != 0 else 0
    
    return weighted_avg_recall



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


# Example usage code

# alias = "Champion"
# model_path = f'models/{registered_model_name}@Champion'
# champion_model = mlflow.pyfunc.load_model(model_path)

# test_predictions = champion_model.predict(test_df)
# plot_confusion_matrix(test_target_df, test_predictions, test_label_for_target)