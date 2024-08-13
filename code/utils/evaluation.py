import pandas as pd
from sklearn.metrics import recall_score

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
