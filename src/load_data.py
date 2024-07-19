import pandas as pd
import numpy as np
from typing import Tuple

def load_split_csv_data(features_path: str, labels_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load split data from separate CSV files for features and labels.
    
    Parameters:
    features_path (str): The path to the CSV file containing the features.
    labels_path (str): The path to the CSV file containing the labels.
    
    Returns:
    tuple: Tuple containing:
        - features (np.ndarray): The input features.
        - labels (np.ndarray): The target labels.
    """
    try:
        features = pd.read_csv(features_path).values
        labels = pd.read_csv(labels_path).values
        return features, labels
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return np.array([]), np.array([])
    except pd.errors.EmptyDataError as e:
        print(f"Error: {e}")
        return np.array([]), np.array([])
    except pd.errors.ParserError as e:
        print(f"Error: {e}")
        return np.array([]), np.array([])
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return np.array([]), np.array([])

# Example usage:
# train_features, train_labels = load_split_csv_data('path/to/preprocessed_X_train.csv', 'path/to/preprocessed_y_train.csv')
# val_features, val_labels = load_split_csv_data('path/to/preprocessed_X_val.csv', 'path/to/preprocessed_y_val.csv')
# test_features, test_labels = load_split_csv_data('path/to/preprocessed_X_test.csv', 'path/to/preprocessed_y_test.csv')
