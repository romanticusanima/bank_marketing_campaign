import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, Any, List
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report

def split_data(df: pd.DataFrame, target_col: str) -> Dict[str, pd.DataFrame]:
    """
    Split the dataframe into training and validation sets.

    Args:
        df (pd.DataFrame): The raw dataframe.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary containing the train and validation dataframes.
    """
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[target_col])
    return {'train': train_df, 'val': val_df}


def create_inputs_targets(df_dict: Dict[str, pd.DataFrame], input_cols: list, target_col: str) -> Dict[str, Any]:
    """
    Create inputs and targets for training and validation sets.

    Args:
        df_dict (Dict[str, pd.DataFrame]): Dictionary containing the train and validation dataframes.
        input_cols (list): List of input columns.
        target_col (str): Target column.

    Returns:
        Dict[str, Any]: Dictionary containing inputs and targets for train and val sets.
    """
    data = {}
    for split in df_dict:
        data[f'{split}_inputs'] = df_dict[split][input_cols].copy()
        data[f'{split}_targets'] = df_dict[split][target_col].map({'yes': 1, 'no': 0}).copy()
    return data


def evaluate_model(model, train_inputs, train_targets, val_inputs, val_targets, print_results = False):
    train_preds = model.predict(train_inputs)
    train_pred_proba = model.predict_proba(train_inputs)[:, 1]

    train_fpr, train_tpr, train_thresholds = roc_curve(train_targets, train_pred_proba)
    train_auroc = auc(train_fpr, train_tpr)

    val_preds = model.predict(val_inputs)
    val_pred_proba = model.predict_proba(val_inputs)[:, 1]

    val_fpr, val_tpr, val_thresholds = roc_curve(val_targets, val_pred_proba)
    val_auroc = auc(val_fpr, val_tpr)

    if print_results:
        print(f"AUROC for train:      {train_auroc:.2f}")
        print(f"AUROC for validation: {val_auroc:.2f}")

    return {
      'Train AUROC': f'{train_auroc:.2f}',
      'Validation AUROC': f'{val_auroc:.2f}',
    }


def evaluate_predictions(model_pipeline, inputs, targets, name=''):
    preds = model_pipeline.predict(inputs)
    y_pred_proba = model_pipeline.predict_proba(inputs)[:, 1]

    fpr, tpr, thresholds = roc_curve(targets, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    print(f'Area under ROC score on {name} dataset: {roc_auc:.2f}')

    print('')
    confusion_matrix_ = confusion_matrix(targets, preds, normalize='true')
    plt.figure()
    sns.heatmap(confusion_matrix_, annot=True, cmap='Blues', yticklabels=['no', 'yes'])
    plt.xlabel('Prediction')
    plt.ylabel('Target')
    plt.title('{} Confusion Matrix'.format(name))
    plt.show()

    print('')
    print(classification_report(targets, preds, digits=3))