from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : sklearn.model
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    preds[preds>=0.5] = 1.0
    preds[preds<0.5] = 0.

    return preds


def slice_inference(model, test_df: pd.DataFrame, c, y, preds):
    """
    Infer model performance on slice of data, split by column c
    Inputs
    ------
    model : sklearn.model
        Trained machine learning model.
    test_df : pd.DataFrame
        Test set dataframe.
    c: str
        column to slice the data
    y: np.array
        Ground truth
    preds: np.array
        Model predictions.
    Returns
    -------
    df : pd.DataFrame
        Model performance on each slice. Columns:
            - feature_value
            - feature_name
            - precision
            - recall
            - fbeta
    """

    feature_values = test_df[c].unique().tolist()
    ret_df = pd.DataFrame(index=feature_values, columns=['precision', 'recall', 'fbeta'])

    for feature_value in feature_values:
        idxs = np.where(test_df[c] == feature_value)[0]
        precision, recall, fbeta = compute_model_metrics(y[idxs], preds[idxs])

        ret_df.at[feature_value, 'precision'] = precision
        ret_df.at[feature_value, 'recall'] = recall
        ret_df.at[feature_value, 'fbeta'] = fbeta

    f_values = ret_df.index
    ret_df.reset_index()
    ret_df.insert(0, column='feature_value', value=f_values)
    ret_df.insert(0, column='feature_name', value=c)

    return ret_df
