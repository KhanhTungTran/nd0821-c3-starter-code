from starter.starter.ml.model import compute_model_metrics, inference
from starter.starter.ml.data import process_data
import pytest
import logging
import os
import pandas as pd
from joblib import load
from sklearn.exceptions import NotFittedError

@pytest.fixture(scope='session')
def path():
    return 'starter/data/census.csv'


@pytest.fixture(scope='session')
def model_dir():
    return 'starter/model'


@pytest.fixture(scope='session')
def data(path):
    return pd.read_csv(path, index_col=0)


@pytest.fixture(scope='session')
def cat_features():
    return [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]


def test_import_data(path):
    try:
        data = pd.read_csv(path, index_col=0)
    except Exception as e:
        logging.error(e)
        raise e

    assert data.shape[0] > 1
    assert data.shape[1] > 1
    assert 'salary' in data.columns


def test_process_data(data, cat_features):
    try:
        X, y, encoder, lb = process_data(
            data, categorical_features=cat_features, label="salary", training=True
        )
    except Exception as e:
        logging.error(e)
        raise e

    assert len(X) == len(y)


def test_load_model(data, cat_features, model_dir):
    X, y, encoder, lb = process_data(
        data, categorical_features=cat_features, label="salary", training=True,
    )

    try:
        model = load(os.path.join(model_dir, 'rf_model.joblib'))
    except Exception as e:
        logging.error(e)
        raise e


def test_model_is_fitted(data, cat_features, model_dir):
    X, y, encoder, lb = process_data(
        data, categorical_features=cat_features, label="salary", training=True,
    )

    model = load(os.path.join(model_dir, 'rf_model.joblib'))
    try:
        preds = inference(model, X)
    except NotFittedError as e:
        logging.error(e)
        raise e


def test_inference(data, cat_features, model_dir):
    try:
        encoder = load(os.path.join(model_dir, 'encoder.joblib'))
        lb = load(os.path.join(model_dir, 'lb.joblib'))
    except Exception as e:
        logging.error(e)
        raise e

    X, y, encoder, lb = process_data(
        data, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )

    model = load(os.path.join(model_dir, 'rf_model.joblib'))
    try:
        preds = inference(model, X)
    except Exception as e:
        logging.error(e)
        raise e


def test_compute_metrics(data, cat_features, model_dir):
    try:
        encoder = load(os.path.join(model_dir, 'encoder.joblib'))
        lb = load(os.path.join(model_dir, 'lb.joblib'))
    except Exception as e:
        logging.error(e)
        raise e

    X, y, encoder, lb = process_data(
        data, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )

    model = load(os.path.join(model_dir, 'rf_model.joblib'))
    preds = inference(model, X)
    try:
        precision, recall, fbeta = compute_model_metrics(y, preds)
    except Exception as e:
        logging.error(e)
        raise e
