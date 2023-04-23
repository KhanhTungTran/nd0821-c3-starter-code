# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pandas as pd
from ml.model import train_model, compute_model_metrics, inference, slice_inference
from ml.data import process_data
import logging
from joblib import dump

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# Add code to load in the data.
data = pd.read_csv('starter/data/census.csv', index_col=0)

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20, stratify=data['salary'])

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)

# Train and save a model.
model = train_model(X_train, y_train)

train_preds = inference(model, X_train)
precision, recall, fbeta = compute_model_metrics(y_train, train_preds)
logger.info(f"Performance on training set: precision {precision:.4f}, recall {recall:.4f}, fbeta {fbeta:.4f}")

test_preds = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, test_preds)
logger.info(f"Performance on test set: precision {precision:.4f}, recall {recall:.4f}, fbeta {fbeta:.4f}")

logger.info("The performance of the model on slices of the data")

result_df = []

for c in cat_features:
    temp_df = slice_inference(model, test, c, y_test, test_preds)
    logger.info(f"Slice performance of model on test set on feature {c}")
    logger.info(temp_df)
    result_df.append(temp_df)

result_df = pd.concat(result_df)
result_df.to_csv("starter/model/slice_result.csv")


logger.info("Saving model")
dump(model, 'starter/model/rf_model.joblib')
dump(encoder, 'starter/model/encoder.joblib')
dump(lb, 'starter/model/lb.joblib')
