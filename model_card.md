# Model Card

Model card for a Random Forest model trained on census.csv dataset

## Model Details
This `Random Forest Ensemble` model  is trained on the publicly available `Census` data to classify users into two groups based on their `salary`. The threshold salary used is $50,000. The data is publicly available on the Internet.

The model is implemented and trained using the `Scikit-Learn` package with default parameters.

## Intended Use
The model is implemented and trained to predict the salary of a user based on their information. However, it is intended to use only in the scope of Udacity course on MLOps.

## Training Data
The `Census` dataset is used in this work. Input features to the model are:
- age
- capital-gain
- capital-loss
- education-num
- education
- fnlwgt
- hours-per-week
- marital-status
- native-country
- occupation
- race
- relationship
- sex
- workclass

There are 32,561 samples in the dataset.

## Evaluation Data
20% is used as evaluation dataset. We use a stratify strategy on the target column `salary` to split the training and test set.

## Metrics
Performance of the model on test set:
- precision: 0.7373
- recall: 0.6122
- fbeta: 0.6690

Additionally, we evaluate model's performance on slice of test data, please see [here](starter/slice_output.txt) for more information.
## Ethical Considerations
Please see [here](starter/slice_output.txt) for more information on how the model performed with regards to different group of users based on specific column's values.

## Caveats and Recommendations
We hold no responsibility when this model is used, either for testing or in production environment.
