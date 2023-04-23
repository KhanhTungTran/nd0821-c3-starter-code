# Put the code for your API here.
from fastapi import FastAPI
from pydantic import BaseModel
from starter.ml.data import process_data
from starter.ml.model import inference
from joblib import load
import pandas as pd


def to_underscore(orig: str) -> str:
    return orig.replace('-', '_')


class SingleSample(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

    class Config:
        alias_generator = to_underscore
        scheme_extra = {
            "example":  {
                'age': 54,
                'workclass': "Private",
                'fnlgt': 302146,
                'education': "HS-grad",
                'education_num': 9,
                'marital_status': "Separated",
                'occupation': "Other-service",
                'relationship': "Unmarried",
                'race': "Black",
                'sex': "Female",
                'capital_gain': 0,
                'capital_loss': 0,
                'hours_per_week': 20,
                'native_country': "United-States"
            }
        }


app = FastAPI()


# @app.on_event(event_type='startup')
# async def load_model() -> None:
#     global model, encoder, lb, cat_features
model = load("model/rf_model.joblib")
encoder = load("model/encoder.joblib")
lb = load("model/lb.joblib")
cat_features = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]


@app.get('/')
async def welcome() -> str:
    return "Welcome to APIs for inference model trained on census data"


@app.post('/infer')
async def infer(input: SingleSample):
    input_dct = input.dict(by_alias=True)
    input_df = pd.DataFrame([input_dct])

    input_data, _, _, _ = process_data(
        input_df, categorical_features=cat_features, 
        training=False, encoder=encoder, lb=lb
    )

    result = inference(model, input_data)

    return {
        'prediction': int(result[0])
    }
