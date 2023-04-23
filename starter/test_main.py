from fastapi.testclient import TestClient
import logging, json
from main import app

client = TestClient(app)


def test_get():
    resp = client.get('/')

    assert resp.status_code == 200
    assert type(resp.json() == str)


def test_post_true():
    sample = {
        'age':40,
        'workclass':"Private",
        'fnlgt':193524,
        'education':"Doctorate",
        'education_num':16,
        'marital_status':"Married-civ-spouse",
        'occupation':"Prof-specialty",
        'relationship':"Husband",
        'race':"White",
        'sex':"Male",
        'capital_gain':0,
        'capital_loss':0,
        'hours_per_week':60,
        'native_country':"United-States"
    }

    sample = json.dumps(sample)

    resp = client.post('/infer', data=sample)
    assert resp.status_code == 200
    assert resp.json()['prediction'] == 1


def test_post_false():
    sample = {
        'age':54,
        'workclass':"Private",
        'fnlgt':302146,
        'education':"HS-grad",
        'education_num':9,
        'marital_status':"Separated",
        'occupation':"Other-service",
        'relationship':"Unmarried",
        'race':"Black",
        'sex':"Female",
        'capital_gain':0,
        'capital_loss':0,
        'hours_per_week':20,
        'native_country':"United-States"
    }

    sample = json.dumps(sample)

    resp = client.post('/infer', data=sample)
    assert resp.status_code == 200
    assert resp.json()['prediction'] == 0
