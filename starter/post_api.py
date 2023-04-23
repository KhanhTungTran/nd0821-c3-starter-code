import requests
import json

# Set the API endpoint URL
url = "https://udacity-web-service.onrender.com/infer"

# Set the headers
headers = {
    "Content-Type": "application/json"
}

sample = {
    'age': 40,
    'workclass': "Private",
    'fnlgt': 193524,
    'education': "Doctorate",
    'education_num': 16,
    'marital_status': "Married-civ-spouse",
    'occupation': "Prof-specialty",
    'relationship': "Husband",
    'race': "White",
    'sex': "Male",
    'capital_gain': 0,
    'capital_loss': 0,
    'hours_per_week': 60,
    'native_country': "United-States"
}

sample = json.dumps(sample)

# Send the POST request to the API endpoint
response = requests.post(url, data=sample, headers=headers)

# Print the response status code
print("Response status code:", response.status_code)

# Print the response content
print("Response content:", response.json())
