import requests

# API URL
url = "http://127.0.0.1:5000/predict"

# Sample input data
data = {
    "traffic_data": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
}

# Send POST request
response = requests.post(url, json=data)

# Print response
print("Response:", response.json())
