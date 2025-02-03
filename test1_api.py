import requests

url = "https://smart-city-traffic-analysis.onrender.com"

data = {
    "traffic_data": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
}

response = requests.post(url, json=data)

print("Status Code:", response.status_code)
print("Response:", response.json())  # Expected output: {"predicted_traffic_count": 0.63}
