import tensorflow as tf
from flask import Flask, request, jsonify
import numpy as np

# Explicitly define 'mse' loss function
custom_objects = {"mse": tf.keras.losses.MeanSquaredError()}

# Load the model with custom_objects
model = tf.keras.models.load_model("traffic_forecast_model.h5", custom_objects=custom_objects)

# Initialize Flask app
app = Flask(__name__)

@app.route("/", methods=["GET"])  # Home route
def home():
    return "Smart City Traffic Forecasting API is Running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data or "traffic_data" not in data:
            return jsonify({"error": "Invalid input"}), 400

        # Convert input data for LSTM model
        traffic_sequence = np.array(data["traffic_data"]).reshape(1, 10, 1)

        # Make prediction
        prediction = model.predict(traffic_sequence)[0][0]

        return jsonify({"predicted_traffic_count": float(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
