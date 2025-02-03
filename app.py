import numpy as np 
import tensorflow as tf
from flask import Flask, request, jsonify

# Load the model with explicit loss function
model = tf.keras.models.load_model("traffic_forecast_model.h5", 
                                   custom_objects={"mse": tf.keras.losses.MeanSquaredError()})

# Initialize Flask app
app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return "Traffic Forecasting API is Running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        traffic_sequence = np.array(data["traffic_data"]).reshape(1, 10, 1)  # Reshape for LSTM
        prediction = model.predict(traffic_sequence)[0][0]
        return jsonify({"predicted_traffic_count": float(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
