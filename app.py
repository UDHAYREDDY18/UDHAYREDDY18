from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS

# Initialize app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend to call API

# Load the saved model
model = joblib.load("antenna_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Extract values from incoming JSON
        input_values = np.array([
            [
                data['frequency'],
                data['dielectric'],
                data['height'],
                data['d'],
                data['s'],
                data['t']
            ]
        ])

        prediction = model.predict(input_values)[0]

        return jsonify({
            "g": float(prediction[0]),
            "w": float(prediction[1]),
            "l": float(prediction[2])
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
