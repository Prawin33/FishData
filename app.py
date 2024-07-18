from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the model and scaler
model = joblib.load('fish_weight_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array([[data['length1'], data['length2'], data['length3'], data['height'], data['width'], data['species']]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    return jsonify({'weight': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
