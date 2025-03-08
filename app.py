from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# Global variable to store the model
model = None

def load_model():
    """Load the model once and store it in the global variable."""
    global model
    if model is None:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    load_model()  # Ensure the model is loaded before making predictions

    try:
        # Extract and validate inputs
        inputs = [request.form.get(key, '').strip() for key in 
                  ['no_of_sites', 'inc_mat', 'freq_breakdown', 'inacc_cost', 'satisfaction']]
        if any(i == '' for i in inputs):  # Check if any input is missing
            return render_template('index.html', prediction_text="Please fill all fields.")

        # Convert inputs to float
        input_features = np.array([[float(i) for i in inputs]])

        # Make prediction
        prediction = model.predict(input_features)[0]

        return render_template('index.html', prediction_text=f'Predicted Delay Rate: {prediction:.2f}')

    except ValueError:
        return render_template('index.html', prediction_text="Invalid input. Enter valid numbers.")

if __name__ == '__main__':
    load_model()  # Load model once at startup
    port = int(os.environ.get("PORT", 5000))  # Use PORT from Render
    app.run(host="0.0.0.0", port=port, debug=True)

