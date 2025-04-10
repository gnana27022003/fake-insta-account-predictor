from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
import os

app = Flask(__name__)

# Load the saved model and scaler
model_path = os.path.join(os.path.dirname(__file__), 'fake_account_rf_model.pkl')
scaler_path = os.path.join(os.path.dirname(__file__), 'scaler.pkl')

with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

with open(scaler_path, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)


@app.route('/')
def home():
    return render_template('form.ejs')  # Render the form


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from frontend
        data = request.json

        # Convert received data into DataFrame
        user_input = pd.DataFrame([[
            data["f1"], data["f2"], data["f3"], data["f4"], data["f5"],
            data["f6"], data["f7"], data["f8"], data["f9"], data["f10"], data["f11"]
        ]], columns=[
            "profile pic", "nums/length username", "fullname words", "nums/length fullname",
            "name==username", "description length", "external URL", "private", "#posts",
            "#followers", "#follows"
        ])
        
        # Scale the input data
        user_input = scaler.transform(user_input)

        # Make prediction
        prediction = model.predict(user_input)
        prediction_label = "Fake Account" if prediction[0] == 1 else "Not Fake"

        return jsonify({"prediction": prediction_label})

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    app.run(debug=True)
