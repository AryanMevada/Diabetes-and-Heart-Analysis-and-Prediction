from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load models
diabetes_model = pickle.load(open(os.path.join('models', 'diabetes_model.sav'), 'rb'))
heart_model = pickle.load(open(os.path.join('models', 'heart_model.sav'), 'rb'))
parkinsons_model = pickle.load(open(os.path.join('models', 'parkinsons_model.sav'), 'rb'))

@app.route('/diabetes', methods=['GET', 'POST'])
def diabetes():
    result = ''
    if request.method == 'POST':
        try:
            inputs = [float(request.form[f'input{i}']) for i in range(1, 9)]
            prediction = diabetes_model.predict([inputs])
            result = 'Diabetic' if prediction[0] == 1 else 'Not Diabetic'
        except Exception as e:
            result = f"Error: {e}"
    return render_template('DiabetesPrediction.html', result=result)

@app.route('/heart', methods=['GET', 'POST'])
def heart():
    result = ''
    if request.method == 'POST':
        try:
            inputs = [float(request.form[f'input{i}']) for i in range(1, 14)]
            prediction = heart_model.predict([inputs])
            result = 'Heart Disease Detected' if prediction[0] == 1 else 'No Heart Disease'
        except Exception as e:
            result = f"Error: {e}"
    return render_template('HeartDisease.html', result=result)

@app.route('/parkinsons', methods=['GET', 'POST'])
def parkinsons():
    result = ''
    if request.method == 'POST':
        try:
            inputs = [float(request.form[f'input{i}']) for i in range(1, 23)]
            prediction = parkinsons_model.predict([inputs])
            result = "Parkinson's Detected" if prediction[0] == 1 else "No Parkinson's"
        except Exception as e:
            result = f"Error: {e}"
    return render_template("Parkinson's Disease.html", result=result)

# Optional home redirect
@app.route('/')
def home():
    return render_template('DiabetesPrediction.html')

if __name__ == '__main__':
    app.run(debug=True)
