import pickle
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('diabetes_model.sav', 'rb'))  # Replace 'diabetes_model.sav' with your model file

@app.route('/')
def index():
    return render_template('index.html')  # Assuming your HTML file is named 'index.html'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from the form
        Pregnancies = float(request.form['Pregnancies'])
        Glucose = float(request.form['Glucose'])
        BloodPressure = float(request.form['BloodPressure'])
        SkinThickness = float(request.form['SkinThickness'])
        Insulin = float(request.form['Insulin'])
        BMI = float(request.form['BMI'])
        DiabetesPedigreeFunction = float(request.form['DiabetesPedigreeFunction'])
        Age = float(request.form['Age'])
        
        # Create input data as a NumPy array
        input_data = np.array([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]).reshape(1, -1)

        # Make prediction using the loaded model
        prediction = model.predict(input_data)[0]

        # Prepare prediction result for display
        if prediction == 1:
            result = "The patient is predicted to have diabetes."
        else:
            result = "The patient is predicted to not have diabetes."

        return render_template('index.html', prediction_result=result)
        
    except ValueError:
        error_message = "Invalid input. Please enter numeric values for all fields."
        return render_template('index.html', error=error_message) 

if __name__ == '__main__':
    app.run(debug=True)