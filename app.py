from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("salary_predictor_model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    age = float(request.form['age'])
    experience = float(request.form['experience'])

    prediction = model.predict(np.array([[age, experience]]))
    salary = f"₹{prediction[0]:,.2f}"

    return render_template('result.html', age=age, experience=experience, salary=salary)

if __name__ == '__main__':
    app.run(debug=True)
