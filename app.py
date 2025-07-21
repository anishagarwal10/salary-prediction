from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the ML model
model = pickle.load(open('salary_predictor_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = float(request.form['age'])
        experience = float(request.form['experience'])

        prediction = model.predict([[age, experience]])
        output = round(prediction[0], 2)

        return render_template('index.html', prediction_text=f'Estimated Salary: ₹ {output}')
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
