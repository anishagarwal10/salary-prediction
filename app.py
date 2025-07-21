from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('salary_predictor_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            experience = float(request.form['experience'])
            prediction = model.predict(np.array([[experience]]))
            return render_template('index.html', prediction_text=f'Predicted Salary: ₹{prediction[0]:,.2f}')
        except Exception as e:
            return render_template('index.html', prediction_text=f'Error: {e}')
    return "Method Not Allowed", 405

if __name__ == '__main__':
    app.run(debug=True)
