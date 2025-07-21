from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('salary_predictor_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(x) for x in request.form.values()]
        final_features = np.array(features).reshape(1, -1)
        prediction = model.predict(final_features)[0]
        return render_template('index.html', prediction_text=f'Predicted Salary: ₹{prediction:.2f}')
    except Exception as e:
        return render_template('index.html', prediction_text='Error in prediction: ' + str(e))

if __name__ == '__main__':
    app.run(debug=True)
