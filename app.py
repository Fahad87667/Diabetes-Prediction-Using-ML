from flask import Flask, render_template, request
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the trained SVM model
with open('Diabetes_model.pkl', 'rb') as model_file:
    classifier = pickle.load(model_file)

# StandardScaler for input data
scaler = StandardScaler()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user input from the form
        input_features = [float(request.form[col]) for col in [
            'pregnancies', 'glucose', 'blood_pressure', 'skin_thickness',
            'insulin', 'bmi', 'diabetes_pedigree_function', 'age'
        ]]
        
        # Standardize the input data using the previously fitted scaler
        input_data = np.array(input_features).reshape(1, -1)
        standardized_input = scaler.fit_transform(input_data)  # Use transform instead of fit_transform

        # Make prediction
        prediction = classifier.predict(standardized_input)

        # Display the prediction result
        if prediction[0] == 0:
            result = 'The Person is not Diabetic'
        else:
            result = 'The Person is Diabetic'

        print(result)  # This will print the result to the console for debugging

        return render_template('index.html', prediction_text='Prediction: {}'.format(result))

if __name__ == "__main__":
    app.run(debug=True)
