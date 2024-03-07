import streamlit as st
import numpy as np
import pickle

# Load the trained SVM model
with open('Diabetes_model.pkl', 'rb') as model_file:
    classifier = pickle.load(model_file)

# Load the scaler
with open('Scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Call set_page_config() here
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon=":hospital:",
    layout="wide"
)

def main():
    
    st.title('Diabetes Prediction App')
    st.sidebar.header('Results ⏳')

    # Get user input from the form
    pregnancies = st.number_input('Number of Pregnancies 👶', min_value=0, value=0)
    glucose = st.number_input('Glucose Level 🍬', min_value=0, value=0)
    blood_pressure = st.number_input('Blood Pressure 💓', min_value=0, value=0)
    skin_thickness = st.number_input('Skin Thickness 🤚', min_value=0, value=0)
    insulin = st.number_input('Insulin Level 💉', min_value=0, value=0)
    bmi = st.number_input('BMI (Body Mass Index) 🧍‍♂️', min_value=0.0, value=0.0)
    diabetes_pedigree_function = st.number_input('Diabetes Pedigree Function 🧬', min_value=0.0, value=0.0)
    age = st.number_input('Age 🎂', min_value=0, value=0)

    if st.button('Predict Diabetes 📊'):
        # Standardize the input data using the loaded scaler
        input_data = np.array([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]).reshape(1, -1)
        standardized_input = scaler.transform(input_data)

        # Make prediction
        prediction = classifier.predict(standardized_input)

        # Display the prediction result in the sidebar
        st.sidebar.subheader('Prediction 🔮:')
        if prediction[0] == 0:
            result = 'Person is not diabetic.'
            st.sidebar.success(result)
            st.sidebar.write('Recommendation: Continue a healthy lifestyle. 🌱')
        else:
            result = 'Person is diabetic.'
            st.sidebar.error(result)
            st.sidebar.write('Recommendation: Consult with a healthcare professional. 🏥')

        # Show a popup message
        st.text(result)

if __name__ == '__main__':
    main()
