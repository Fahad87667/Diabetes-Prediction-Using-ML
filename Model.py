import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
dataset = pd.read_csv('diabetes.csv')

# Splitting Dataset for training and testing
x = dataset.drop(columns=['Outcome'], axis=1)
y = dataset['Outcome']

# Data standardization
scaler = StandardScaler()
x = scaler.fit_transform(x)

# Train Test Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

# Model Training
classifier = svm.SVC(kernel='linear', C=1)  # You can experiment with different kernels and C values
classifier.fit(x_train, y_train)

# Save the trained model and scaler to pickle files
with open('Diabetes_model.pkl', 'wb') as model_file:
    pickle.dump(classifier, model_file)

with open('Scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

# Model Evaluation
# Accuracy score on training data
x_train_prediction = classifier.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction, y_train)

# Accuracy score on test data
x_test_prediction = classifier.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction, y_test)

print("Training Data Accuracy:", training_data_accuracy)
print("Test Data Accuracy:", test_data_accuracy)
