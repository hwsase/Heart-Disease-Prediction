
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# Load the UCI Heart Disease dataset (or your dataset)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
col_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
heart_df = pd.read_csv(url, names=col_names)

# Replace missing values (?) with NaN
heart_df.replace('?', np.nan, inplace=True)

# Drop rows with missing values (or handle them appropriately)
heart_df.dropna(inplace=True)

# Convert all columns to numeric
heart_df = heart_df.apply(pd.to_numeric)

# Show the first few rows of the dataset
print(heart_df.head())

# Separate features and target
X = heart_df.drop(columns='target', axis=1)
Y = heart_df['target']

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Define the base models
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, criterion="entropy")
nn_model = MLPClassifier(hidden_layer_sizes=(50, 50), max_iter=1000, random_state=42)

# Define the stacking model
stacking_model = StackingClassifier(
    estimators=[('rf', rf_model), ('nn', nn_model)],
    final_estimator=RandomForestClassifier(n_estimators=100, random_state=42)
)

# Fit the stacking model on the training set
stacking_model.fit(X_train, y_train)

# Accuracy on training data
X_train_prediction = stacking_model.predict(X_train)
training_data_accuracy = accuracy_score(y_train, X_train_prediction)
print('Accuracy on Training data:', training_data_accuracy)

# Accuracy on test data
stacking_pred = stacking_model.predict(X_test)
stacking_acc = accuracy_score(y_test, stacking_pred)
print("Stacking accuracy on Test data:", stacking_acc)

# Calculate the error rate on the test data
error_rate = 1 - stacking_acc
print('Ensemble model error rate on test data:', error_rate)

# Test the model with new data
age = int(input("Enter age: "))
sex = int(input("Enter 1 for Male, 0 for Female: "))
cp = int(input("Enter chest pain type (1-4): "))
trestbps = int(input("Enter resting blood pressure (mm/Hg): "))
chol = int(input("Enter serum cholesterol (mg/dl): "))
fbs = int(input("Enter fasting blood sugar (>120 mg/dl: 1, else: 0): "))
restecg = int(input("Enter resting ECG result (0-2): "))
thalach = int(input("Enter maximum heart rate achieved: "))
exang = int(input("Exercise-induced angina (1: Yes, 0: No): "))
oldpeak = float(input("Enter ST depression induced by exercise relative to rest: "))
slope = int(input("Enter slope of peak exercise ST segment (1-3): "))
ca = int(input("Number of major vessels (0-3): "))
thal = int(input("Thalassemia (1: normal, 2: fixed defect, 3: reversible defect): "))

# Create input data array
input_data = (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)

# Change the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape the numpy array as we are predicting for only one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Make prediction
prediction = stacking_model.predict(input_data_reshaped)

# Output prediction
if prediction[0] == 0:
    print('The Person does not have Heart Disease')
else:
    print('The Person has Heart Disease')
    
