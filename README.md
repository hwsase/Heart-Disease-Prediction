Heart Disease Prediction using Ensemble Learning
This project predicts the likelihood of heart disease using an ensemble machine learning model. The dataset used is the UCI Heart Disease dataset, and the model leverages a stacking classifier that combines a Random Forest Classifier and a Neural Network.

Table of Contents
Overview
Installation
Usage
Modeling Approach
Input Features
Results
Acknowledgments
Overview
Heart disease is a leading cause of death globally. Early detection of heart disease can help reduce the mortality rate by enabling timely medical intervention. This project builds a machine learning-based prediction model using an ensemble approach, combining Random Forest and Neural Network classifiers to predict whether a person has heart disease based on various medical features.

Installation
To run this project locally, you will need:

Python 3.x installed

The following Python packages:

bash
Copy code
pip install pandas numpy scikit-learn matplotlib
An internet connection to fetch the UCI Heart Disease dataset directly from the UCI repository.

Usage
Clone or download this repository to your local machine.

Run the Python script heart_disease_prediction.py (or the name of your file) in your preferred IDE or terminal.

Follow the prompts to input patient data for a heart disease prediction.

Example:

bash
Copy code
python heart_disease_prediction.py
You will be prompted to enter the following medical details, such as age, gender, cholesterol level, etc., to predict if the person is likely to have heart disease.

Modeling Approach
Data Preprocessing:

The UCI Heart Disease dataset is loaded from the UCI repository.
Missing values are handled by replacing them with NaN and dropping rows that contain these values.
The dataset contains 13 features that are numeric and used to train the model.
Modeling:

Two base classifiers are used: a Random Forest Classifier and a Neural Network (MLP).
A stacking ensemble is built using these classifiers, with the Random Forest model acting as the meta-classifier.
The dataset is split into training (80%) and test (20%) sets using train_test_split.
Prediction:

The model predicts if a person has heart disease based on medical features.
New patient data can be entered manually to make predictions.
Input Features
The prediction model uses the following 13 features to predict the likelihood of heart disease:

Age: Age in years
Sex: 1 = Male, 0 = Female
Chest Pain Type (cp):
1 = Typical angina
2 = Atypical angina
3 = Non-anginal pain
4 = Asymptomatic
Resting Blood Pressure (trestbps): Resting blood pressure (in mm Hg)
Cholesterol (chol): Serum cholesterol in mg/dl
Fasting Blood Sugar (fbs): 1 = Fasting blood sugar > 120 mg/dl, 0 otherwise
Resting Electrocardiographic Results (restecg):
0 = Normal
1 = Having ST-T wave abnormality
2 = Showing probable or definite left ventricular hypertrophy
Maximum Heart Rate Achieved (thalach)
Exercise Induced Angina (exang): 1 = Yes, 0 = No
ST Depression (oldpeak): ST depression induced by exercise relative to rest
Slope of Peak Exercise ST Segment (slope):
1 = Upsloping
2 = Flat
3 = Downsloping
Number of Major Vessels (ca): Ranging from 0 to 3
Thalassemia (thal):
1 = Normal
2 = Fixed defect
3 = Reversible defect
Results
Training Accuracy: The accuracy of the model on the training set is printed.
Test Accuracy: The accuracy on the test set is reported to evaluate model performance.
Error Rate: The error rate on the test data is also calculated and displayed.
Once the model is trained, you can test it by inputting new data. The model will output whether the individual is likely to have heart disease or not.

Acknowledgments
This project uses the UCI Heart Disease Dataset and the machine learning libraries scikit-learn, pandas, and numpy.
