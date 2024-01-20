import pandas as pd
import joblib
from flask import Flask, render_template, request

depressed_model = joblib.load('depressed_model.pkl')
anxiety_model = joblib.load('anxiety_model.pkl')

def map_categorical_values(value):
    return 1 if value == 'Yes' else 0

def predict_depression(name, age, gender, year, work_load, edu_level, fear_of_failure, alcohol_or_drugs, social_media_usage, alone, study_hours):
    # Convert gender to numerical value
    gender_numeric = 0 if gender == 'Male' else 1

    features = pd.DataFrame({
        'Name': [name],
        'Gender': [gender_numeric],
        'Year': [year],
        'Work Load': [work_load],
        'Edu Level': [edu_level],
        'Fear of Failure in Exam': [map_categorical_values(fear_of_failure)],
        'Alcohol or Drugs': [map_categorical_values(alcohol_or_drugs)],
        'Social Media Usage': [social_media_usage],
        'Alone': [map_categorical_values(alone)],
        'Study Hours': [study_hours],
        'Age': [age],
    }, index=[0])

    result = depressed_model.predict(features)[0]

    if result == 1:
        return 'Yes'
    else:
        return 'No'

def predict_anxiety(name, age, gender, year, work_load, edu_level, fear_of_failure, alcohol_or_drugs, social_media_usage, alone, study_hours):
    # Convert gender to numerical value
    gender_numeric = 0 if gender == 'Male' else 1

    features = pd.DataFrame({
        'Name': [name],
        'Gender': [gender_numeric],
        'Year': [year],
        'Work Load': [work_load],
        'Edu Level': [edu_level],
        'Fear of Failure in Exam': [map_categorical_values(fear_of_failure)],
        'Alcohol or Drugs': [map_categorical_values(alcohol_or_drugs)],
        'Social Media Usage': [social_media_usage],
        'Alone': [map_categorical_values(alone)],
        'Study Hours': [study_hours],
        'Age': [age],
    }, index=[0])

    result = anxiety_model.predict(features)[0]

    if result == 1:
        return 'Yes'
    else:
        return 'No'



app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    name = request.form['name']
    age = int(request.form['age'])  # Convert age to int
    gender = request.form['gender']
    year = request.form['year']
    work_load = request.form['work_load']
    edu_level = request.form['edu_level']
    fear_of_failure = request.form['fear_of_failure']
    alcohol_or_drugs = request.form['alcohol_or_drugs']
    social_media_usage = int(request.form['social_media_usage'])
    alone = request.form['alone']
    study_hours = int(request.form['study_hours'])

    depression_result = predict_depression(name, age, gender, year, work_load, edu_level, fear_of_failure, alcohol_or_drugs, social_media_usage, alone, study_hours)
    anxiety_result = predict_anxiety(name, age, gender, year, work_load, edu_level, fear_of_failure, alcohol_or_drugs, social_media_usage, alone, study_hours)

    # Render the result page with the predicted results
    return render_template('result.html', name=name, age=age, depression_result=depression_result, anxiety_result=anxiety_result)

if __name__ == '__main__':
    app.run(debug=True)
