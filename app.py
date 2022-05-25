import streamlit as st
import pickle
import pandas as pd
import numpy as np

# import the model
pipe = pickle.load(open('pipe.pkl','rb'))
df = pickle.load(open('df.pkl','rb'))

st.title("Heart Disease Predictor by Jazan studen")
# st.text("1 means you've a heart disease, 0 means you don't have a heart disease")
#Age
Age = st.number_input('Age of Person')
#Sex
Sex = st.selectbox('Sex',df['sex'].unique())
#ChestPainType
cp = st.selectbox('Chest Pain Type',df['cp'].unique())
#RestingBP
trestbps = st.number_input('Resting Blood Pressure')
#Cholesterol
chol = st.number_input('Serum Cholesterol')
#FastingBS
fbs = st.selectbox('Fasting Blood Sugar',[1,0])
#RestingECG
restecg = st.selectbox('Resting Electrocardiogram Results',df['restecg'].unique())
#MaxHR
thalach = st.number_input('Maximum Heart Rate Achieved')
#ExerciseAngina
exang = st.selectbox('Exercise Induced Angina',df['exang'].unique())
#Oldpeak
Oldpeak = weight = st.number_input('Old peak')
#ST_Slope
slope = st.selectbox('The slope of the peak exercise ST segment',df['slope'].unique())
if st.button('Predict Heart Health'):
    if (Age > 65 and trestbps > 170 and chol > 265 and thalach > 210):
       st.header("You've a heart disease. Please, consult your doctor")
    else :
       query = np.array([Age,Sex,cp,trestbps,chol,fbs,restecg,thalach,exang,Oldpeak,slope])
       query = query.reshape(1, 11)
       result=str(pipe.predict(query)[0])
       if result == 1:
          st.header("You've a heart disease. Please, consult your doctor")
       else:
           st.header("You don't have a heart disease")
            
            # Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the Random Forest CLassifier model
filename = 'heart-disease-prediction-knn-model.pkl'
model = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('main.html')


@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':

        age = int(request.form['age'])
        sex = request.form.get('sex')
        cp = request.form.get('cp')
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = request.form.get('fbs')
        restecg = int(request.form['restecg'])
        thalach = int(request.form['thalach'])
        exang = request.form.get('exang')
        oldpeak = float(request.form['oldpeak'])
        slope = request.form.get('slope')
        ca = int(request.form['ca'])
        thal = request.form.get('thal')
        
        data = np.array([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])
        my_prediction = model.predict(data)
        
        return render_template('result.html', prediction=my_prediction)
        
        

if __name__ == '__main__':
	app.run(debug=True)


