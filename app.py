
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


