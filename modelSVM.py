import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# Load the data
dataset = pd.read_csv('datasheet.csv')  # Gantilah 'your_data.csv' dengan nama file data Anda

# Split the data into features and target
x = dataset.drop (columns='Diagnosis', axis=1)
y = dataset['Diagnosis']

scaler = StandardScaler()
scaler.fit(x)
standarized_data = scaler.transform(x)

x = standarized_data
y = dataset['Diagnosis']
# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,stratify=y,random_state=2)

# Train a random forest classifier
classifier = svm.SVC(kernel='linear')
classifier.fit(x_train, y_train)

# Make predictions on the test set
x_train_prediction = classifier.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction, y_train)
x_test_prediction = classifier.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction, y_test)

st.title('Prediksi Penyakit')

# membagi kolom
col1, col2 = st.columns(2)

with col1:
    nama = st.text_input('Nama')
    Hb = st.text_input('Hb (gr%)')
    Sistol = st.text_input('Sistol')
    Diastol = st.text_input('Diastol')
    ProteinUrine = st.text_input('Protein Urine')
    Interval = st.text_input('Interval')

with col2:
    usia = st.text_input('Usia')
    Hight = st.text_input('Height')
    Weight = st.text_input('Weight')
    BMI = st.text_input('BMI')
    HistoryofPE = st.text_input('History of PE')
    HistoryofHipertensi = st.text_input('History of Hipertensi')

# Input untuk MAP
MAP = st.text_input('MAP')

# Create a user input DataFrame
input_data = pd.DataFrame({
    'Hb (gr%)': [Hb],
    'Sistol (mmHg)': [Sistol],
    'Diastol (mmHg)': [Diastol],
    'Protein Urine': [ProteinUrine],
    'Interval (month)': [Interval],
    'Hight (cm)': [Hight],
    'Weight (kg)': [Weight],
    'BMI': [BMI],
    'History of PE': [HistoryofPE],
    'History of Hipertensi': [HistoryofHipertensi],
    'MAP': [MAP]
})

# Predict the diagnosis
if st.button('Prediksi Penyakit'):
    input_data_as_numpy_array = np.array(input_data)
    input_data_reshape = input_data_as_numpy_array.reshape(1, -1)
    std_data = scaler.transform(input_data_reshape)
    prediction = classifier.predict(std_data)

    if prediction[0] == 1:
        diagnosis = "Pasien diduga terkena Hipertensi kronis"
    elif prediction[0] == 2:
        diagnosis = "Pasien diduga terkena Hipertensi pada masa kehamilan"
    elif prediction[0] == 3:
        diagnosis = "Pasien diduga terkena Preeklamsia"
    elif prediction[0] == 4:
        diagnosis = "Pasien diduga terkena Preeklamsia berat"
    elif prediction[0] == 5:
        diagnosis = "Pasien diduga terkena Superimposed Preeklampsia"
    else:
        diagnosis = "Pasien Sehat"

    # Menampilkan nama, usia, dan hasil prediksi
    st.success(f'Nama: {nama}')
    st.success(f'Usia: {usia}')
    st.success(diagnosis)
