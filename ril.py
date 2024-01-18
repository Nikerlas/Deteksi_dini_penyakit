import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
import base64

def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img = get_img_as_base64("background.jpg")

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] {{
background-image: url("data:image/png;base64,{img}");
background-size: cover;
}}
</style>
"""
# gambar
st.markdown(page_bg_img, unsafe_allow_html=True)

# Load the original training data
original_data = pd.read_csv('datasheet.csv')

# Split the data into features and target
x = original_data.drop(columns='Diagnosis', axis=1)
y = original_data['Diagnosis']

# Initialize or load your KNN model
scaler = StandardScaler()
scaler.fit(x)
standarized_data = scaler.transform(x)
x = standarized_data

# Split the data into training
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)

k = 5  # Number of neighbors
classifier = KNeighborsClassifier(n_neighbors=k)
classifier.fit(x_train, y_train)

st.title('Prediksi Penyakit')

dt_classifier = DecisionTreeClassifier(random_state=2)
dt_classifier.fit(x_train, y_train)

ensemble_classifier = VotingClassifier(estimators=[
    ('knn', classifier),
    ('dt', dt_classifier)
], voting='hard')

ensemble_classifier.fit(x_train, y_train)

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
        diagnosis = "1"
        teks = "Chronic Hypertension"
    elif prediction[0] == 2:
        diagnosis = "2"
        teks = "Hypertension in Pregnancy"
    elif prediction[0] == 3:
        diagnosis = "3"
        teks = "Preeclampsia"
    elif prediction[0] == 4:
        diagnosis = "4"
        teks = "Severe Preeclampsia"
    elif prediction[0] == 5:
        diagnosis = "5"
        teks = "Superimposed Preeclampsia"
    else:
        diagnosis = "6"
        teks = "Healthy Pregnant Women"
        

    # Menampilkan nama, usia, dan hasil prediksi
    st.success(f'Nama: {nama}')
    st.success(f'Usia: {usia}')
    st.success(teks)

   # Simpan input data ke dalam file CSV
    input_data['Diagnosis'] = [diagnosis]
    with open('datasheet.csv', 'a', newline='') as file:
        input_data[['Hb (gr%)', 'Sistol (mmHg)', 'Diastol (mmHg)', 'Protein Urine', 'Interval (month)',
                     'Hight (cm)', 'Weight (kg)', 'BMI', 'History of PE', 'History of Hipertensi', 'MAP', 'Diagnosis']].to_csv(
            file, mode='a', header=file.tell() == 0, index=False
        )



# Tambahkan tombol untuk mengunduh file CSV
if st.button('Unduh Data Input'):
    st.download_button(
        label="Klik di sini untuk mengunduh data input",
        data=input_data.to_csv(index=False),
        key='input_data.csv',
        on_click=None,
    )

