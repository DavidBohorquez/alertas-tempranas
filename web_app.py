import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from PIL import Image

# get images
image_matriz_correlacion = Image.open('img/matriz_correlacion_coef_peirson.jpeg')
image_variables_relevantes = Image.open('img/variables_relevantes.jpeg')
image_matriz_confusion1 = Image.open('img/matriz_confusion_modelo2.jpeg')
image_matriz_confusion2 = Image.open('img/matriz_confusion_modelo1.jpeg')
image_configruacion_modelo_ia = Image.open('img/configuracion_perceptron_multicapa.jpeg')

st.title('Centro de Bienestar Institucional (PDI) - Universidad Distrital FJC')
st.write('''
    ### Integrantes:
    * David: ndbohorquezg@correo.udistrital.edu.co
    * Jordy: jepinedav@correo.udistrital.edu.co
    * Oscar : ocgutierrezv@correo.udistrital.edu.co 
    * Daniel : danarodriguezs@correo.udistrital.edu.co
''')

st.write('Creación de una aplicación web que permita el análisis y predicción de la deserción académica de los estudiantes basados en su historial, que le sirva a las diferentes entidades educativas para tomar decisiones correctivas en etapas tempranas del proceso educativo y evaluar su impacto en el porcentaje de abandono de los sistemas educativos.')

application_mode = st.sidebar.selectbox('Formato de aplicación', (1, 2, 5, 7, 10, 15, 16, 17, 18, 26, 27, 39, 42, 43, 44, 51, 53, 57))
course = st.sidebar.selectbox('Carrera', (33, 171, 8014, 9003, 9070, 9085, 9119, 9130, 9147, 9238, 9254, 9500, 9556, 9670, 9773, 9853, 9991))
previous_qualification_grade = st.sidebar.number_input('Puntaje de admisión previo', 0, 200, 100)
admission_grade = st.sidebar.number_input('Puntaje de admisión', 0, 200, 100)
debtor = st.sidebar.selectbox('¿Es deudor?', (0,1))
tutiton_fees_up_to_date = st.sidebar.selectbox('Matricula al dia', (0,1))
gender = st.sidebar.selectbox('Sexo', (0,1))
scholarship_holder = st.sidebar.selectbox('Becado', (0,1))
age_at_enrollment = st.sidebar.number_input('Edad de ingreso', 15, 60, 18)
curricular_units_1nd_sem_enrolled = st.sidebar.number_input('Número créditos inscritos 1º semestre', 0, 20, 10)
curricular_units_1st_sem_evaluations = st.sidebar.number_input('Número evaluaciones 1º semestre', 0, 200, 10)
curricuar_units_1nd_sem_approved = st.sidebar.number_input('Unidades curriculares 1º sem (aprobado)', 0, 20, 10)
curricuar_units_1nd_sem_grade = st.sidebar.number_input('Unidades curriculares 1º sem (promedio de calificaciones)', 0, 20, 10)

curricular_units_2nd_sem_enrolled = st.sidebar.number_input('Número créditos inscritos 2º semestre', 0, 20, 10)
curricuar_units_2nd_sem_evaluations = st.sidebar.number_input('Número evaluaciones 2º semestre', 0, 200, 10)
curricuar_units_2nd_sem_approved = st.sidebar.number_input('Unidades curriculares 2º sem (aprobado)', 0, 20, 10)
curricuar_units_2nd_sem_grade = st.sidebar.number_input('Unidades curriculares 2º sem (promedio de calificaciones)', 0, 20, 10)
unemployment_rate = st.sidebar.number_input('Tasa de desempleo', 0, 100, 8)


data_predict = {
    'application_mode': application_mode,
    'course': course,
    'previous_qualification_grade': previous_qualification_grade,
    'admission_grade': admission_grade,
    'debtor': debtor,
    'tutiton_fees_up_to_date': tutiton_fees_up_to_date,
    'gender': gender,
    'scholarship_holder': scholarship_holder,
    'age_at_enrollment': age_at_enrollment,
    'curricular_units_1nd_sem_enrolled': curricular_units_1nd_sem_enrolled,
    'curricular_units_1st_sem_evaluations': curricular_units_1st_sem_evaluations,
    'curricular_units_1nd_sem_approved': curricuar_units_1nd_sem_approved,
    'curricular_units_1nd_sem_grade': curricuar_units_1nd_sem_grade,
    'curricular_units_2nd_sem_enrolled': curricular_units_2nd_sem_enrolled,
    'curricluar_units_2nd_sem_evaluations': curricuar_units_2nd_sem_evaluations, 
    'curricular_units_2nd_sem_approved': curricuar_units_2nd_sem_approved,
    'curricular_units_2nd_sem_grade': curricuar_units_2nd_sem_grade,
    'unemployment_rate': unemployment_rate
}


features = pd.DataFrame(data_predict, index=[0])

DATA_URL = ('https://raw.githubusercontent.com/DavidBohorquez/Hack4Edu/main/Dropout_Academic%20Success%20-%20Sheet1.csv')
@st.cache
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    return data


data = load_data(10)


st.subheader('Conjunto de datos original')
st.write('[Se escogió el conjunto de datos de educación universitaria en Portugal debido a que estos datos ya se encuentran organizados, y a que los conjuntos de datos de África se encuentran caídos o aún falta por hacer la limpieza de los datos](https://www.kaggle.com/datasets/ankanhore545/dropout-or-academic-success?select=Dropout_Academic+Success+-+Sheet1.csv)')
st.write(data)

st.write('## Preprocesamiento de datos')
st.write('### Reducción de dimensionalidad')
st.write('''
Lo primero que se hizo fue la creación de una matriz de correlación con la que 
se pudieran evaluar cuales atributos eran los mas influyentes y así reducir el número
de variables que tiene que usar el modelo de inteligencia artificial.

#### Matriz de correlaciones
''')
st.image(image_matriz_correlacion, caption='Matriz de correlación')

st.write('#### Clasificador de árboles extra')
st.image(image_variables_relevantes, caption='Variables relavantes para el modelo')

st.write('''#### Selección de atributos
Conjunto de datos de 10 registros para validación del modelo con los 18 atributos seleccionados de los 36 atributos originales''')
st.write(data[['application mode',
    'course',
    'previous qualification (grade)',
    'admission grade',
    'debtor',
    'tuition fees up to date',
    'gender',
    'scholarship holder',
    'age at enrollment',
    'curricular units 1st sem (enrolled)',
    'curricular units 1st sem (evaluations)',
    'curricular units 1st sem (approved)',
    'curricular units 1st sem (grade)',
    'curricular units 2nd sem (enrolled)',
    'curricular units 2nd sem (evaluations)',
    'curricular units 2nd sem (approved)', 
    'curricular units 2nd sem (grade)',
    'unemployment rate',
    'target']])

st.write('''
    ### Construcción de modelos de IA
    Se crearon dos modelos de inteligencia artificial, los dos modelos comparten los mismos 18 atributos que se pueden ver
    en el panel izquierdo, con diferencia que el primero modelo también cuenta otros 2 atributos que son las ocupaciones de la madre y el padre del estudiante
    ''')
st.image(image_matriz_confusion1, caption="matriz de confusión - modelo 1")
st.image(image_matriz_confusion2, caption="matriz de confusión - modelo 2")

st.write('''
    Ambos modelos se construyeron con base a un modelo de perceptrón multicapa de tres capas, cada una con 10 neuronas.
''')
st.image(image_configruacion_modelo_ia, caption="configuración perceptrón multicapa")

st.write('''
    ## Resultados del modelo predictivo
    El modelo de inteligencia artificial puede predecir basado en datos
    históricos y socioeconomicos de los estudiantes si estos se van a retirar o no
    de la carrera que cursan en la universidad.
    Esta página web le permite interactuar con el modelo, alterando las diferentes variables
    de entrada que se encuentran en el panel de la izquierda y así obtener un resultado.

    [De clic aquí para obtener una descripción detallada de los datos](https://drive.google.com/file/d/196pHYmHcZWwuKEgetPp5GQQ3cSZXpZ-C/view)
''')

load_clf = pickle.load(open('Hack4Edu_1.pkl', 'rb'))
clasificacion = np.array(['ABANDONA','GRADUADO'])


prediction = load_clf.predict(features)
estado_prediccion = str(clasificacion[prediction][0])
st.write('''### Predicción del modelo: \n'''
            +"# "+estado_prediccion)
st.write(prediction)

st.write('''
    ## Referencias:
    * Hore, A. (2022). "Predict Dropout or Academic Success". Kaggle. Recuperado de: https://www.kaggle.com/datasets/ankanhore545/dropout-or-academic-success/discussion/333138?select=Dropout_Academic+Success+-+Sheet1.csv
    * Realinho V, Machado J, Baptista L, Martins MV. (2022) "Predicting Student Dropout and Academic Success". Data. 2022; 7(11):146. https://doi.org/10.3390/data7110146
    * Selwaness I, Adam T, Lawson L, & Heady L. (2022). Guidance Note on Education Data Mapping in Sub-Saharan Africa: Moving from theory to practice [Technical Report]. EdTech Hub. https://doi.org/10.53832/edtechhub.0096
''')
