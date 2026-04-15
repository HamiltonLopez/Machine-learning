import streamlit as st
import pandas as pd
import joblib
import json

# 1. Configuración de la página
st.set_page_config(page_title="Predicción de Fuga de Clientes (Churn)", layout="centered")
st.title("📡 Predicción de Fuga de Clientes - Telecomunicaciones")
st.write("Ingrese los datos del cliente para predecir si existe riesgo de que abandone el servicio (Churn).")

# 2. Cargar el modelo y el diccionario de categorías
@st.cache_resource
def load_resources():
    model = joblib.load('modelo_churn_telecom.joblib')
    with open('categorias_churn.json', 'r') as f:
        categories = json.load(f)
    return model, categories

modelo, diccionario_categorias = load_resources()

# 3. Interfaz para ingresar los datos
st.header("👤 Datos del Cliente")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Demografía y Cuenta")
    senior_citizen_input = st.selectbox("¿Es adulto mayor (Senior Citizen)?", options=["No", "Sí"])
    senior_citizen = 1 if senior_citizen_input == "Sí" else 0
    
    partner = st.selectbox("¿Tiene pareja (Partner)?", options=diccionario_categorias['Partner'])
    dependents = st.selectbox("¿Tiene dependientes (Dependents)?", options=diccionario_categorias['Dependents'])
    
    tenure = st.number_input("Meses de antigüedad (Tenure)", min_value=0, max_value=120, value=1)
    
    contract = st.selectbox("Tipo de Contrato (Contract)", options=diccionario_categorias['Contract'])
    payment_method = st.selectbox("Método de Pago (PaymentMethod)", options=diccionario_categorias['PaymentMethod'])
    monthly_charges = st.number_input("Cargo Mensual ($ MonthlyCharges)", min_value=0.0, max_value=300.0, value=50.0, step=0.5)

with col2:
    st.subheader("Servicios Contratados")
    multiple_lines = st.selectbox("Múltiples Líneas", options=diccionario_categorias['MultipleLines'])
    internet_service = st.selectbox("Servicio de Internet", options=diccionario_categorias['InternetService'])
    online_security = st.selectbox("Seguridad Online", options=diccionario_categorias['OnlineSecurity'])
    online_backup = st.selectbox("Respaldo Online", options=diccionario_categorias['OnlineBackup'])
    device_protection = st.selectbox("Protección del Dispositivo", options=diccionario_categorias['DeviceProtection'])
    tech_support = st.selectbox("Soporte Técnico", options=diccionario_categorias['TechSupport'])
    streaming_tv = st.selectbox("Streaming TV", options=diccionario_categorias['StreamingTV'])
    streaming_movies = st.selectbox("Streaming Películas", options=diccionario_categorias['StreamingMovies'])

# 4. Botón de Predicción
st.markdown("---")
if st.button("🔍 Realizar Predicción", use_container_width=True):
    
    # Simular el LabelEncoder: Obtener el índice numérico que le corresponde a la categoría elegida
    input_data = pd.DataFrame([{
        'SeniorCitizen': senior_citizen,
        'Partner': diccionario_categorias['Partner'].index(partner),
        'Dependents': diccionario_categorias['Dependents'].index(dependents),
        'tenure': tenure,
        'MultipleLines': diccionario_categorias['MultipleLines'].index(multiple_lines),
        'InternetService': diccionario_categorias['InternetService'].index(internet_service),
        'OnlineSecurity': diccionario_categorias['OnlineSecurity'].index(online_security),
        'OnlineBackup': diccionario_categorias['OnlineBackup'].index(online_backup),
        'DeviceProtection': diccionario_categorias['DeviceProtection'].index(device_protection),
        'TechSupport': diccionario_categorias['TechSupport'].index(tech_support),
        'StreamingTV': diccionario_categorias['StreamingTV'].index(streaming_tv),
        'StreamingMovies': diccionario_categorias['StreamingMovies'].index(streaming_movies),
        'Contract': diccionario_categorias['Contract'].index(contract),
        'PaymentMethod': diccionario_categorias['PaymentMethod'].index(payment_method),
        'MonthlyCharges': monthly_charges
    }])
    
    # Realizar la predicción conectando el dataframe al pipeline exportado
    prediccion = modelo.predict(input_data)
    probabilidades = modelo.predict_proba(input_data)[0]
    
    st.subheader("Resultado de la Predicción:")
    if prediccion[0] == 1:
        st.error(f"🚨 **¡ALERTA DE ABANDONO (CHURN)!**🚨\n\nEl modelo predice que es altamente probable que este cliente cancele su servicio. (Probabilidad: {probabilidades[1]*100:.2f}%)")
    else:
        st.success(f"✅ **CLIENTE RETENIDO**\n\nEl modelo predice que el cliente continuará usando el servicio. (Probabilidad de quedarse: {probabilidades[0]*100:.2f}%)")
