import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import json

 
# CONFIGURACIÓN Y MENÚ LATERAL (LO PRIMERO QUE SE EJECUTA)

st.set_page_config(page_title="Valuador NPL — Grupo 4", layout="wide", initial_sidebar_state="expanded")

st.sidebar.title(" Menú Principal")
modo_app = st.sidebar.radio(
    "Selecciona qué deseas hacer:",
    ["Calculadora Individual", "Procesamiento Masivo (Lotes)"]
)
st.sidebar.markdown("---")

 
# CARGA DE MODELOS
 
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

@st.cache_resource
def load_artifacts():
    clf = joblib.load(os.path.join(MODELS_DIR, 'clasificador_pago.pkl'))
    reg = joblib.load(os.path.join(MODELS_DIR, 'regresor_monto.pkl'))
    cols = joblib.load(os.path.join(MODELS_DIR, 'columnas_modelo.pkl'))
    return clf, reg, cols

try:
    clf, reg, model_cols = load_artifacts()
except Exception as e:
    st.error(f"Error cargando modelos: {e}")
    st.stop()

 
# FUNCIONES DE PROCESAMIENTO

def preprocesar_cliente(saldo, dias_mora, antiguedad, recencia, edad, sexo, civil, score_contact=3.0, ratio_cuota=0.02):
    input_data = pd.DataFrame(columns=model_cols)
    input_data.loc[0] = 0
    input_data['LOG_SALDO'] = np.log1p(saldo)
    input_data['ANTIGUEDAD_MESES'] = antiguedad
    input_data['EDAD_CLIENTE'] = edad
    input_data['MESES_DESDE_ULTIMO_PAGO'] = recencia
    input_data['DIAS MORA'] = min(dias_mora, 720)
    input_data['SCORE_CONTACTABILIDAD'] = score_contact
    input_data['RATIO_CUOTA_SALDO'] = ratio_cuota
    if f'SEXO_{sexo}' in input_data.columns: input_data[f'SEXO_{sexo}'] = 1
    if f'EST_CIVIL_CLEAN_{civil}' in input_data.columns: input_data[f'EST_CIVIL_CLEAN_{civil}'] = 1
    return input_data

def limpiar_moneda(val):
    if isinstance(val, str):
        try: return float(val.replace('$', '').replace(',', '').replace(' ', ''))
        except: return 0.0
    return val

def preprocesar_lote(df_saldo, df_detalles=None):
    df = df_saldo.copy()
    if 'SALDO TOTAL' in df.columns: df['SALDO TOTAL'] = df['SALDO TOTAL'].apply(limpiar_moneda)
    
    if df_detalles is not None and 'CUENTA' in df.columns and 'CUENTA' in df_detalles.columns:
        df['CUENTA'] = pd.to_numeric(df['CUENTA'], errors='coerce')
        df_det = df_detalles.copy()
        df_det['CUENTA'] = pd.to_numeric(df_det['CUENTA'], errors='coerce')
        df = pd.merge(df, df_det, on='CUENTA', how='left', suffixes=('', '_det'))

    df['LOG_SALDO'] = np.log1p(df['SALDO TOTAL'].fillna(0).astype(float)) if 'SALDO TOTAL' in df.columns else 0
    df['DIAS MORA'] = pd.to_numeric(df.get('DIAS MORA', 360), errors='coerce').fillna(360).clip(upper=720)

    fecha_ref = pd.Timestamp.now()
    if 'FECHA_APERTURA' in df.columns:
        df['FECHA_APERTURA'] = pd.to_datetime(df['FECHA_APERTURA'], dayfirst=True, errors='coerce')
        df['ANTIGUEDAD_MESES'] = (fecha_ref - df['FECHA_APERTURA']).dt.days / 30
    df['ANTIGUEDAD_MESES'] = df.get('ANTIGUEDAD_MESES', pd.Series(36, index=df.index)).fillna(36)
    
    if 'FECHA NACIMIENTO' in df.columns:
        df['FECHA NACIMIENTO'] = pd.to_datetime(df['FECHA NACIMIENTO'], dayfirst=True, errors='coerce')
        df['EDAD_CLIENTE'] = (fecha_ref - df['FECHA NACIMIENTO']).dt.days / 365.25
    df['EDAD_CLIENTE'] = df.get('EDAD_CLIENTE', pd.Series(40, index=df.index)).fillna(40)
    
    if 'FECHA ULTIMO PAGO' in df.columns:
        df['FECHA ULTIMO PAGO'] = pd.to_datetime(df['FECHA ULTIMO PAGO'], dayfirst=True, errors='coerce')
        df['MESES_DESDE_ULTIMO_PAGO'] = (fecha_ref - df['FECHA ULTIMO PAGO']).dt.days / 30
    df['MESES_DESDE_ULTIMO_PAGO'] = df.get('MESES_DESDE_ULTIMO_PAGO', pd.Series(999, index=df.index)).fillna(999)

    df['SCORE_CONTACTABILIDAD'] = 3.0 # Simplificado para evitar errores si faltan columnas
    df['RATIO_CUOTA_SALDO'] = 0.02
    df['SEXO'] = df.get('SEXO', pd.Series('X', index=df.index)).apply(lambda v: 'M' if str(v).upper().strip() in ['M', 'MASCULINO'] else 'F')
    df['EST_CIVIL_CLEAN'] = 'OTROS'

    df_enc = pd.get_dummies(df, columns=['SEXO', 'EST_CIVIL_CLEAN'], drop_first=True)
    X = pd.DataFrame(0, index=range(len(df_enc)), columns=model_cols, dtype=float)
    for c in model_cols:
        if c in df_enc.columns: X[c] = df_enc[c].values

    return X, df

 
# LÓGICA DE LA INTERFAZ SEGÚN EL MENÚ ELEGIDO
 

if modo_app == "Calculadora Individual":
    st.title("Valuador Individual de Cartera NPL")
    st.markdown("Evalúa a un cliente específico ingresando sus datos manualmente.")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        saldo_raw = st.number_input("Saldo Total ($)", 1.0, 500000.0, 5000.0)
        dias_mora = st.number_input("Días de Mora", 0, 3000, 180)
        antiguedad = st.slider("Antigüedad (Meses)", 0, 200, 24)
    with c2:
        edad = st.slider("Edad", 18, 90, 35)
        sexo = st.selectbox("Sexo", ["M", "F", "X"])
        civil = st.selectbox("Estado Civil", ["SOLTERO", "CASADO", "DIVORCIADO", "UNION_LIBRE", "OTROS"])
    with c3:
        recencia = st.slider("Meses sin pago", 0, 100, 6)
        score_contact = st.slider("Score Contactabilidad", 0.0, 6.0, 3.0)
        ratio_cuota = st.number_input("Ratio Cuota/Saldo", 0.0, 1.0, 0.02)

    if st.button("CALCULAR VALUACIÓN", type="primary"):
        vec = preprocesar_cliente(saldo_raw, dias_mora, antiguedad, recencia, edad, sexo, civil, score_contact, ratio_cuota)
        prob = clf.predict_proba(vec)[0, 1]
        monto = max(reg.predict(vec)[0], 0)
        ev = prob * monto

        st.success("Cálculo completado")
        m1, m2, m3 = st.columns(3)
        m1.metric("Probabilidad de Pago", f"{prob:.1%}")
        m2.metric("Recuperación Estimada", f"${monto:,.2f}")
        m3.metric("Valor Esperado Total", f"${ev:,.2f}")

elif modo_app == "Procesamiento Masivo (Lotes)":
    st.title("Dashboard Analítico Masivo")
    st.markdown("Sube tus archivos CSV para procesar miles de cuentas a la vez.")
    
    # Estos controles solo aparecen en la barra lateral SI estás en modo masivo
    st.sidebar.header("Carga de Datos")
    file_saldo = st.sidebar.file_uploader("Subir Saldo.csv", type=['csv'])
    file_detalles = st.sidebar.file_uploader("Subir Detalles.csv (Opcional)", type=['csv'])
    
    st.sidebar.header("Parámetros")
    precio_dolar = st.sidebar.slider("Precio de compra por $1 de saldo", 0.01, 0.50, 0.05)

    if file_saldo is None:
        st.info("Sube tu archivo CSV en la barra lateral para ver los resultados.")
    else:
        df_saldo = pd.read_csv(file_saldo, encoding='latin1')
        df_det = pd.read_csv(file_detalles, encoding='latin1') if file_detalles else None
        
        with st.spinner("Procesando lote con Machine Learning..."):
            X, df_orig = preprocesar_lote(df_saldo, df_det)
            prob_pago = clf.predict_proba(X)[:, 1]
            monto_pred = np.maximum(reg.predict(X), 0)
            valor_esperado = prob_pago * monto_pred
            
        saldo_tot = pd.to_numeric(df_orig['SALDO TOTAL'].apply(limpiar_moneda), errors='coerce').sum() if 'SALDO TOTAL' in df_orig.columns else valor_esperado.sum()
        precio_compra = saldo_tot * precio_dolar
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Cuentas Totales", f"{len(X):,}")
        c2.metric("Saldo Facial", f"${saldo_tot:,.0f}")
        c3.metric("Valor Esperado (Modelo)", f"${valor_esperado.sum():,.0f}")
        c4.metric("Costo de Compra", f"${precio_compra:,.0f}")
        
        st.markdown("### Top Cuentas Prioritarias")
        df_orig['Prob_Pago'] = prob_pago
        df_orig['Valor_Esperado'] = valor_esperado
        st.dataframe(df_orig.sort_values('Valor_Esperado', ascending=False).head(30), use_container_width=True)
