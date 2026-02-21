import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os

# ============================================================
# 1. CONFIGURACIÓN Y MENÚ LATERAL
# ============================================================
st.set_page_config(page_title="Valuador NPL — Grupo 4", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    [data-testid="stMetric"] {
        border-radius: 8px; padding: 12px 16px;
        border-left: 4px solid #3498db;
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

st.sidebar.title("📌 Menú Principal")
modo_app = st.sidebar.radio(
    "Selecciona qué deseas hacer:",
    ["👤 Calculadora Individual", "📊 Procesamiento Masivo (Lotes)"]
)
st.sidebar.markdown("---")

# ============================================================
# 2. CARGA DE MODELOS
# ============================================================
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
    st.error(f"Error crítico cargando modelos: {e}")
    st.stop()

# ============================================================
# 3. FUNCIONES DE PROCESAMIENTO Y FINANZAS
# ============================================================
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

    df['SCORE_CONTACTABILIDAD'] = 3.0 
    df['RATIO_CUOTA_SALDO'] = 0.02
    df['SEXO'] = df.get('SEXO', pd.Series('X', index=df.index)).apply(lambda v: 'M' if str(v).upper().strip() in ['M', 'MASCULINO'] else 'F')
    df['EST_CIVIL_CLEAN'] = 'OTROS'

    df_enc = pd.get_dummies(df, columns=['SEXO', 'EST_CIVIL_CLEAN'], drop_first=True)
    X = pd.DataFrame(0, index=range(len(df_enc)), columns=model_cols, dtype=float)
    for c in model_cols:
        if c in df_enc.columns: X[c] = df_enc[c].values

    return X, df

def calcular_van(flujos_mensuales, tasa_descuento_anual, inversion_inicial):
    tasa_mensual = (1 + tasa_descuento_anual) ** (1 / 12) - 1
    van = -inversion_inicial
    for t, flujo in enumerate(flujos_mensuales, 1):
        van += flujo / (1 + tasa_mensual) ** t
    return van

def calcular_tir(flujos_mensuales, inversion_inicial, max_iter=500, tol=1e-6):
    cashflows = np.array([-inversion_inicial] + list(flujos_mensuales))
    r = 0.02
    for _ in range(max_iter):
        npv = sum(cf / (1 + r) ** t for t, cf in enumerate(cashflows))
        dnpv = sum(-t * cf / (1 + r) ** (t + 1) for t, cf in enumerate(cashflows))
        if abs(dnpv) < 1e-14: break
        r_new = r - npv / dnpv
        if abs(r_new - r) < tol:
            r = r_new
            break
        r = r_new
    return (1 + r) ** 12 - 1

def distribuir_recuperacion_mensual(ve_total, perfil='decreciente', meses=12):
    if perfil == 'uniforme': return np.full(meses, ve_total / meses)
    elif perfil == 'decreciente':
        pesos = np.array([12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1], dtype=float)[:meses]
        return ve_total * (pesos / pesos.sum())
    return np.full(meses, ve_total / meses)

# ============================================================
# 4. INTERFAZ: MODO INDIVIDUAL
# ============================================================
if modo_app == "👤 Calculadora Individual":
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

# ============================================================
# 5. INTERFAZ: MODO MASIVO (DASHBOARD FULL)
# ============================================================
elif modo_app == "📊 Procesamiento Masivo (Lotes)":
    st.title("Dashboard Analítico Masivo")
    st.markdown("Sube tus archivos CSV para procesar miles de cuentas y generar el análisis financiero.")
    
    st.sidebar.header("📂 1. Carga de Datos")
    file_saldo = st.sidebar.file_uploader("Subir Saldo.csv", type=['csv'])
    file_detalles = st.sidebar.file_uploader("Subir Detalles.csv (Opcional)", type=['csv'])
    
    st.sidebar.header("⚙️ 2. Parámetros Financieros")
    precio_dolar = st.sidebar.slider("Precio de compra por $1", 0.01, 0.50, 0.05)
    costo_operativo_pct = st.sidebar.slider("Costo operativo (%)", 0, 30, 12)
    perfil_cobranza = st.sidebar.selectbox("Perfil de cobranza", ['decreciente', 'uniforme'])

    if file_saldo is None:
        st.info("👈 Sube tu archivo CSV en la barra lateral para habilitar el botón de ejecución.")
    else:
        # BOTÓN DE EJECUCIÓN AGREGADO AQUÍ
        ejecutar = st.button("▶️ EJECUTAR ANÁLISIS MASIVO", type="primary", use_container_width=True)
        
        # Usamos session_state para que el dashboard no desaparezca al tocar las pestañas
        if ejecutar:
            st.session_state['datos_procesados'] = True
            
        if st.session_state.get('datos_procesados', False):
            df_saldo = pd.read_csv(file_saldo, encoding='latin1')
            df_det = pd.read_csv(file_detalles, encoding='latin1') if file_detalles else None
            
            with st.spinner("Procesando lote con Machine Learning..."):
                X, df_orig = preprocesar_lote(df_saldo, df_det)
                prob_pago = clf.predict_proba(X)[:, 1]
                monto_pred = np.maximum(reg.predict(X), 0)
                valor_esperado = prob_pago * monto_pred
                
            # Agregamos las predicciones al dataframe original
            df_orig['Prob_Pago'] = prob_pago
            df_orig['Monto_Estimado'] = monto_pred
            df_orig['Valor_Esperado'] = valor_esperado
            
            # Segmentación de riesgo
            df_orig['Segmento'] = df_orig.apply(lambda r: 'Alto Potencial' if r['Prob_Pago'] >= 0.4 else ('Moderado' if r['Prob_Pago'] >= 0.2 else 'Bajo Potencial'), axis=1)

            # Cálculos Financieros
            saldo_tot = pd.to_numeric(df_orig['SALDO TOTAL'].apply(limpiar_moneda), errors='coerce').sum() if 'SALDO TOTAL' in df_orig.columns else valor_esperado.sum()
            ve_total = valor_esperado.sum()
            precio_compra = saldo_tot * precio_dolar
            costo_operativo = ve_total * (costo_operativo_pct / 100)
            recuperacion_neta = ve_total - costo_operativo
            utilidad = recuperacion_neta - precio_compra
            roi_global = utilidad / precio_compra if precio_compra > 0 else 0

            flujos = distribuir_recuperacion_mensual(recuperacion_neta, perfil_cobranza)
            try: tir_anual = calcular_tir(flujos, precio_compra)
            except: tir_anual = np.nan
            van_10 = calcular_van(flujos, 0.10, precio_compra)
            
            # KPIs Principales
            st.markdown("---")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Cuentas Totales", f"{len(X):,}")
            c2.metric("Saldo Facial", f"${saldo_tot:,.0f}")
            c3.metric("Valor Esperado (Bruto)", f"${ve_total:,.0f}")
            c4.metric("Precio de Compra", f"${precio_compra:,.0f}")
            
            c5, c6, c7, c8 = st.columns(4)
            c5.metric("Utilidad Neta", f"${utilidad:,.0f}")
            c6.metric("ROI Global", f"{roi_global:.1%}")
            c7.metric("TIR Anual", f"{tir_anual:.1%}" if not np.isnan(tir_anual) else "N/D")
            c8.metric("VAN (10%)", f"${van_10:,.0f}")

            # ============================================================
            # TABS DEL DASHBOARD (VISUALIZACIONES)
            # ============================================================
            tab1, tab2, tab3 = st.tabs(["📊 Distribución de Riesgo", "📈 Análisis Financiero", "📋 Detalle de Cuentas"])

            with tab1:
                st.subheader("Análisis de Riesgo y Segmentación")
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    fig_pie = px.pie(df_orig.groupby('Segmento')['Valor_Esperado'].sum().reset_index(), 
                                     values='Valor_Esperado', names='Segmento', 
                                     title="Distribución del Valor Esperado por Segmento", hole=0.4,
                                     color='Segmento', color_discrete_map={'Alto Potencial': '#2ecc71', 'Moderado': '#f39c12', 'Bajo Potencial': '#e74c3c'})
                    st.plotly_chart(fig_pie, use_container_width=True)
                with col_d2:
                    fig_prob = px.histogram(df_orig, x='Prob_Pago', nbins=40, 
                                            title="Distribución de Probabilidad de Pago",
                                            color_discrete_sequence=['#3498db'])
                    st.plotly_chart(fig_prob, use_container_width=True)

            with tab2:
                st.subheader("Cascada de Valor")
                fig_waterfall = go.Figure(go.Waterfall(
                    name="Flujo", orientation="v", measure=["absolute", "relative", "relative", "total"],
                    x=["Recuperación Bruta", "− Costo Operativo", "− Precio Compra", "Utilidad Neta"],
                    y=[ve_total, -costo_operativo, -precio_compra, utilidad],
                    text=[f"${ve_total:,.0f}", f"-${costo_operativo:,.0f}", f"-${precio_compra:,.0f}", f"${utilidad:,.0f}"],
                    textposition="outside", decreasing={"marker": {"color": "#e74c3c"}}, increasing={"marker": {"color": "#2ecc71"}},
                    totals={"marker": {"color": "#3498db" if utilidad > 0 else "#e74c3c"}}
                ))
                fig_waterfall.update_layout(height=400)
                st.plotly_chart(fig_waterfall, use_container_width=True)

            with tab3:
                st.subheader("Top 50 Cuentas Prioritarias (Mayor Valor Esperado)")
                cols_to_show = [c for c in ['CUENTA', 'SALDO TOTAL', 'DIAS MORA', 'Prob_Pago', 'Valor_Esperado', 'Segmento'] if c in df_orig.columns]
                
                df_display = df_orig.sort_values('Valor_Esperado', ascending=False).head(50)[cols_to_show]
                st.dataframe(df_display.style.format({
                    'Prob_Pago': '{:.2%}',
                    'Valor_Esperado': '${:,.2f}',
                    'SALDO TOTAL': '${:,.2f}'
                }), use_container_width=True, hide_index=True)
