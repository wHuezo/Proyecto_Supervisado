"""
Dashboard de Valoración de Cartera NPL — Grupo 4
================================================
App interactiva Streamlit + Plotly para evaluar lotes de créditos vencidos,
generar un dashboard dinámico de resultados y apoyar la decisión de compra
con funciones de retorno (ROI, VAN, TIR) y simulación de escenarios.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import json

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================
st.set_page_config(
    page_title="Dashboard NPL — Grupo 4",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Estilo CSS compacto
st.markdown("""
<style>
    .block-container {padding-top: 1rem; padding-bottom: 1rem;}
    [data-testid="stMetric"] {
        border-radius: 8px; padding: 12px 16px;
        border-left: 4px solid #3498db;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
    }
    h1 {font-size: 1.8rem !important;}
    h2 {font-size: 1.3rem !important; border-bottom: 2px solid #ecf0f1; padding-bottom: 6px;}
    h3 {font-size: 1.1rem !important;}
    .stTabs [data-baseweb="tab-list"] {gap: 4px;}
    .stTabs [data-baseweb="tab"] {padding: 8px 20px;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# CARGA DE MODELOS
# ============================================================
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')


@st.cache_resource
def load_artifacts():
    clf = joblib.load(os.path.join(MODELS_DIR, 'clasificador_pago.pkl'))
    reg = joblib.load(os.path.join(MODELS_DIR, 'regresor_monto.pkl'))
    cols = joblib.load(os.path.join(MODELS_DIR, 'columnas_modelo.pkl'))
    with open(os.path.join(MODELS_DIR, 'config_mejores_params.json'), 'r') as f:
        config = json.load(f)
    return clf, reg, cols, config


try:
    clf, reg, model_cols, model_config = load_artifacts()
except Exception as e:
    st.error(f"Error cargando modelos: {e}")
    st.info("Ejecuta el Notebook 4 primero para generar los artefactos `.pkl`.")
    st.stop()


# ============================================================
# FUNCIONES DE PREPROCESAMIENTO
# ============================================================
def limpiar_moneda(val):
    if isinstance(val, str):
        val = val.replace('$', '').replace(',', '').replace(' ', '')
        try:
            return float(val)
        except ValueError:
            return 0.0
    return val


def preprocesar_lote(df_saldo, df_detalles=None):
    """Preprocesa un lote completo y retorna (X, df_original)."""
    df = df_saldo.copy()

    for col in ['SALDO TOTAL', 'VALOR CUOTA']:
        if col in df.columns:
            df[col] = df[col].apply(limpiar_moneda)

    if df_detalles is not None and 'CUENTA' in df.columns and 'CUENTA' in df_detalles.columns:
        df['CUENTA'] = pd.to_numeric(df['CUENTA'], errors='coerce')
        df_det = df_detalles.copy()
        df_det['CUENTA'] = pd.to_numeric(df_det['CUENTA'], errors='coerce')
        df = pd.merge(df, df_det, on='CUENTA', how='left', suffixes=('', '_det'))

    # --- Features numéricas ---
    df['LOG_SALDO'] = np.log1p(df['SALDO TOTAL'].fillna(0).astype(float)) if 'SALDO TOTAL' in df.columns else 0
    df['DIAS MORA'] = pd.to_numeric(df.get('DIAS MORA', 360), errors='coerce').fillna(360).clip(upper=720)

    for col in ['FECHA_APERTURA', 'FECHA NACIMIENTO', 'FECHA ULTIMO PAGO']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')

    fecha_ref = pd.Timestamp.now()

    if 'FECHA_APERTURA' in df.columns:
        df['ANTIGUEDAD_MESES'] = (fecha_ref - df['FECHA_APERTURA']) / pd.Timedelta(days=30)
        df['ANTIGUEDAD_MESES'] = df['ANTIGUEDAD_MESES'].fillna(df['ANTIGUEDAD_MESES'].median())
    else:
        df['ANTIGUEDAD_MESES'] = 36

    if 'FECHA NACIMIENTO' in df.columns:
        df['EDAD_CLIENTE'] = (fecha_ref - df['FECHA NACIMIENTO']) / pd.Timedelta(days=365.25)
        med = df['EDAD_CLIENTE'].median()
        df.loc[(df['EDAD_CLIENTE'] < 18) | (df['EDAD_CLIENTE'] > 90), 'EDAD_CLIENTE'] = med
        df['EDAD_CLIENTE'] = df['EDAD_CLIENTE'].fillna(med if pd.notna(med) else 40)
    else:
        df['EDAD_CLIENTE'] = 40

    if 'FECHA ULTIMO PAGO' in df.columns:
        df['MESES_DESDE_ULTIMO_PAGO'] = (fecha_ref - df['FECHA ULTIMO PAGO']) / pd.Timedelta(days=30)
        df['MESES_DESDE_ULTIMO_PAGO'] = df['MESES_DESDE_ULTIMO_PAGO'].fillna(999)
    else:
        df['MESES_DESDE_ULTIMO_PAGO'] = 999

    # Score de contactabilidad
    email = df.get('EMAIL CLIENTE', pd.Series('', index=df.index)).fillna('').astype(str).str.strip()
    email = email.apply(lambda x: 1 if len(x) > 0 and x != '0' and x.lower() != 'nan' else 0)
    direccion = df.get('DIRECCION RESIDENCIAL', pd.Series('', index=df.index)).fillna('').astype(str).str.strip()
    direccion = direccion.apply(lambda x: 1 if len(x) > 0 and x != '0' and x.lower() != 'nan' else 0)
    trabajo = df.get('LUGAR_TRABAJO', pd.Series('', index=df.index)).fillna('').astype(str).str.strip()
    trabajo = trabajo.apply(lambda x: 1 if len(x) > 0 and x != '0' and x.lower() != 'nan' else 0)
    telefonos = pd.to_numeric(df.get('TELEFONOS', pd.Series(0, index=df.index)), errors='coerce').fillna(0)
    df['SCORE_CONTACTABILIDAD'] = email * 1 + direccion * 1 + trabajo * 2 + telefonos * 0.5

    cuota = df.get('VALOR CUOTA', pd.Series(0, index=df.index)).fillna(0).astype(float)
    saldo_v = df.get('SALDO TOTAL', pd.Series(1, index=df.index)).fillna(1).astype(float)
    df['RATIO_CUOTA_SALDO'] = np.where(saldo_v > 0, cuota / saldo_v, 0)

    # --- Categóricas ---
    if 'SEXO' in df.columns:
        df['SEXO'] = df['SEXO'].apply(
            lambda v: 'M' if str(v).upper().strip() in ['M', 'MASCULINO']
            else ('F' if str(v).upper().strip() in ['F', 'FEMENINO'] else 'X'))
    else:
        df['SEXO'] = 'X'

    if 'EST CIVIL' in df.columns:
        def _norm_civil(val):
            val = str(val).upper().strip()
            if val in ['S', 'SOLTERO', 'SOLTERA']:
                return 'SOLTERO'
            if val in ['C', 'CASADO', 'CASADA']:
                return 'CASADO'
            if val in ['D', 'DIVORCIADO']:
                return 'DIVORCIADO'
            if val in ['U', 'UL', 'UNION LIBRE', 'UNION_LIBRE']:
                return 'UNION_LIBRE'
            return 'OTROS'
        df['EST_CIVIL_CLEAN'] = df['EST CIVIL'].apply(_norm_civil)
    else:
        df['EST_CIVIL_CLEAN'] = 'OTROS'

    df_enc = pd.get_dummies(df, columns=['SEXO', 'EST_CIVIL_CLEAN'], drop_first=True)

    X = pd.DataFrame(0, index=range(len(df_enc)), columns=model_cols, dtype=float)
    for c in model_cols:
        if c in df_enc.columns:
            X[c] = df_enc[c].values

    return X, df


# ============================================================
# FUNCIONES FINANCIERAS
# ============================================================
def calcular_van(flujos_mensuales, tasa_descuento_anual, inversion_inicial):
    """VAN con flujos mensuales y tasa anual."""
    tasa_mensual = (1 + tasa_descuento_anual) ** (1 / 12) - 1
    van = -inversion_inicial
    for t, flujo in enumerate(flujos_mensuales, 1):
        van += flujo / (1 + tasa_mensual) ** t
    return van


def calcular_tir(flujos_mensuales, inversion_inicial, max_iter=500, tol=1e-6):
    """TIR mensual → anualizada. Newton-Raphson sobre flujos mensuales."""
    # flujos = [-inversión, f1, f2, ..., f12]
    cashflows = np.array([-inversion_inicial] + list(flujos_mensuales))
    # Semilla inicial
    r = 0.02
    for _ in range(max_iter):
        npv = sum(cf / (1 + r) ** t for t, cf in enumerate(cashflows))
        dnpv = sum(-t * cf / (1 + r) ** (t + 1) for t, cf in enumerate(cashflows))
        if abs(dnpv) < 1e-14:
            break
        r_new = r - npv / dnpv
        if abs(r_new - r) < tol:
            r = r_new
            break
        r = r_new
    tir_anual = (1 + r) ** 12 - 1
    return tir_anual


def distribuir_recuperacion_mensual(ve_total, perfil='decreciente', meses=12):
    """
    Distribuye la recuperación esperada total en flujos mensuales según un perfil.
    - 'uniforme': mismo monto cada mes.
    - 'decreciente': mayor peso en los primeros meses (70-30 primera/segunda mitad).
    - 'concentrado_inicio': 50% en meses 1-3, resto distribuido.
    """
    if perfil == 'uniforme':
        return np.full(meses, ve_total / meses)
    elif perfil == 'decreciente':
        pesos = np.array([12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1], dtype=float)[:meses]
        return ve_total * (pesos / pesos.sum())
    elif perfil == 'concentrado_inicio':
        pesos = np.array([15, 13, 11, 9, 8, 7, 7, 7, 7, 6, 5, 5], dtype=float)[:meses]
        return ve_total * (pesos / pesos.sum())
    return np.full(meses, ve_total / meses)


def sensibilidad_tir(ve_total, precio_compra, perfil, escenarios_pct):
    """Calcula TIR para distintos porcentajes de cumplimiento de la recuperación."""
    resultados = []
    for pct in escenarios_pct:
        ve_ajustado = ve_total * pct
        flujos = distribuir_recuperacion_mensual(ve_ajustado, perfil)
        try:
            tir = calcular_tir(flujos, precio_compra)
            van_10 = calcular_van(flujos, 0.10, precio_compra)
        except Exception:
            tir = np.nan
            van_10 = np.nan
        resultados.append({
            'Escenario': f'{pct:.0%}',
            'Recuperación Ajustada': ve_ajustado,
            'TIR Anual': tir,
            'VAN (10%)': van_10,
            'ROI': (ve_ajustado - precio_compra) / precio_compra if precio_compra > 0 else 0,
            'Múltiplo': ve_ajustado / precio_compra if precio_compra > 0 else 0,
        })
    return pd.DataFrame(resultados)


# ============================================================
# HEADER
# ============================================================
st.title("Dashboard de Valuación de Cartera NPL")
st.caption("Modelo Híbrido de Dos Etapas · Gradient Boosting · Grupo 4 — ML Supervisado UDB")

# ============================================================
# SIDEBAR — CARGA DE DATOS
# ============================================================
st.sidebar.header("Carga del Lote")
file_saldo = st.sidebar.file_uploader("Saldo CSV", type=['csv'], key='saldo')
file_detalles = st.sidebar.file_uploader("Detalles CSV", type=['csv'], key='detalles')

st.sidebar.markdown("---")
st.sidebar.header("Parámetros de Compra")
margen_pct = st.sidebar.slider("Margen de seguridad (%)", 5, 50, 20, 1)
margen = margen_pct / 100
precio_por_dolar = st.sidebar.number_input(
    "Precio de compra por $1 de saldo", min_value=0.01, max_value=0.50,
    value=0.05, step=0.01, format="%.2f",
    help="Dólares pagados por cada dólar de saldo facial del lote (ej: 0.05 = 5 centavos)"
)
perfil_cobranza = st.sidebar.selectbox(
    "Perfil de cobranza", ['decreciente', 'uniforme', 'concentrado_inicio'],
    help="Cómo se distribuye la recuperación a lo largo de 12 meses"
)
costo_operativo_pct = st.sidebar.slider(
    "Costo operativo de gestión (%)", 0, 30, 12, 1,
    help="Porcentaje del valor recuperado destinado a costos de cobranza"
)

st.sidebar.markdown("---")
st.sidebar.header("Modelo")
st.sidebar.markdown(f"""
- **Clasificador:** {model_config['mejor_clasificador']['nombre']}
  - AUC Test: {model_config['mejor_clasificador']['resultados_cv']['AUC_Test']:.4f}
- **Regresor:** {model_config['mejor_regresor']['nombre']}
  - R² Test: {model_config['mejor_regresor']['resultados_cv']['R2_Test']:.4f}
  - MAE: ${model_config['mejor_regresor']['resultados_cv']['MAE_Test']:,.0f}
""")

# ============================================================
# PROCESAMIENTO PRINCIPAL
# ============================================================
if file_saldo is None or file_detalles is None:
    st.info("Sube los archivos **Saldo.csv** y **Detalles.csv** del lote a evaluar en la barra lateral para comenzar.")
    st.markdown("""
    ### ¿Cómo funciona?
    1. **Sube ambos CSVs** (Saldo y Detalles) del lote que deseas evaluar.
    2. El modelo predice la **probabilidad de pago** y el **monto recuperable** de cada cuenta.
    3. Se genera un **dashboard interactivo** con métricas de retorno: ROI, VAN, TIR.
    4. Puedes ajustar el **precio de compra**, **márgenes** y **escenarios** para tomar la decisión.
    """)
    st.stop()

# --- Procesar datos ---
df_saldo_raw = pd.read_csv(file_saldo, encoding='latin1')
df_detalles_raw = pd.read_csv(file_detalles, encoding='latin1')

with st.spinner(f"Procesando {len(df_saldo_raw):,} cuentas..."):
    X, df_orig = preprocesar_lote(df_saldo_raw, df_detalles_raw)

    # Predicciones
    prob_pago = clf.predict_proba(X)[:, 1]
    monto_pred = np.maximum(reg.predict(X), 0)
    valor_esperado = prob_pago * monto_pred

# --- Construir DataFrame de resultados ---
df_res = df_orig.copy()
df_res['Prob_Pago'] = prob_pago
df_res['Monto_Estimado'] = monto_pred
df_res['Valor_Esperado'] = valor_esperado
saldo_col = 'SALDO TOTAL' if 'SALDO TOTAL' in df_res.columns else None
if saldo_col:
    df_res[saldo_col] = pd.to_numeric(df_res[saldo_col], errors='coerce').fillna(0)
    df_res['ROI_Cuenta'] = np.where(
        df_res[saldo_col] > 0,
        df_res['Valor_Esperado'] / df_res[saldo_col],
        0
    )
else:
    df_res['ROI_Cuenta'] = 0

# --- Segmentar por riesgo ---
def clasificar_riesgo(row):
    if row['Prob_Pago'] >= 0.4:
        return 'Alto Potencial'
    elif row['Prob_Pago'] >= 0.2:
        return 'Moderado'
    else:
        return 'Bajo Potencial'

df_res['Segmento'] = df_res.apply(clasificar_riesgo, axis=1)

# --- Cálculos financieros globales ---
ve_total = df_res['Valor_Esperado'].sum()
saldo_total = df_res[saldo_col].sum() if saldo_col else ve_total
precio_compra = saldo_total * precio_por_dolar
costo_operativo = ve_total * (costo_operativo_pct / 100)
recuperacion_neta = ve_total - costo_operativo
utilidad = recuperacion_neta - precio_compra
roi_global = utilidad / precio_compra if precio_compra > 0 else 0

flujos = distribuir_recuperacion_mensual(recuperacion_neta, perfil_cobranza)
try:
    tir_anual = calcular_tir(flujos, precio_compra)
except Exception:
    tir_anual = np.nan
van_10 = calcular_van(flujos, 0.10, precio_compra)
multiplo = recuperacion_neta / precio_compra if precio_compra > 0 else 0

n_cuentas = len(df_res)
n_pagadores = (df_res['Prob_Pago'] >= 0.5).sum()
tasa_pago_est = n_pagadores / n_cuentas if n_cuentas > 0 else 0

# ==============================================================
# DASHBOARD
# ==============================================================
st.success(f"Lote procesado: **{n_cuentas:,}** cuentas evaluadas")

# --- KPIs PRINCIPALES ---
st.header("Resumen Ejecutivo del Lote")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Cuentas", f"{n_cuentas:,}")
k2.metric("Saldo Facial", f"${saldo_total:,.0f}")
k3.metric("Valor Esperado", f"${ve_total:,.0f}")
k4.metric("Precio Compra", f"${precio_compra:,.0f}")

k5, k6, k7 = st.columns(3)
k5.metric("Utilidad Neta", f"${utilidad:,.0f}", delta=f"{roi_global:+.1%}")
k6.metric("TIR Anual", f"{tir_anual:.1%}" if not np.isnan(tir_anual) else "N/D")
k7.metric("VAN (tasa 10%)", f"${van_10:,.0f}")

k8, k9, k10 = st.columns(3)
k8.metric("Múltiplo (x)", f"{multiplo:.2f}x")
k9.metric("Pagadores Est. (>50%)", f"{n_pagadores:,} ({tasa_pago_est:.1%})")
k10.metric("Costo Operativo", f"${costo_operativo:,.0f}")

# ==============================================================
# TABS DEL DASHBOARD
# ==============================================================
tab_decision, tab_dist, tab_segmentos, tab_sensibilidad, tab_detalle = st.tabs([
    "Decisión de Compra",
    "Distribución",
    "Segmentación",
    "Sensibilidad & TIR",
    "Detalle de Cuentas"
])

# ==============================================================
# TAB 1: DECISIÓN DE COMPRA
# ==============================================================
with tab_decision:
    st.header("Análisis de Decisión de Compra")

    # --- Semáforo de decisión ---
    col_sem, col_gauge = st.columns([1, 1])

    with col_sem:
        st.subheader("Indicadores Clave")
        checks = []

        if roi_global > 0.30:
            checks.append(("[+]", "ROI > 30%", f"{roi_global:.1%}"))
        elif roi_global > 0.10:
            checks.append(("[~]", "ROI entre 10-30%", f"{roi_global:.1%}"))
        else:
            checks.append(("[-]", "ROI < 10%", f"{roi_global:.1%}"))

        if not np.isnan(tir_anual) and tir_anual > 0.25:
            checks.append(("[+]", "TIR > 25%", f"{tir_anual:.1%}"))
        elif not np.isnan(tir_anual) and tir_anual > 0.10:
            checks.append(("[~]", "TIR entre 10-25%", f"{tir_anual:.1%}"))
        else:
            checks.append(("[-]", "TIR < 10%", f"{tir_anual:.1%}" if not np.isnan(tir_anual) else "N/D"))

        if van_10 > 0:
            checks.append(("[+]", "VAN positivo (tasa 10%)", f"${van_10:,.0f}"))
        else:
            checks.append(("[-]", "VAN negativo (tasa 10%)", f"${van_10:,.0f}"))

        if multiplo > 1.5:
            checks.append(("[+]", "Múltiplo > 1.5x", f"{multiplo:.2f}x"))
        elif multiplo > 1.0:
            checks.append(("[~]", "Múltiplo entre 1.0-1.5x", f"{multiplo:.2f}x"))
        else:
            checks.append(("[-]", "Múltiplo < 1.0x", f"{multiplo:.2f}x"))

        positivos = sum(1 for c in checks if c[0] == "[+]")

        for icon, label, val in checks:
            st.markdown(f"{icon} **{label}:** `{val}`")

        st.markdown("---")
        if positivos >= 3:
            st.success("### RECOMENDACIÓN: COMPRAR")
            st.markdown("El lote presenta indicadores financieros favorables.")
        elif positivos >= 2:
            st.warning("### RECOMENDACIÓN: EVALUAR CON CUIDADO")
            st.markdown("Indicadores mixtos. Considerar ajustar precio o margen.")
        else:
            st.error("### RECOMENDACIÓN: NO COMPRAR / RENEGOCIAR")
            st.markdown("Los indicadores sugieren riesgo alto. Renegociar el precio.")

    with col_gauge:
        # Gauge de ROI
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=roi_global * 100,
            title={'text': "ROI Esperado (%)"},
            delta={'reference': 20, 'suffix': '%'},
            gauge={
                'axis': {'range': [-50, 200], 'ticksuffix': '%'},
                'bar': {'color': '#2ecc71' if roi_global > 0.2 else '#f39c12' if roi_global > 0 else '#e74c3c'},
                'steps': [
                    {'range': [-50, 0], 'color': '#fadbd8'},
                    {'range': [0, 20], 'color': '#fef9e7'},
                    {'range': [20, 200], 'color': '#eafaf1'},
                ],
                'threshold': {
                    'line': {'color': '#2c3e50', 'width': 3},
                    'thickness': 0.8,
                    'value': 20
                }
            }
        ))
        fig_gauge.update_layout(height=300, margin=dict(t=40, b=10, l=30, r=30))
        st.plotly_chart(fig_gauge, use_container_width=True)

    # --- Cascada de valor ---
    st.subheader("Cascada de Valor")
    fig_waterfall = go.Figure(go.Waterfall(
        name="Flujo",
        orientation="v",
        measure=["absolute", "relative", "relative", "total"],
        x=["Recuperación Bruta", "− Costo Operativo", "− Precio Compra", "Utilidad Neta"],
        y=[ve_total, -costo_operativo, -precio_compra, 0],
        connector={"line": {"color": "#7f8c8d"}},
        increasing={"marker": {"color": "#2ecc71"}},
        decreasing={"marker": {"color": "#e74c3c"}},
        totals={"marker": {"color": "#3498db" if utilidad > 0 else "#e74c3c"}},
        text=[f"${ve_total:,.0f}", f"-${costo_operativo:,.0f}",
              f"-${precio_compra:,.0f}", f"${utilidad:,.0f}"],
        textposition="outside"
    ))
    fig_waterfall.update_layout(
        title="Descomposición del Retorno",
        height=380, margin=dict(t=50, b=30),
        yaxis_title="USD ($)"
    )
    st.plotly_chart(fig_waterfall, use_container_width=True)

    # --- Flujos mensuales ---
    st.subheader("Flujos de Recuperación Mensual Proyectados")
    meses_lbl = [f"Mes {i}" for i in range(1, 13)]
    flujos_brutos = distribuir_recuperacion_mensual(ve_total, perfil_cobranza)
    flujos_netos = flujos_brutos * (1 - costo_operativo_pct / 100)
    flujos_acum = np.cumsum(flujos_netos)

    fig_flujos = make_subplots(specs=[[{"secondary_y": True}]])
    fig_flujos.add_trace(
        go.Bar(x=meses_lbl, y=flujos_netos, name="Flujo Neto Mensual",
               marker_color='#3498db', opacity=0.8),
        secondary_y=False
    )
    fig_flujos.add_trace(
        go.Scatter(x=meses_lbl, y=flujos_acum, name="Acumulado",
                   line=dict(color='#2ecc71', width=3), mode='lines+markers'),
        secondary_y=True
    )
    fig_flujos.add_hline(y=precio_compra, line_dash="dash", line_color="#e74c3c",
                         annotation_text=f"Inversión: ${precio_compra:,.0f}",
                         annotation_position="bottom left",
                         annotation_font_color="#e74c3c",
                         secondary_y=True)
    # Punto de equilibrio
    meses_payback = int(np.searchsorted(flujos_acum, precio_compra)) + 1
    if meses_payback <= 12:
        fig_flujos.add_shape(
            type="line", x0=meses_payback - 1, x1=meses_payback - 1,
            y0=0, y1=1, yref="paper",
            line=dict(dash="dot", color="#f39c12", width=2)
        )
        fig_flujos.add_annotation(
            x=meses_payback - 1, y=1, yref="paper",
            text=f"Break-even: Mes {meses_payback}",
            showarrow=False, yshift=10,
            font=dict(color="#f39c12", size=11)
        )

    fig_flujos.update_layout(
        title=f"Proyección de Flujos ({perfil_cobranza.title()})",
        height=400, margin=dict(t=50, b=30),
        legend=dict(orientation="h", y=1.12)
    )
    fig_flujos.update_yaxes(title_text="Flujo Mensual ($)", secondary_y=False)
    fig_flujos.update_yaxes(title_text="Acumulado ($)", secondary_y=True)
    st.plotly_chart(fig_flujos, use_container_width=True)

    if meses_payback <= 12:
        st.info(f"**Payback estimado:** Mes {meses_payback} de 12")
    else:
        st.warning("El payback no se alcanza dentro de los 12 meses con estos parámetros.")


# ==============================================================
# TAB 2: DISTRIBUCIÓN
# ==============================================================
with tab_dist:
    st.header("Distribución de Predicciones")

    col_d1, col_d2 = st.columns(2)

    with col_d1:
        fig_prob = px.histogram(
            df_res, x='Prob_Pago', nbins=50,
            title="Distribución de Probabilidad de Pago",
            labels={'Prob_Pago': 'Probabilidad de Pago'},
            color_discrete_sequence=['#3498db'],
            opacity=0.8
        )
        fig_prob.add_vline(x=0.5, line_dash='dash', line_color='red',
                           annotation_text='Umbral 50%')
        fig_prob.update_layout(height=350, margin=dict(t=50, b=30))
        st.plotly_chart(fig_prob, use_container_width=True)

    with col_d2:
        fig_ve = px.histogram(
            df_res[df_res['Valor_Esperado'] > 0], x='Valor_Esperado', nbins=50,
            title="Distribución de Valor Esperado (VE > 0)",
            labels={'Valor_Esperado': 'Valor Esperado ($)'},
            color_discrete_sequence=['#2ecc71'],
            opacity=0.8
        )
        fig_ve.update_layout(height=350, margin=dict(t=50, b=30))
        st.plotly_chart(fig_ve, use_container_width=True)

    col_d3, col_d4 = st.columns(2)

    with col_d3:
        if 'DIAS MORA' in df_res.columns:
            fig_mora = px.scatter(
                df_res, x='DIAS MORA', y='Valor_Esperado',
                color='Prob_Pago', size='Monto_Estimado',
                title="Mapa de Riesgo: Mora vs Valor Esperado",
                labels={'DIAS MORA': 'Días de Mora', 'Valor_Esperado': 'Valor Esperado ($)'},
                color_continuous_scale='RdYlGn',
                opacity=0.6, size_max=15
            )
            fig_mora.update_layout(height=400, margin=dict(t=50, b=30))
            st.plotly_chart(fig_mora, use_container_width=True)

    with col_d4:
        if saldo_col:
            fig_saldo = px.scatter(
                df_res, x=saldo_col, y='Valor_Esperado',
                color='Segmento',
                title="Saldo vs Valor Esperado por Segmento",
                labels={saldo_col: 'Saldo Total ($)', 'Valor_Esperado': 'Valor Esperado ($)'},
                color_discrete_map={
                    'Alto Potencial': '#2ecc71',
                    'Moderado': '#f39c12',
                    'Bajo Potencial': '#e74c3c'
                },
                opacity=0.6
            )
            fig_saldo.update_layout(height=400, margin=dict(t=50, b=30))
            st.plotly_chart(fig_saldo, use_container_width=True)

    # --- Box plots ---
    st.subheader("Distribución por Segmento")
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        fig_box_prob = px.box(
            df_res, x='Segmento', y='Prob_Pago',
            color='Segmento', title="Probabilidad de Pago por Segmento",
            color_discrete_map={
                'Alto Potencial': '#2ecc71', 'Moderado': '#f39c12', 'Bajo Potencial': '#e74c3c'
            }
        )
        fig_box_prob.update_layout(height=350, margin=dict(t=50, b=30), showlegend=False)
        st.plotly_chart(fig_box_prob, use_container_width=True)

    with col_b2:
        fig_box_ve = px.box(
            df_res, x='Segmento', y='Valor_Esperado',
            color='Segmento', title="Valor Esperado por Segmento",
            color_discrete_map={
                'Alto Potencial': '#2ecc71', 'Moderado': '#f39c12', 'Bajo Potencial': '#e74c3c'
            }
        )
        fig_box_ve.update_layout(height=350, margin=dict(t=50, b=30), showlegend=False)
        st.plotly_chart(fig_box_ve, use_container_width=True)


# ==============================================================
# TAB 3: SEGMENTACIÓN
# ==============================================================
with tab_segmentos:
    st.header("Segmentación de la Cartera")

    seg_stats = df_res.groupby('Segmento').agg(
        Cuentas=('Valor_Esperado', 'count'),
        VE_Total=('Valor_Esperado', 'sum'),
        VE_Promedio=('Valor_Esperado', 'mean'),
        VE_Mediana=('Valor_Esperado', 'median'),
        Prob_Pago_Promedio=('Prob_Pago', 'mean'),
        Saldo_Total=(saldo_col, 'sum') if saldo_col else ('Valor_Esperado', 'sum'),
    ).reset_index()
    seg_stats['% del VE Total'] = seg_stats['VE_Total'] / ve_total * 100
    seg_stats['% de Cuentas'] = seg_stats['Cuentas'] / n_cuentas * 100

    col_s1, col_s2 = st.columns(2)

    with col_s1:
        fig_pie = px.pie(
            seg_stats, values='VE_Total', names='Segmento',
            title="Distribución del Valor Esperado por Segmento",
            color='Segmento',
            color_discrete_map={
                'Alto Potencial': '#2ecc71', 'Moderado': '#f39c12', 'Bajo Potencial': '#e74c3c'
            },
            hole=0.4
        )
        fig_pie.update_layout(height=350, margin=dict(t=50, b=30))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_s2:
        fig_pie2 = px.pie(
            seg_stats, values='Cuentas', names='Segmento',
            title="Distribución de Cuentas por Segmento",
            color='Segmento',
            color_discrete_map={
                'Alto Potencial': '#2ecc71', 'Moderado': '#f39c12', 'Bajo Potencial': '#e74c3c'
            },
            hole=0.4
        )
        fig_pie2.update_layout(height=350, margin=dict(t=50, b=30))
        st.plotly_chart(fig_pie2, use_container_width=True)

    st.dataframe(
        seg_stats.style.format({
            'VE_Total': '${:,.0f}',
            'VE_Promedio': '${:,.2f}',
            'VE_Mediana': '${:,.2f}',
            'Prob_Pago_Promedio': '{:.2%}',
            'Saldo_Total': '${:,.0f}',
            '% del VE Total': '{:.1f}%',
            '% de Cuentas': '{:.1f}%',
        }),
        use_container_width=True, hide_index=True
    )

    # --- Análisis por variable categórica ---
    st.subheader("Análisis por Características")
    cat_col = None
    if 'SEXO' in df_orig.columns:
        cat_col = 'SEXO'
    available_cats = [c for c in ['SEXO', 'EST_CIVIL_CLEAN', 'PRODUCTO', 'Lote', 'RANGO MORA']
                      if c in df_res.columns]
    if available_cats:
        selected_cat = st.selectbox("Agrupar por:", available_cats)
        cat_analysis = df_res.groupby(selected_cat).agg(
            Cuentas=('Valor_Esperado', 'count'),
            VE_Total=('Valor_Esperado', 'sum'),
            VE_Promedio=('Valor_Esperado', 'mean'),
            Prob_Pago_Promedio=('Prob_Pago', 'mean'),
        ).reset_index().sort_values('VE_Total', ascending=False)

        fig_cat = px.bar(
            cat_analysis, x=selected_cat, y='VE_Total',
            color='Prob_Pago_Promedio',
            title=f"Valor Esperado por {selected_cat}",
            color_continuous_scale='RdYlGn',
            text_auto='$.2s'
        )
        fig_cat.update_layout(height=400, margin=dict(t=50, b=30))
        st.plotly_chart(fig_cat, use_container_width=True)


# ==============================================================
# TAB 4: SENSIBILIDAD & TIR
# ==============================================================
with tab_sensibilidad:
    st.header("Análisis de Sensibilidad y TIR")

    st.markdown("""
    Simulación de escenarios variando el **porcentaje de cumplimiento** de la
    recuperación estimada y el **precio de compra** para evaluar la robustez del retorno.
    """)

    col_t1, col_t2 = st.columns([2, 1])

    with col_t2:
        st.subheader("Parámetros de Simulación")
        rango_cumplimiento = st.slider(
            "Rango de cumplimiento del VE (%)",
            min_value=10, max_value=150, value=(50, 120), step=5
        )
        n_escenarios = st.slider("Número de escenarios", 5, 20, 10)
        tasa_descuento_ref = st.number_input(
            "Tasa de descuento de referencia (%)", 5.0, 50.0, 10.0, 1.0
        ) / 100

    escenarios_pct = np.linspace(rango_cumplimiento[0] / 100, rango_cumplimiento[1] / 100, n_escenarios)
    ve_neto = ve_total * (1 - costo_operativo_pct / 100)
    df_sens = sensibilidad_tir(ve_neto, precio_compra, perfil_cobranza, escenarios_pct)

    with col_t1:
        # Gráfico TIR vs Cumplimiento
        fig_tir = make_subplots(specs=[[{"secondary_y": True}]])
        fig_tir.add_trace(
            go.Scatter(
                x=df_sens['Escenario'], y=df_sens['TIR Anual'] * 100,
                name='TIR Anual (%)', mode='lines+markers',
                line=dict(color='#3498db', width=3),
                marker=dict(size=8)
            ), secondary_y=False
        )
        fig_tir.add_trace(
            go.Bar(
                x=df_sens['Escenario'], y=df_sens['VAN (10%)'],
                name='VAN ($, tasa 10%)',
                marker_color=df_sens['VAN (10%)'].apply(
                    lambda v: '#2ecc71' if v > 0 else '#e74c3c'),
                opacity=0.5
            ), secondary_y=True
        )
        fig_tir.add_hline(y=0, line_dash='dash', line_color='gray', secondary_y=False)
        fig_tir.add_hline(y=tasa_descuento_ref * 100, line_dash='dot', line_color='#e74c3c',
                          annotation_text=f"Tasa referencia: {tasa_descuento_ref:.0%}",
                          secondary_y=False)
        fig_tir.update_layout(
            title="TIR y VAN por Escenario de Cumplimiento",
            height=420, margin=dict(t=50, b=30),
            legend=dict(orientation="h", y=1.12)
        )
        fig_tir.update_yaxes(title_text="TIR (%)", secondary_y=False)
        fig_tir.update_yaxes(title_text="VAN ($)", secondary_y=True)
        st.plotly_chart(fig_tir, use_container_width=True)

    # Tabla de sensibilidad
    st.subheader("Tabla de Escenarios")
    df_sens_display = df_sens.copy()
    st.dataframe(
        df_sens_display.style.format({
            'Recuperación Ajustada': '${:,.0f}',
            'TIR Anual': '{:.1%}',
            'VAN (10%)': '${:,.0f}',
            'ROI': '{:.1%}',
            'Múltiplo': '{:.2f}x',
        }).applymap(
            lambda v: 'background-color: #eafaf1' if isinstance(v, (int, float)) and v > 0
            else 'background-color: #fadbd8' if isinstance(v, (int, float)) and v < 0
            else '', subset=['VAN (10%)']
        ),
        use_container_width=True, hide_index=True
    )

    # --- Sensibilidad bidimensional: Precio vs Cumplimiento ---
    st.subheader("Mapa de Calor: TIR por Precio y Cumplimiento")

    precios_dolar = np.arange(0.01, 0.16, 0.01)
    cumpls = np.linspace(0.4, 1.2, 9)
    heatmap_data = []
    for p in precios_dolar:
        row = []
        for c in cumpls:
            pc = saldo_total * p
            ve_adj = ve_neto * c
            fl = distribuir_recuperacion_mensual(ve_adj, perfil_cobranza)
            try:
                t = calcular_tir(fl, pc)
                row.append(t * 100)
            except Exception:
                row.append(np.nan)
        heatmap_data.append(row)

    fig_heat = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=[f"{c:.0%}" for c in cumpls],
        y=[f"${p:.2f}" for p in precios_dolar],
        colorscale=[
            [0, '#e74c3c'], [0.3, '#f39c12'], [0.5, '#f1c40f'],
            [0.7, '#2ecc71'], [1.0, '#27ae60']
        ],
        text=[[f"{v:.0f}%" if not np.isnan(v) else "" for v in row] for row in heatmap_data],
        texttemplate="%{text}",
        colorbar_title="TIR (%)",
        zmid=0,
    ))
    fig_heat.update_layout(
        title="TIR Anual (%) por Precio de Compra vs Cumplimiento de Recuperación",
        xaxis_title="Cumplimiento del VE",
        yaxis_title="Precio por $1 de saldo",
        height=450, margin=dict(t=50, b=30)
    )
    # Marcador del escenario actual
    fig_heat.add_annotation(
        x=f"{1.0:.0%}",
        y=f"${precio_por_dolar:.2f}",
        text="Actual",
        showarrow=True, arrowhead=2, arrowcolor='white',
        font=dict(color='white', size=12, family='Arial Black'),
        bordercolor='white', borderwidth=2, bgcolor='rgba(0,0,0,0.7)',
        borderpad=4
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # --- Precio máximo para TIR objetivo ---
    st.subheader("Precio Máximo según TIR Objetivo")
    tir_objetivo = st.slider("TIR Objetivo Anual (%)", 10, 100, 25, 5) / 100

    # Búsqueda binaria del precio máximo
    lo, hi = 0.001 * saldo_total, saldo_total * 0.5
    for _ in range(60):
        mid = (lo + hi) / 2
        fl = distribuir_recuperacion_mensual(ve_neto, perfil_cobranza)
        try:
            t = calcular_tir(fl, mid)
        except Exception:
            t = -1
        if t > tir_objetivo:
            lo = mid
        else:
            hi = mid
    precio_max = lo
    precio_por_dolar_max = precio_max / saldo_total if saldo_total > 0 else 0

    colp1, colp2, colp3 = st.columns(3)
    colp1.metric("Precio Máximo ($)", f"${precio_max:,.0f}")
    colp2.metric("Precio por $1 de saldo", f"${precio_por_dolar_max:.2f}")
    colp3.metric("TIR al precio máximo", f"{tir_objetivo:.0%}")

    st.info(f"Para alcanzar una TIR del **{tir_objetivo:.0%}** con recuperación al 100%, "
            f"el precio máximo de compra es **${precio_max:,.0f}** "
            f"(**${precio_por_dolar_max:.2f}** por cada $1 de saldo facial).")


# ==============================================================
# TAB 5: DETALLE DE CUENTAS
# ==============================================================
with tab_detalle:
    st.header("Detalle de Cuentas")

    # Filtros
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        seg_filter = st.multiselect("Segmento", df_res['Segmento'].unique().tolist(),
                                     default=df_res['Segmento'].unique().tolist())
    with col_f2:
        prob_range = st.slider("Rango de Probabilidad", 0.0, 1.0, (0.0, 1.0), 0.05)
    with col_f3:
        top_n = st.number_input("Top N cuentas", 10, len(df_res), min(50, len(df_res)), 10)

    mask = (
        df_res['Segmento'].isin(seg_filter) &
        df_res['Prob_Pago'].between(prob_range[0], prob_range[1])
    )
    df_filtered = df_res[mask].sort_values('Valor_Esperado', ascending=False).head(top_n)

    cols_to_show = ['CUENTA', saldo_col, 'DIAS MORA', 'Prob_Pago', 'Monto_Estimado',
                    'Valor_Esperado', 'ROI_Cuenta', 'Segmento']
    cols_to_show = [c for c in cols_to_show if c and c in df_filtered.columns]

    st.dataframe(
        df_filtered[cols_to_show].style.format({
            'Prob_Pago': '{:.2%}',
            'Monto_Estimado': '${:,.2f}',
            'Valor_Esperado': '${:,.2f}',
            'ROI_Cuenta': '{:.1%}',
            saldo_col: '${:,.2f}' if saldo_col else '{}',
        }).background_gradient(subset=['Valor_Esperado'], cmap='Greens'),
        use_container_width=True, hide_index=True,
        height=500
    )

    # Resumen de la selección
    st.markdown("---")
    cs1, cs2, cs3, cs4 = st.columns(4)
    cs1.metric("Cuentas mostradas", f"{len(df_filtered):,}")
    cs2.metric("VE Selección", f"${df_filtered['Valor_Esperado'].sum():,.0f}")
    cs3.metric("% del VE Total", f"{df_filtered['Valor_Esperado'].sum() / ve_total * 100:.1f}%")
    cs4.metric("Prob. Promedio", f"{df_filtered['Prob_Pago'].mean():.2%}")

    # Descargar
    csv_out = df_filtered[cols_to_show].to_csv(index=False)
    st.download_button(
        "Descargar selección (CSV)", csv_out,
        "cuentas_seleccionadas.csv", "text/csv",
        use_container_width=True
    )

# ==============================================================
# FOOTER
# ==============================================================
st.markdown("---")
st.caption("Valuador de Cartera NPL — Grupo 4 · ML Supervisado · UDB Virtual · 2026")
