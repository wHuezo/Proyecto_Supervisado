# Valuador de Carteras de Crédito (NPL) — Grupo 4

## Descripción del Problema

En la industria de adquisición de deuda, valorar incorrectamente una cartera de créditos vencidos (NPLs) puede resultar en pérdidas millonarias. Actualmente, la valuación se realiza mediante promedios históricos simples, ignorando la calidad individual de los deudores.

## Solución Propuesta

Desarrollamos un **Modelo Híbrido de Dos Etapas (Hurdle Model)** que predice el Valor Esperado de Recuperación para cada cuenta individual:

- **Etapa 1 — Clasificador:** Random Forest estima la *Probabilidad de Pago*.
- **Etapa 2 — Regresor:** Random Forest estima el *Monto Recuperable* condicionado al pago.
- **Predicción Final:** `Valor_Esperado = P(pago) × Monto_Estimado`

## Stack Tecnológico

| Componente | Tecnología |
|------------|------------|
| Lenguaje | Python 3.10+ |
| Ingeniería de Datos | Pandas, NumPy |
| Modelado | Scikit-Learn (Random Forest, Gradient Boosting, Ridge, Logistic Regression) |
| Optimización | RandomizedSearchCV con K-Fold Cross-Validation |
| Explicabilidad | SHAP |
| Validación Estadística | SciPy (Mann-Whitney U) |
| Visualización | Matplotlib, Seaborn |
| Despliegue | Streamlit (WebApp interactiva) |

## Estructura del Proyecto

```
PROYECTO/
├── app/
│   └── app.py                  # Interfaz Streamlit (evaluación individual + batch)
├── data/
│   ├── raw/                    # Datos crudos (4 archivos CSV)
│   │   ├── Saldo.csv
│   │   ├── Detallesanonimo.csv
│   │   ├── Pagos.csv
│   │   └── adquisicion.csv
│   └── processed/
│       └── dataset_master_analitico.csv
├── models/                     # Artefactos serializados
│   ├── clasificador_pago.pkl
│   ├── regresor_monto.pkl
│   ├── columnas_modelo.pkl
│   └── config_mejores_params.json
├── notebooks/
│   ├── 1_Ingenieria_y_Limpieza.ipynb   # ETL y Feature Engineering
│   ├── 2_EDA_Insights.ipynb            # Análisis Exploratorio completo
│   ├── 3_Seleccion_Modelos.ipynb       # Comparación, CV, optimización
│   └── 4_Entrenamiento_Final.ipynb     # SHAP, residuos, métricas de negocio
├── requirements.txt
└── README.md
```

## Cómo Ejecutar el Proyecto

### 1. Instalación de Dependencias

```bash
pip install -r requirements.txt
```

### 2. Ejecución de Notebooks (en orden)

Ejecutar los notebooks en secuencia desde la carpeta `notebooks/`:

1. **`1_Ingenieria_y_Limpieza.ipynb`** — Genera `dataset_master_analitico.csv`
2. **`2_EDA_Insights.ipynb`** — Análisis exploratorio y visualizaciones
3. **`3_Seleccion_Modelos.ipynb`** — Comparación de algoritmos y optimización de hiperparámetros
4. **`4_Entrenamiento_Final.ipynb`** — Entrenamiento final, SHAP, métricas de negocio. Genera los archivos `.pkl`

### 3. Ejecución de la Interfaz Web

```bash
cd app
streamlit run app.py
```

La aplicación permite:
- **Evaluación individual:** Ingresar datos de un cliente y obtener la valuación.
- **Evaluación masiva:** Subir archivos CSV de un lote completo y obtener la valuación agregada con tabla descargable.

## Metodología

1. **EDA:** Análisis univariado y multivariado, pruebas de hipótesis (Mann-Whitney U), PCA.
2. **Preprocesamiento:** Limpieza monetaria, normalización de llaves, encoding categórico, tratamiento de outliers (clipping, log-transform), ingeniería de features (score de contactabilidad, ratios financieros).
3. **Modelado:** Comparación sistemática de 4 familias de algoritmos (Baseline, Lineal, Árboles, Boosting) con validación cruzada 5-Fold.
4. **Optimización:** RandomizedSearchCV sobre Random Forest y Gradient Boosting (40 iteraciones, 5 folds).
5. **Evaluación:** Métricas técnicas (AUC, MAE, R², RMSE) + métricas de negocio (desviación del lote, precio sugerido de compra).
6. **Interpretabilidad:** SHAP summary plots + feature importance.

## Equipo

**Grupo 4** — Machine Learning Supervisado, UDB Virtual
