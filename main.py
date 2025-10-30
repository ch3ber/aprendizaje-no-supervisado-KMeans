import io
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import streamlit as st

# Configuración de la página de Streamlit
st.set_page_config(page_title="Segmentación de Clientes (K-Means)", layout="wide")

st.title("Segmentación de clientes con K-Means")
st.write("Sube un CSV con dos columnas numéricas (cualesquiera).")

# Carga de archivo CSV
archivo = st.file_uploader("Cargar archivo CSV", type=["csv"])

if archivo is not None:
    try:
        df_raw = pd.read_csv(archivo)
    except Exception as e:
        st.error(f"No se pudo leer el CSV: {e}")
        st.stop()

    st.subheader("Datos originales")
    st.dataframe(df_raw)

    if df_raw.shape[1] < 2:
        st.error("El CSV debe tener al menos 2 columnas.")
        st.stop()

    # Sugerir columnas: si hay 2 columnas, tomarlas; si no, primeras 2 numéricas
    cols_all = list(df_raw.columns)
    numeric_candidates = list(df_raw.select_dtypes(include=["number"]).columns)
    if len(df_raw.columns) == 2:
        default_cols = cols_all
    elif len(numeric_candidates) >= 2:
        default_cols = numeric_candidates[:2]
    else:
        default_cols = cols_all[:2]

    sel_cols = st.sidebar.multiselect(
        "Selecciona 2 columnas para K-Means",
        options=cols_all,
        default=default_cols,
        max_selections=2,
        help="Elige exactamente dos columnas (se convertirán a numéricas).",
    )

    if len(sel_cols) != 2:
        st.info("Selecciona exactamente 2 columnas para continuar.")
        st.stop()

    # Convertir a numérico y limpiar filas no válidas
    df_two = df_raw[sel_cols].apply(pd.to_numeric, errors="coerce")
    filas_invalidas = df_two.isna().any(axis=1)
    invalid_count = int(filas_invalidas.sum())
    if invalid_count > 0:
        st.warning(f"Se ignoraron {invalid_count} filas con valores no numéricos o vacíos en las columnas seleccionadas.")
    df_two_clean = df_two.dropna()

    if len(df_two_clean) < 2:
        st.error("No hay suficientes filas válidas (>=2) tras limpiar datos no numéricos.")
        st.stop()

    # Normalización Min-Max en columnas seleccionadas
    escalador = MinMaxScaler().fit(df_two_clean.values)
    df_norm = pd.DataFrame(
        escalador.transform(df_two_clean.values),
        columns=sel_cols,
        index=df_two_clean.index,
    )

    st.subheader("Datos normalizados (0 a 1)")
    st.dataframe(df_norm)

    # Control para seleccionar k (número de clusters) limitado por n_samples
    max_k = max(2, min(9, len(df_norm) - 1))
    k_default = 3 if 3 <= max_k else max_k
    k = st.sidebar.slider("Número de clusters (k)", min_value=2, max_value=max_k, value=k_default, step=1)

    # Ajuste de K-Means con k seleccionado
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(df_norm.values)
    df_result = df_norm.copy()
    df_result["cluster"] = kmeans.labels_

    # Gráfica de clusters con centroides
    colores = ["red", "blue", "orange", "black", "purple", "pink", "brown"]
    fig1, ax1 = plt.subplots(figsize=(6, 5), dpi=100)
    for c in range(kmeans.n_clusters):
        ax1.scatter(
            df_result[df_result["cluster"] == c][sel_cols[0]],
            df_result[df_result["cluster"] == c][sel_cols[1]],
            marker="o",
            s=180,
            color=colores[c % len(colores)],
            alpha=0.5,
            label=f"Cluster {c}",
        )
        ax1.scatter(
            kmeans.cluster_centers_[c][0],
            kmeans.cluster_centers_[c][1],
            marker="P",
            s=280,
            color=colores[c % len(colores)],
            edgecolors="white",
        )
    ax1.set_title("Clusters", fontsize=16)
    ax1.set_xlabel(sel_cols[0], fontsize=12)
    ax1.set_ylabel(sel_cols[1], fontsize=12)
    ax1.text(1.02, 0.2, f"k={kmeans.n_clusters}", transform=ax1.transAxes, fontsize=12)
    ax1.text(1.02, 0.1, f"Inercia={kmeans.inertia_:.2f}", transform=ax1.transAxes, fontsize=12)
    ax1.set_xlim(-0.1, 1.15)
    ax1.set_ylim(-0.1, 1.15)
    ax1.legend(loc="best")
    st.subheader("Clusters (K-Means)")
    st.pyplot(fig1)

    # Método del codo (k = 2..9)
    ks = list(range(2, max_k + 1))
    if len(ks) >= 1:
        inercias = []
        for kk in ks:
            km = KMeans(n_clusters=kk, random_state=42, n_init=10).fit(df_norm.values)
            inercias.append(km.inertia_)

        fig2, ax2 = plt.subplots(figsize=(6, 5), dpi=100)
        ax2.scatter(ks, inercias, marker="o", s=180, color="purple")
        ax2.set_xlabel("Número de clusters", fontsize=12)
        ax2.set_ylabel("Inercia", fontsize=12)
        ax2.set_title("Método del codo", fontsize=16)
        st.subheader("Método del codo")
        st.pyplot(fig2)
    else:
        st.info("No es posible calcular el método del codo con tan pocas filas válidas.")

    # Tabla con resultado (agregamos la etiqueta de cluster a los datos originales)
    salida = df_raw.copy()
    salida["cluster"] = pd.NA
    salida.loc[df_result.index, "cluster"] = kmeans.labels_
    st.subheader("Tabla de resultados (con cluster)")
    st.dataframe(salida)

    # Botón para descargar CSV con cluster asignado
    csv_bytes = salida.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Descargar CSV con clusters",
        data=csv_bytes,
        file_name="clientes_con_kmeans.csv",
        mime="text/csv",
    )
else:
    st.info("Esperando a que cargues un archivo CSV…")
