# aprendizaje-no-supervisado-KMeans

App de Streamlit para segmentación de clientes con K-Means.

Cómo ejecutar:
- Crear entorno e instalar dependencias: `pip install -r requirements.txt`
- Iniciar la app: `streamlit run main.py`

Uso:
- Sube un archivo `.csv` con dos columnas numéricas (p. ej. `ingresos,puntuacion`).
- Elige en la barra lateral las dos columnas a usar si el archivo tiene más columnas.
- La app mostrará:
  - Tabla de datos originales
  - Tabla de datos normalizados
  - Gráfica de clusters con centroides
  - Gráfica del método del codo
- Puedes descargar un CSV con la etiqueta de `cluster` añadida.
