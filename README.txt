Sistema de Análisis Multivariado

Aplicación web local desarrollada con Python y Streamlit para resolver ejercicios de Análisis Multivariado.

Ejecución:
python -m streamlit run app.py

Requisitos:
pip install -r requirements.txt

Funcionalidades principales:
- Carga de archivos CSV, XLSX y XLS.
- Limpieza básica de datos.
- Diagnóstico automático de columnas numéricas, categóricas y posibles fechas.
- Series de tiempo y pronósticos con métodos simples.
- Comparación de modelos de pronóstico mediante MAE, RMSE y MAPE.
- Proyección de tendencia lineal.
- Descomposición de series de tiempo.
- Regresión lineal múltiple con selección de variable dependiente Y y variables independientes X.
- Modelo de regresión con coeficientes, ecuación estimada, R², R² ajustado y p-value.
- Evaluación de multicolinealidad mediante VIF.
- Diagnóstico del modelo mediante residuos y gráficos.
- Modelo optimizado.
- Predicción con nuevos valores de entrada.
- Clasificación y matriz de confusión.
- Conversión de variables numéricas a categorías Alto/Bajo usando mediana o promedio.
- Métricas de matriz de confusión: VP, FP, FN, VN, exactitud, precisión, sensibilidad, especificidad y F1-score.

Uso recomendado:
1. Ejecutar la aplicación.
2. Cargar una data CSV o Excel.
3. Elegir el módulo correspondiente al ejercicio:
   - Series de tiempo y pronósticos.
   - Regresión lineal múltiple.
   - Clasificación y matriz de confusión.
4. Seleccionar las columnas solicitadas por el ejercicio.
5. Ejecutar el análisis.
6. Copiar resultados o tomar capturas para la entrega.

Notas:
- El sistema trabaja localmente con Python.
- Si una data no cumple las condiciones necesarias para un método, el sistema mostrará una advertencia para revisar las columnas seleccionadas.
- Para series de tiempo, el horizonte representa la cantidad de periodos a pronosticar según la frecuencia de la data.
