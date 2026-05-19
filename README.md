# 📊 Calculadora de Análisis Multivariado

Aplicación web desarrollada con **Python** y **Streamlit** para cargar datos en formato **CSV o Excel** y resolver análisis relacionados con series de tiempo, pronósticos, regresión lineal múltiple y matriz de confusión.

## 📌 Descripción

El sistema permite cargar una base de datos, visualizar su contenido y aplicar diferentes métodos de análisis de forma interactiva.  
Está orientado al desarrollo de ejercicios académicos donde se requiere seleccionar variables, generar resultados, interpretar métricas y obtener gráficos de apoyo.

La aplicación permite trabajar con datos numéricos, columnas de tiempo, variables dependientes, variables independientes y variables de clasificación.

## ✨ Funcionalidades principales

- Carga de archivos CSV, XLSX o XLS.
- Vista previa de la data cargada.
- Limpieza y revisión básica de datos.
- Análisis de series de tiempo.
- Generación de pronósticos.
- Comparación de métodos de pronóstico.
- Regresión lineal múltiple.
- Modelo optimizado según significancia estadística.
- Matriz de correlación.
- Diagnóstico del modelo mediante residuos y VIF.
- Gráficos automáticos para presentación.
- Predicción con nuevos valores.
- Clasificación y matriz de confusión.
- Conversión de variables numéricas a categorías Alto/Bajo.
- Cálculo de métricas como exactitud, precisión, sensibilidad, especificidad y F1-score.

## 📈 Series de tiempo y pronósticos

Este módulo permite seleccionar una columna de tiempo, fecha, año o periodo, junto con una variable numérica a pronosticar.

Métodos incluidos:

- Método ingenuo.
- Método de la media.
- Media móvil simple.
- Método de la deriva.
- Suavizamiento exponencial simple.
- Proyección de tendencia lineal.
- Método ingenuo estacional.

El sistema genera una tabla de pronósticos, gráficos comparativos y una comparación de errores para identificar el método con mejor desempeño según RMSE.

## 📉 Regresión lineal múltiple

Este módulo permite estimar una variable dependiente numérica a partir de varias variables independientes.

El sistema muestra:

- Matriz de correlación.
- Ecuación del modelo.
- Coeficientes.
- R² y R² ajustado.
- p-value global del modelo.
- Variables significativas.
- Modelo optimizado.
- VIF para revisar multicolinealidad.
- Diagnóstico gráfico de residuos.
- Predicción con nuevos valores.

La forma general del modelo es:

```txt
Y = a + b1X1 + b2X2 + ... + bnXn
