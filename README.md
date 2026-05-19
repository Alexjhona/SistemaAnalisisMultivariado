# 📈 Sistema de Pronóstico de Series de Tiempo

Aplicación web desarrollada en Python con Streamlit para cargar datos en formato CSV o Excel y generar pronósticos mediante métodos básicos de series de tiempo.

## 📌 Descripción

El sistema permite al usuario cargar un archivo con datos de una serie temporal, seleccionar una columna de tiempo y una columna numérica, y generar pronósticos automáticos utilizando diferentes métodos clásicos.

La aplicación presenta una vista previa de los datos, información general del conjunto cargado, tabla de pronósticos, resumen de resultados, descripción de los métodos aplicados y gráficas comparativas e individuales.

## 🧠 Métodos implementados

Los métodos de pronóstico incluidos en la aplicación son:

- Método ingenuo
- Método de la media
- Método de la media móvil
- Método de la deriva
- Método ingenuo estacional

## 📁 Estructura del proyecto

```txt
TIME-SERIES-BENCHMARK-PROJECT
├── .streamlit
│   └── config.toml
├── datos
│   └── apple.csv
├── app.py
├── README.md
├── requirements.txt
└── venv  

## Formato de estructura: 

fecha,valor
1/05/2023,169.5
2/05/2023,170.2
3/05/2023,172.1
4/05/2023,173.5


⚙️ Requisitos



Si el proyecto fue descargado manualmente, solo abrir la carpeta del proyecto en Visual Studio Code.

🐍 Crear entorno virtual

En Windows, ejecutar:


python -m venv venv


Activar el entorno virtual:


.\venv\Scripts\activate

Si se usa PowerShell y aparece un error de permisos, ejecutar:

Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser


Luego volver a activar:

.\venv\Scripts\activate

📥 Instalar librerías

Actualizar pip:

python -m pip install --upgrade pip

Instalar las dependencias del proyecto:


pip install -r requirements.txt


El archivo requirements.txt debe contener:

streamlit
pandas
numpy
matplotlib
statsmodels
openpyxl



▶️ Ejecutar la aplicación

Para iniciar la aplicación, ejecutar:





Este en Windows: 

python -m streamlit run app.py

