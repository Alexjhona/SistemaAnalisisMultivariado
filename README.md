# 📊 Calculadora de Análisis Multivariado

Aplicación web desarrollada con **Python** y **Streamlit** para cargar datos en formato CSV o Excel y resolver análisis relacionados con series de tiempo, pronósticos, regresión lineal múltiple, clasificación y matriz de confusión.

## 📌 Descripción

El sistema permite cargar una base de datos, visualizar su contenido y aplicar diferentes métodos de análisis de forma interactiva.

Está orientado al desarrollo de ejercicios académicos donde se requiere seleccionar variables, generar resultados, interpretar métricas y obtener gráficos de apoyo.

La aplicación permite trabajar con datos numéricos, columnas de tiempo, variables dependientes, variables independientes y variables de clasificación.

---

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

---

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

---

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

```text
Y = a + b1X1 + b2X2 + ... + bnXn
```

---

## 🧮 Clasificación y matriz de confusión

Este módulo permite convertir una variable numérica en categorías, por ejemplo:

```text
Alto / Bajo
Positivo / Negativo
Aprobado / Desaprobado
```

Luego, el sistema calcula la matriz de confusión y sus métricas principales.

Métricas incluidas:

- Verdaderos positivos.
- Falsos positivos.
- Falsos negativos.
- Verdaderos negativos.
- Exactitud.
- Precisión.
- Sensibilidad.
- Especificidad.
- F1-score.

---

# ⚙️ Guía de instalación y ejecución

Para ejecutar el sistema se debe tener instalado **Python 3.10 o superior**.

---

## 1. Clonar el repositorio

```bash
git clone https://github.com/Alexjhona/SistemaAnalisisMultivariado.git
```

---

## 2. Entrar a la carpeta del proyecto

```bash
cd SistemaAnalisisMultivariado
```

---

## 3. Crear un entorno virtual

En Windows:

```bash
python -m venv venv
```

---

## 4. Activar el entorno virtual

En CMD:

```bash
venv\Scripts\activate
```

En PowerShell:

```powershell
.\venv\Scripts\activate
```

Si PowerShell no permite activar el entorno virtual, ejecutar:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Luego volver a activar:

```powershell
.\venv\Scripts\activate
```

---

## 5. Instalar las librerías necesarias

```bash
pip install -r requirements.txt
```

---

## 6. Ejecutar la aplicación

```bash
streamlit run app.py
```

Después de ejecutar el comando, Streamlit abrirá la aplicación en el navegador.

También se puede ingresar manualmente a:

```text
http://localhost:8501
```

---

# 📦 Librerías utilizadas

El archivo `requirements.txt` contiene las librerías necesarias para ejecutar el sistema.

Las principales librerías utilizadas son:

```text
streamlit
pandas
numpy
matplotlib
openpyxl
statsmodels
scikit-learn
scipy
requests
```

Estas librerías permiten cargar archivos, procesar datos, generar gráficos, aplicar modelos estadísticos y ejecutar la interfaz web del sistema.

---

# 📁 Estructura del proyecto

```text
SistemaAnalisisMultivariado/
│
├── app.py
├── funciones_analisis.py
├── requirements.txt
├── README.md
├── datos/
├── .streamlit/
│   └── config.toml
└── .gitignore
```

---

# 📄 Archivos principales

## `app.py`

Archivo principal de la aplicación. Contiene la interfaz desarrollada con Streamlit.

## `funciones_analisis.py`

Archivo que contiene las funciones para series de tiempo, pronósticos, regresión lineal múltiple, clasificación y matriz de confusión.

## `requirements.txt`

Archivo donde se encuentran las librerías necesarias para ejecutar el sistema.

## `.streamlit/config.toml`

Archivo de configuración visual de Streamlit.

---

# ⚠️ Archivos que no se suben al repositorio

No se sube el entorno virtual ni archivos temporales.  
Por eso el proyecto incluye un archivo `.gitignore` con configuraciones como:

```gitignore
venv/
__pycache__/
*.pyc
.env
.streamlit/secrets.toml
.ipynb_checkpoints/
*.zip
*.rar
*.7z
```

El entorno virtual `venv` no se incluye en GitHub porque cada usuario debe crearlo en su propia computadora.

El archivo `.env` tampoco se sube al repositorio porque puede contener configuraciones privadas, claves o rutas locales.

---

# 🤖 Consulta con IA local

La sección de consulta con IA local es opcional.

El sistema principal funciona sin IA y permite resolver:

- Series de tiempo y pronósticos.
- Regresión lineal múltiple.
- Clasificación.
- Matriz de confusión.

Para usar la consulta con IA local se requiere tener **LM Studio** ejecutándose localmente en:

```text
http://localhost:1234
```

Si no se utiliza LM Studio, los módulos estadísticos funcionan normalmente.

---

# ✅ Uso general del sistema

1. Ejecutar la aplicación con Streamlit.
2. Cargar un archivo CSV, XLSX o XLS.
3. Revisar la vista previa de los datos.
4. Seleccionar el módulo de análisis.
5. Elegir las variables correspondientes.
6. Generar resultados, tablas, métricas y gráficos.
7. Interpretar los resultados obtenidos.

---

# 📌 Nota final

El repositorio no incluye el entorno virtual `venv`.

Para ejecutar el proyecto desde otra computadora, se debe crear nuevamente el entorno virtual e instalar las dependencias con:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
