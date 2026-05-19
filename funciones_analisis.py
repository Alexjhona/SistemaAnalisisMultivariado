# funciones_analisis.py
# Funciones de apoyo para la app de Análisis Multivariado.
# Este archivo guarda los cálculos para mantener app.py más ordenado.

import math
import numpy as np
import pandas as pd

from scipy import stats

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.seasonal import seasonal_decompose

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# ============================================================
# 1. LIMPIEZA Y DIAGNÓSTICO GENERAL
# ============================================================

def limpiar_dataframe(df: pd.DataFrame):
    """
    Limpia la data de forma básica:
    - elimina columnas totalmente vacías
    - elimina filas totalmente vacías
    - convierte columnas tipo texto a numéricas cuando se puede
    """
    df = df.copy()
    reporte = []

    columnas_antes = df.shape[1]
    filas_antes = df.shape[0]

    df = df.dropna(axis=1, how="all")
    df = df.dropna(axis=0, how="all")
    df = df.replace([np.inf, -np.inf], np.nan)

    if df.shape[1] < columnas_antes:
        reporte.append(f"Se eliminaron {columnas_antes - df.shape[1]} columnas totalmente vacías.")
    if df.shape[0] < filas_antes:
        reporte.append(f"Se eliminaron {filas_antes - df.shape[0]} filas totalmente vacías.")

    for col in df.columns:
        if df[col].dtype == "object":
            serie_texto = df[col].astype(str).str.strip()
            serie_texto = serie_texto.str.replace(",", ".", regex=False)
            convertida = pd.to_numeric(serie_texto, errors="coerce")

            proporcion_convertible = convertida.notna().mean()
            if proporcion_convertible >= 0.80:
                df[col] = convertida
                reporte.append(f"La columna '{col}' fue convertida a numérica.")

    if not reporte:
        reporte.append("No se realizaron cambios importantes en la estructura de la data.")

    return df, reporte


def detectar_tipos(df: pd.DataFrame):
    """
    Detecta columnas numéricas, categóricas y posibles columnas de fecha.
    """
    numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    categoricas = df.select_dtypes(exclude=[np.number]).columns.tolist()

    posibles_fechas = []
    for col in df.columns:
        if df[col].dtype == "object" or "date" in str(df[col].dtype).lower():
            fechas = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
            if fechas.notna().mean() >= 0.70:
                posibles_fechas.append(col)

    return {
        "numericas": numericas,
        "categoricas": categoricas,
        "posibles_fechas": posibles_fechas,
    }


def diagnostico_basico(df: pd.DataFrame):
    """
    Devuelve resumen general de filas, columnas, tipos y datos faltantes.
    """
    tipos = detectar_tipos(df)
    faltantes = pd.DataFrame({
        "columna": df.columns,
        "tipo": [str(df[c].dtype) for c in df.columns],
        "faltantes": [int(df[c].isna().sum()) for c in df.columns],
        "porcentaje_faltante": [round(df[c].isna().mean() * 100, 2) for c in df.columns],
    })

    resumen = {
        "filas": df.shape[0],
        "columnas": df.shape[1],
        "numericas": tipos["numericas"],
        "categoricas": tipos["categoricas"],
        "posibles_fechas": tipos["posibles_fechas"],
        "faltantes": faltantes,
    }
    return resumen


def recomendacion_modulo(df: pd.DataFrame):
    """
    Sugiere qué módulo usar según la estructura de la data.
    """
    tipos = detectar_tipos(df)
    mensajes = []

    if len(tipos["posibles_fechas"]) >= 1 and len(tipos["numericas"]) >= 1:
        mensajes.append("La data tiene posible columna de fecha y variables numéricas. Conviene usar Series de tiempo y pronósticos.")

    if len(tipos["numericas"]) >= 2:
        mensajes.append("La data tiene varias variables numéricas. Conviene usar Regresión múltiple o matriz de correlación.")

    if len(tipos["categoricas"]) >= 1 or any(df[c].nunique(dropna=True) <= 2 for c in df.columns):
        mensajes.append("La data tiene variables categóricas o binarias. Puede servir para Clasificación y matriz de confusión.")

    if not mensajes:
        mensajes.append("La data requiere revisión manual. Primero identifica una variable objetivo y luego elige el módulo.")

    return mensajes


# ============================================================
# 2. SERIES DE TIEMPO Y PRONÓSTICOS
# ============================================================

def preparar_serie(df: pd.DataFrame, columna_tiempo: str, columna_valor: str):
    """
    Prepara una serie de tiempo.
    Ordena por fecha/periodo y deja solo la variable numérica.
    """
    data = df[[columna_tiempo, columna_valor]].copy()
    data[columna_valor] = pd.to_numeric(data[columna_valor], errors="coerce")
    data = data.dropna(subset=[columna_valor])

    fechas = pd.to_datetime(data[columna_tiempo], errors="coerce", dayfirst=True)
    if fechas.notna().mean() >= 0.70:
        data["_tiempo_orden"] = fechas
    else:
        data["_tiempo_orden"] = pd.to_numeric(data[columna_tiempo], errors="coerce")
        if data["_tiempo_orden"].isna().all():
            data["_tiempo_orden"] = np.arange(1, len(data) + 1)

    data = data.sort_values("_tiempo_orden").reset_index(drop=True)
    data["periodo"] = np.arange(1, len(data) + 1)
    data = data.rename(columns={columna_tiempo: "tiempo", columna_valor: "valor"})

    return data[["tiempo", "periodo", "valor"]]


def pronostico_ingenuo(y, horizonte):
    y = np.asarray(y, dtype=float)
    return np.repeat(y[-1], horizonte)


def pronostico_media(y, horizonte):
    y = np.asarray(y, dtype=float)
    return np.repeat(np.mean(y), horizonte)


def pronostico_media_movil(y, horizonte, ventana=3):
    y = np.asarray(y, dtype=float)
    ventana = max(1, min(int(ventana), len(y)))
    valores = list(y[-ventana:])
    pron = []

    for _ in range(horizonte):
        nuevo = float(np.mean(valores[-ventana:]))
        pron.append(nuevo)
        valores.append(nuevo)

    return np.array(pron)


def pronostico_deriva(y, horizonte):
    y = np.asarray(y, dtype=float)
    if len(y) < 2:
        return pronostico_ingenuo(y, horizonte)

    primero = y[0]
    ultimo = y[-1]
    n = len(y)

    return np.array([
        ultimo + h * ((ultimo - primero) / (n - 1))
        for h in range(1, horizonte + 1)
    ])


def pronostico_suavizamiento_exponencial(y, horizonte, alpha=0.3):
    y = np.asarray(y, dtype=float)
    alpha = float(alpha)

    s = y[0]
    for valor in y[1:]:
        s = alpha * valor + (1 - alpha) * s

    return np.repeat(s, horizonte)


def pronostico_tendencia_lineal(y, horizonte):
    y = np.asarray(y, dtype=float)
    x = np.arange(1, len(y) + 1)

    if len(y) < 2:
        return pronostico_ingenuo(y, horizonte), 0, y[-1]

    b, a = np.polyfit(x, y, 1)
    x_futuro = np.arange(len(y) + 1, len(y) + horizonte + 1)
    pron = a + b * x_futuro

    return pron, a, b


def pronostico_ingenuo_estacional(y, horizonte, periodo_estacional=7):
    y = np.asarray(y, dtype=float)
    periodo_estacional = int(periodo_estacional)

    if periodo_estacional <= 0 or len(y) < periodo_estacional:
        return pronostico_ingenuo(y, horizonte)

    base = y[-periodo_estacional:]
    return np.array([base[i % periodo_estacional] for i in range(horizonte)])


def generar_pronosticos(y, horizonte=5, ventana=3, periodo_estacional=7, alpha=0.3):
    """
    Calcula varios métodos de pronóstico.
    """
    y = np.asarray(y, dtype=float)
    tendencia, a, b = pronostico_tendencia_lineal(y, horizonte)

    pronosticos = {
        "Método ingenuo": pronostico_ingenuo(y, horizonte),
        "Método de la media": pronostico_media(y, horizonte),
        "Método de media móvil": pronostico_media_movil(y, horizonte, ventana),
        "Método de la deriva": pronostico_deriva(y, horizonte),
        "Suavizamiento exponencial simple": pronostico_suavizamiento_exponencial(y, horizonte, alpha),
        "Proyección de tendencia lineal": tendencia,
        "Método ingenuo estacional": pronostico_ingenuo_estacional(y, horizonte, periodo_estacional),
    }

    tabla = pd.DataFrame({"Periodo futuro": np.arange(1, horizonte + 1)})
    for nombre, valores in pronosticos.items():
        tabla[nombre] = np.round(valores, 4)

    info_tendencia = {
        "a_intercepto": float(a),
        "b_pendiente": float(b),
    }

    return tabla, pronosticos, info_tendencia


def metricas_pronostico(y_real, y_pred):
    """
    Calcula MAE, RMSE y MAPE.
    """
    y_real = np.asarray(y_real, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = np.mean(np.abs(y_real - y_pred))
    rmse = np.sqrt(np.mean((y_real - y_pred) ** 2))

    mask = y_real != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_real[mask] - y_pred[mask]) / y_real[mask])) * 100
    else:
        mape = np.nan

    return {
        "MAE": round(float(mae), 4),
        "RMSE": round(float(rmse), 4),
        "MAPE (%)": round(float(mape), 4) if not np.isnan(mape) else np.nan,
    }


def evaluar_pronosticos(y, ventana=3, periodo_estacional=7, alpha=0.3):
    """
    Evalúa métodos usando los últimos datos como prueba.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)

    if n < 8:
        return pd.DataFrame()

    test_size = min(max(2, int(round(n * 0.20))), 10)
    train = y[:-test_size]
    test = y[-test_size:]

    tabla_pron, pronosticos, _ = generar_pronosticos(
        train,
        horizonte=test_size,
        ventana=ventana,
        periodo_estacional=periodo_estacional,
        alpha=alpha
    )

    filas = []
    for nombre, pred in pronosticos.items():
        m = metricas_pronostico(test, pred)
        filas.append({
            "Método": nombre,
            "MAE": m["MAE"],
            "RMSE": m["RMSE"],
            "MAPE (%)": m["MAPE (%)"],
        })

    resultado = pd.DataFrame(filas).sort_values("RMSE", ascending=True).reset_index(drop=True)
    return resultado


def interpretar_serie(y, info_tendencia, tabla_metricas):
    """
    Genera una conclusión textual simple para series de tiempo.
    """
    y = np.asarray(y, dtype=float)
    pendiente = info_tendencia.get("b_pendiente", 0)

    if pendiente > 0:
        tendencia = "creciente"
    elif pendiente < 0:
        tendencia = "decreciente"
    else:
        tendencia = "estable"

    texto = f"La serie presenta una tendencia {tendencia}. "

    if len(tabla_metricas) > 0:
        mejor = tabla_metricas.iloc[0]["Método"]
        rmse = tabla_metricas.iloc[0]["RMSE"]
        texto += f"Según la evaluación con datos recientes, el método con menor RMSE fue '{mejor}' con un valor de {rmse}. "
        texto += "Por ello, ese método puede tomarse como referencia principal para el pronóstico."
    else:
        texto += "No hay suficientes datos para comparar la exactitud de los métodos con una partición de prueba."

    return texto


# ============================================================
# 3. REGRESIÓN LINEAL MÚLTIPLE
# ============================================================

def preparar_xy_regresion(df: pd.DataFrame, y_col: str, x_cols: list):
    """
    Prepara X e Y para regresión.
    Convierte variables categóricas a dummies.
    """
    data = df[[y_col] + x_cols].copy()
    data[y_col] = pd.to_numeric(data[y_col], errors="coerce")
    data = data.dropna(subset=[y_col])

    X = data[x_cols].copy()

    for col in X.columns:
        if X[col].dtype == "object":
            numerica = pd.to_numeric(X[col], errors="coerce")
            if numerica.notna().mean() >= 0.80:
                X[col] = numerica

    X = pd.get_dummies(X, drop_first=True, dtype=float)
    y = data[y_col].astype(float)

    modelo_data = pd.concat([y, X], axis=1).dropna()
    y = modelo_data[y_col]
    X = modelo_data.drop(columns=[y_col])

    # Elimina variables constantes, por ejemplo una columna Intercepto = 1.
    # Statsmodels ya agrega la constante automáticamente.
    X = X.loc[:, X.nunique(dropna=True) > 1]

    if X.shape[1] == 0:
        raise ValueError("Después de limpiar variables constantes no quedaron variables X válidas.")

    X_const = sm.add_constant(X, has_constant="add")
    return y, X, X_const


def ajustar_regresion_multiple(df: pd.DataFrame, y_col: str, x_cols: list):
    """
    Ajusta un modelo de regresión múltiple con mínimos cuadrados.
    """
    y, X, X_const = preparar_xy_regresion(df, y_col, x_cols)

    if len(y) <= X_const.shape[1]:
        raise ValueError("No hay suficientes filas para ajustar el modelo con tantas variables.")

    modelo = sm.OLS(y, X_const).fit()

    coeficientes = pd.DataFrame({
        "Variable": modelo.params.index,
        "Coeficiente": np.round(modelo.params.values, 6),
        "p_value": np.round(modelo.pvalues.values, 6),
        "Significativo_0.05": modelo.pvalues.values < 0.05,
    })

    # VIF
    vif_filas = []
    if X.shape[1] >= 1:
        X_vif = X.copy()
        X_vif = X_vif.loc[:, X_vif.nunique(dropna=True) > 1]

        for i, col in enumerate(X_vif.columns):
            try:
                vif_val = variance_inflation_factor(X_vif.values.astype(float), i)
            except Exception:
                vif_val = np.nan

            if np.isinf(vif_val):
                interpretacion = "Multicolinealidad perfecta"
                valor_mostrar = np.inf
            elif np.isnan(vif_val):
                interpretacion = "No calculable"
                valor_mostrar = np.nan
            elif vif_val > 10:
                interpretacion = "Alta"
                valor_mostrar = round(float(vif_val), 4)
            elif vif_val > 5:
                interpretacion = "Revisar"
                valor_mostrar = round(float(vif_val), 4)
            else:
                interpretacion = "Aceptable"
                valor_mostrar = round(float(vif_val), 4)

            vif_filas.append({"Variable": col, "VIF": valor_mostrar, "Interpretación": interpretacion})

    vif = pd.DataFrame(vif_filas)

    # Diagnósticos de supuestos
    residuos = modelo.resid
    fitted = modelo.fittedvalues

    try:
        shapiro_p = stats.shapiro(residuos).pvalue if len(residuos) <= 5000 else np.nan
    except Exception:
        shapiro_p = np.nan

    try:
        bp_test = het_breuschpagan(residuos, modelo.model.exog)
        bp_p = bp_test[1]
    except Exception:
        bp_p = np.nan

    try:
        dw = durbin_watson(residuos)
    except Exception:
        dw = np.nan

    diagnostico = {
        "r2": round(float(modelo.rsquared), 4),
        "r2_ajustado": round(float(modelo.rsquared_adj), 4),
        "p_value_modelo": round(float(modelo.f_pvalue), 6) if modelo.f_pvalue is not None else np.nan,
        "aic": round(float(modelo.aic), 4),
        "bic": round(float(modelo.bic), 4),
        "shapiro_p_residuos": round(float(shapiro_p), 6) if not np.isnan(shapiro_p) else np.nan,
        "breusch_pagan_p": round(float(bp_p), 6) if not np.isnan(bp_p) else np.nan,
        "durbin_watson": round(float(dw), 4) if not np.isnan(dw) else np.nan,
    }

    return {
        "modelo": modelo,
        "coeficientes": coeficientes,
        "vif": vif,
        "diagnostico": diagnostico,
        "y": y,
        "X": X,
        "X_const": X_const,
        "residuos": residuos,
        "fitted": fitted,
    }


def construir_ecuacion(coeficientes: pd.DataFrame, y_col: str):
    """
    Construye texto de ecuación estimada.
    """
    partes = []
    for _, row in coeficientes.iterrows():
        variable = row["Variable"]
        coef = row["Coeficiente"]
        if variable == "const":
            partes.append(f"{coef}")
        else:
            signo = "+" if coef >= 0 else "-"
            partes.append(f"{signo} {abs(coef)}({variable})")

    return f"{y_col} estimado = " + " ".join(partes)


def interpretar_regresion(resultado, y_col: str):
    """
    Interpretación sencilla del modelo de regresión.
    """
    diag = resultado["diagnostico"]
    coefs = resultado["coeficientes"]

    texto = f"El modelo de regresión lineal múltiple explica aproximadamente el {diag['r2'] * 100:.2f}% de la variabilidad de '{y_col}' según el R². "

    p_modelo = diag.get("p_value_modelo", np.nan)
    if not np.isnan(p_modelo) and p_modelo < 0.05:
        texto += "El p-value global del modelo es menor a 0.05, por lo tanto el modelo es estadísticamente significativo. "
    else:
        texto += "El p-value global del modelo no es menor a 0.05, por lo tanto el modelo debe interpretarse con cuidado. "

    significativas = coefs[(coefs["Variable"] != "const") & (coefs["Significativo_0.05"] == True)]["Variable"].tolist()
    if significativas:
        texto += "Las variables significativas al 5% son: " + ", ".join(significativas) + ". "
    else:
        texto += "No se encontraron variables significativas al 5% en los coeficientes individuales. "

    vif = resultado["vif"]
    if len(vif) > 0:
        vif_valores = vif["VIF"].replace([np.inf, -np.inf], np.nan)
        hay_inf = np.isinf(vif["VIF"]).any()
        hay_alto = vif_valores.dropna().gt(10).any()
        if hay_inf or hay_alto:
            texto += "Existe posible multicolinealidad alta, porque una o más variables tienen VIF mayor a 10 o multicolinealidad perfecta."
        else:
            texto += "No se observa multicolinealidad alta usando el criterio VIF mayor a 10."
    else:
        texto += "No se pudo calcular VIF para evaluar multicolinealidad."

    return texto




def _calcular_vif_desde_x(X: pd.DataFrame):
    """
    Calcula VIF e interpretación para un conjunto X ya numérico/dummy.
    """
    vif_filas = []
    if X is None or X.shape[1] == 0:
        return pd.DataFrame(vif_filas)

    X_vif = X.copy()
    X_vif = X_vif.loc[:, X_vif.nunique(dropna=True) > 1]

    if X_vif.shape[1] == 1:
        col = X_vif.columns[0]
        return pd.DataFrame([{"Variable": col, "VIF": 1.0, "Interpretación": "Aceptable"}])

    for i, col in enumerate(X_vif.columns):
        try:
            vif_val = variance_inflation_factor(X_vif.values.astype(float), i)
        except Exception:
            vif_val = np.nan

        if np.isinf(vif_val):
            interpretacion = "Multicolinealidad perfecta"
            valor_mostrar = np.inf
        elif np.isnan(vif_val):
            interpretacion = "No calculable"
            valor_mostrar = np.nan
        elif vif_val > 10:
            interpretacion = "Alta"
            valor_mostrar = round(float(vif_val), 4)
        elif vif_val > 5:
            interpretacion = "Revisar"
            valor_mostrar = round(float(vif_val), 4)
        else:
            interpretacion = "Aceptable"
            valor_mostrar = round(float(vif_val), 4)

        vif_filas.append({"Variable": col, "VIF": valor_mostrar, "Interpretación": interpretacion})

    return pd.DataFrame(vif_filas)


def _armar_resultado_regresion(y: pd.Series, X: pd.DataFrame, modelo, extra=None):
    """
    Arma el mismo diccionario de salida usado por ajustar_regresion_multiple.
    Sirve para el modelo normal y el modelo optimizado.
    """
    coeficientes = pd.DataFrame({
        "Variable": modelo.params.index,
        "Coeficiente": np.round(modelo.params.values, 6),
        "p_value": np.round(modelo.pvalues.values, 6),
        "Significativo_0.05": modelo.pvalues.values < 0.05,
    })

    vif = _calcular_vif_desde_x(X)

    residuos = modelo.resid
    fitted = modelo.fittedvalues

    try:
        shapiro_p = stats.shapiro(residuos).pvalue if len(residuos) <= 5000 else np.nan
    except Exception:
        shapiro_p = np.nan

    try:
        bp_test = het_breuschpagan(residuos, modelo.model.exog)
        bp_p = bp_test[1]
    except Exception:
        bp_p = np.nan

    try:
        dw = durbin_watson(residuos)
    except Exception:
        dw = np.nan

    diagnostico = {
        "r2": round(float(modelo.rsquared), 4),
        "r2_ajustado": round(float(modelo.rsquared_adj), 4),
        "p_value_modelo": round(float(modelo.f_pvalue), 6) if modelo.f_pvalue is not None else np.nan,
        "aic": round(float(modelo.aic), 4),
        "bic": round(float(modelo.bic), 4),
        "shapiro_p_residuos": round(float(shapiro_p), 6) if not np.isnan(shapiro_p) else np.nan,
        "breusch_pagan_p": round(float(bp_p), 6) if not np.isnan(bp_p) else np.nan,
        "durbin_watson": round(float(dw), 4) if not np.isnan(dw) else np.nan,
    }

    X_const = sm.add_constant(X, has_constant="add")

    resultado = {
        "modelo": modelo,
        "coeficientes": coeficientes,
        "vif": vif,
        "diagnostico": diagnostico,
        "y": y,
        "X": X,
        "X_const": X_const,
        "residuos": residuos,
        "fitted": fitted,
    }

    if extra:
        resultado.update(extra)

    return resultado


def optimizar_regresion_por_pvalue(df: pd.DataFrame, y_col: str, x_cols: list, alpha=0.05):
    """
    Genera un modelo optimizado por eliminación hacia atrás.
    Elimina una a una las variables con mayor p-value mientras sea mayor a alpha.
    Mantiene al menos una variable X para evitar dejar un modelo vacío.
    """
    y, X, _ = preparar_xy_regresion(df, y_col, x_cols)
    activas = list(X.columns)
    eliminadas = []

    if len(activas) == 0:
        raise ValueError("No hay variables X válidas para optimizar.")

    while len(activas) > 1:
        X_temp = X[activas]
        X_const = sm.add_constant(X_temp, has_constant="add")
        modelo_temp = sm.OLS(y, X_const).fit()

        pvalues = modelo_temp.pvalues.drop(labels=["const"], errors="ignore")
        if pvalues.empty:
            break

        peor_variable = pvalues.idxmax()
        peor_pvalue = float(pvalues.max())

        if peor_pvalue <= alpha:
            break

        activas.remove(peor_variable)
        eliminadas.append({
            "Variable eliminada": peor_variable,
            "p_value": round(peor_pvalue, 6),
            "Motivo": f"p-value mayor a {alpha}"
        })

    X_final = X[activas]
    X_final_const = sm.add_constant(X_final, has_constant="add")
    modelo_final = sm.OLS(y, X_final_const).fit()

    extra = {
        "variables_finales": activas,
        "variables_eliminadas": pd.DataFrame(eliminadas),
        "metodo_optimizacion": "Eliminación hacia atrás por p-value"
    }

    return _armar_resultado_regresion(y, X_final, modelo_final, extra=extra)


def interpretar_modelo_optimizado(resultado_opt, y_col: str):
    """
    Interpretación breve del modelo optimizado.
    """
    diag = resultado_opt["diagnostico"]
    finales = resultado_opt.get("variables_finales", [])
    eliminadas = resultado_opt.get("variables_eliminadas", pd.DataFrame())

    texto = (
        f"El modelo optimizado conserva {len(finales)} variable(s) para explicar '{y_col}'. "
        f"Su R² ajustado es {diag['r2_ajustado']:.4f}, por lo que explica aproximadamente "
        f"{diag['r2'] * 100:.2f}% de la variabilidad de la variable dependiente. "
    )

    p_modelo = diag.get("p_value_modelo", np.nan)
    if not np.isnan(p_modelo) and p_modelo < 0.05:
        texto += "El modelo optimizado es estadísticamente significativo porque el p-value global es menor a 0.05. "
    else:
        texto += "El modelo optimizado debe revisarse porque el p-value global no es menor a 0.05. "

    if len(eliminadas) > 0:
        texto += "Se eliminaron variables con bajo aporte estadístico según su p-value. "
    else:
        texto += "No fue necesario eliminar variables por el criterio de p-value. "

    return texto


# ============================================================
# 3.1 DESCOMPOSICIÓN PARA SERIES DE TIEMPO
# ============================================================

def descomponer_serie(y, periodo_estacional=7, modelo="aditivo"):
    """
    Descompone una serie en observado, tendencia, estacionalidad y residuo.
    Requiere al menos dos ciclos completos: 2 * periodo_estacional datos.
    """
    y = pd.Series(pd.to_numeric(pd.Series(y), errors="coerce")).dropna().astype(float)
    periodo_estacional = int(periodo_estacional)

    if periodo_estacional < 2:
        raise ValueError("El periodo estacional debe ser mayor o igual a 2.")

    if len(y) < periodo_estacional * 2:
        raise ValueError(
            f"Se necesitan al menos {periodo_estacional * 2} datos para descomponer con periodo {periodo_estacional}."
        )

    descomp = seasonal_decompose(
        y,
        model=modelo,
        period=periodo_estacional,
        extrapolate_trend="freq"
    )

    tabla = pd.DataFrame({
        "periodo": np.arange(1, len(y) + 1),
        "observado": descomp.observed,
        "tendencia": descomp.trend,
        "estacionalidad": descomp.seasonal,
        "residuo": descomp.resid,
    })

    return tabla


def interpretar_descomposicion(tabla_descomposicion: pd.DataFrame):
    """
    Interpreta de forma simple tendencia, estacionalidad y residuo.
    """
    tendencia = tabla_descomposicion["tendencia"].dropna()
    estacionalidad = tabla_descomposicion["estacionalidad"].dropna()
    residuo = tabla_descomposicion["residuo"].dropna()

    if len(tendencia) >= 2:
        cambio = tendencia.iloc[-1] - tendencia.iloc[0]
        if cambio > 0:
            texto_tendencia = "creciente"
        elif cambio < 0:
            texto_tendencia = "decreciente"
        else:
            texto_tendencia = "estable"
    else:
        texto_tendencia = "no claramente identificable"

    var_est = float(np.nanvar(estacionalidad)) if len(estacionalidad) else 0
    var_res = float(np.nanvar(residuo)) if len(residuo) else 0
    fuerza = 0 if (var_est + var_res) == 0 else var_est / (var_est + var_res)

    if fuerza >= 0.60:
        texto_est = "alta"
    elif fuerza >= 0.30:
        texto_est = "moderada"
    elif fuerza > 0:
        texto_est = "baja"
    else:
        texto_est = "no evidente"

    return (
        f"La descomposición muestra una tendencia {texto_tendencia}. "
        f"La estacionalidad detectada es {texto_est}. "
        "El residuo representa la parte irregular de la serie que no fue explicada por la tendencia ni por la estacionalidad."
    )


# ============================================================
# 4. CLASIFICACIÓN Y MATRIZ DE CONFUSIÓN
# ============================================================

def calcular_matriz_metricas(y_real, y_pred):
    """
    Calcula matriz de confusión y métricas.
    Considera 1 como positivo y 0 como negativo.
    """
    y_real = pd.Series(y_real).astype(int)
    y_pred = pd.Series(y_pred).astype(int)

    cm = confusion_matrix(y_real, y_pred, labels=[0, 1])
    vn, fp, fn, vp = cm.ravel()

    exactitud = accuracy_score(y_real, y_pred)
    precision = precision_score(y_real, y_pred, zero_division=0)
    sensibilidad = recall_score(y_real, y_pred, zero_division=0)
    f1 = f1_score(y_real, y_pred, zero_division=0)

    especificidad = vn / (vn + fp) if (vn + fp) > 0 else 0

    matriz = pd.DataFrame(
        [[vp, fp], [fn, vn]],
        index=["Real Positivo", "Real Negativo"],
        columns=["Predicho Positivo", "Predicho Negativo"]
    )

    metricas = {
        "VP": int(vp),
        "FP": int(fp),
        "FN": int(fn),
        "VN": int(vn),
        "Exactitud": round(float(exactitud), 4),
        "Precisión": round(float(precision), 4),
        "Sensibilidad": round(float(sensibilidad), 4),
        "Especificidad": round(float(especificidad), 4),
        "F1": round(float(f1), 4),
    }

    return matriz, metricas


def convertir_a_binario(serie, metodo="mediana"):
    """
    Convierte una variable numérica a clase binaria alto/bajo.
    """
    s = pd.to_numeric(serie, errors="coerce")

    if metodo == "promedio":
        umbral = s.mean()
    else:
        umbral = s.median()

    clase = (s >= umbral).astype(int)
    return clase, float(umbral)


def entrenar_logistica_para_matriz(df: pd.DataFrame, objetivo: str, x_cols: list, metodo_umbral="mediana"):
    """
    Crea una clasificación binaria y entrena una regresión logística.
    Sirve para obtener matriz de confusión cuando no existe columna predicha.
    """
    data = df[[objetivo] + x_cols].copy()

    # Variable objetivo
    if pd.api.types.is_numeric_dtype(data[objetivo]) and data[objetivo].nunique(dropna=True) > 2:
        data["objetivo_binario"], umbral = convertir_a_binario(data[objetivo], metodo_umbral)
        explicacion_objetivo = f"La variable '{objetivo}' fue convertida a clase binaria usando el umbral {umbral:.4f}."
    else:
        codigos, categorias = pd.factorize(data[objetivo])
        data["objetivo_binario"] = codigos
        data = data[data["objetivo_binario"].isin([0, 1])]
        umbral = None
        explicacion_objetivo = f"La variable '{objetivo}' fue tratada como clase binaria."

    X = data[x_cols].copy()
    for col in X.columns:
        if X[col].dtype == "object":
            numerica = pd.to_numeric(X[col], errors="coerce")
            if numerica.notna().mean() >= 0.80:
                X[col] = numerica

    X = pd.get_dummies(X, drop_first=True, dtype=float)
    y = data["objetivo_binario"].astype(int)

    modelo_data = pd.concat([y, X], axis=1).dropna()
    y = modelo_data["objetivo_binario"].astype(int)
    X = modelo_data.drop(columns=["objetivo_binario"])

    if y.nunique() < 2:
        raise ValueError("La variable objetivo no tiene dos clases suficientes para clasificación.")

    test_size = 0.30
    stratify = y if y.value_counts().min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=stratify
    )

    modelo = LogisticRegression(max_iter=2000)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    matriz, metricas = calcular_matriz_metricas(y_test, y_pred)

    return {
        "modelo": modelo,
        "matriz": matriz,
        "metricas": metricas,
        "y_test": y_test,
        "y_pred": y_pred,
        "X_columnas": X.columns.tolist(),
        "explicacion_objetivo": explicacion_objetivo,
    }


def interpretar_matriz(metricas):
    """
    Genera interpretación simple de matriz de confusión.
    """
    texto = (
        f"El modelo tuvo una exactitud de {metricas['Exactitud'] * 100:.2f}%. "
        f"La precisión fue de {metricas['Precisión'] * 100:.2f}%, lo que indica qué tan confiables fueron las predicciones positivas. "
        f"La sensibilidad fue de {metricas['Sensibilidad'] * 100:.2f}%, lo que indica cuántos positivos reales fueron detectados correctamente. "
        f"La especificidad fue de {metricas['Especificidad'] * 100:.2f}%, lo que indica cuántos negativos reales fueron detectados correctamente. "
        f"VP={metricas['VP']}, FP={metricas['FP']}, FN={metricas['FN']} y VN={metricas['VN']}."
    )
    return texto