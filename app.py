# app.py
# Aplicación web local tipo calculadora para Análisis Multivariado.
# Ejecutar con: python -m streamlit run app.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import statsmodels.api as sm
from scipy import stats

from funciones_analisis import (
    limpiar_dataframe,
    diagnostico_basico,
    recomendacion_modulo,
    detectar_tipos,
    preparar_serie,
    generar_pronosticos,
    evaluar_pronosticos,
    interpretar_serie,
    ajustar_regresion_multiple,
    construir_ecuacion,
    interpretar_regresion,
    optimizar_regresion_por_pvalue,
    interpretar_modelo_optimizado,
    descomponer_serie,
    interpretar_descomposicion,
    calcular_matriz_metricas,
    entrenar_logistica_para_matriz,
    interpretar_matriz,
)


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

st.set_page_config(
    page_title="Calculadora de Análisis Multivariado",
    page_icon="🧮",
    layout="wide"
)

st.title("🧮 Calculadora de Análisis Multivariado")
st.write(
    "Carga una data en CSV o Excel y el programa intenta resolver automáticamente "
    "series de tiempo, pronósticos, regresión múltiple y matriz de confusión."
)


# ============================================================
# FUNCIONES LOCALES DE LA INTERFAZ
# ============================================================

@st.cache_data
def leer_archivo(uploaded_file):
    """
    Lee CSV o Excel de forma flexible.
    En Excel también intenta detectar automáticamente la fila de encabezados,
    útil cuando el archivo viene con filas/columnas vacías antes de la tabla.
    """
    nombre = uploaded_file.name.lower()

    def normalizar_columnas(columnas):
        nuevas = []
        usados = set()
        for i, col in enumerate(columnas):
            texto = str(col).strip()
            if texto == "" or texto.lower() in ["nan", "none"]:
                texto = f"columna_{i + 1}"
            base = texto
            contador = 2
            while texto in usados:
                texto = f"{base}_{contador}"
                contador += 1
            usados.add(texto)
            nuevas.append(texto)
        return nuevas

    def convertir_numericas_posibles(df):
        df = df.copy()
        for col in df.columns:
            if df[col].dtype == "object":
                texto = (
                    df[col]
                    .astype(str)
                    .str.strip()
                    .str.replace(",", ".", regex=False)
                    .str.replace(" ", "", regex=False)
                )
                convertido = pd.to_numeric(texto, errors="coerce")
                if convertido.notna().mean() >= 0.70:
                    df[col] = convertido
        return df

    def leer_excel_con_encabezado_automatico(uploaded_file):
        uploaded_file.seek(0)
        bruto = pd.read_excel(uploaded_file, header=None)
        bruto = bruto.dropna(axis=1, how="all")

        if bruto.empty:
            return pd.DataFrame()

        mejor_fila = 0
        mejor_puntaje = -1
        limite = min(20, len(bruto))

        for i in range(limite):
            fila = bruto.iloc[i]
            no_vacios = fila.notna().sum()
            textos = fila.dropna().map(lambda x: isinstance(x, str)).sum()
            siguiente = bruto.iloc[i + 1:i + 6] if i + 1 < len(bruto) else pd.DataFrame()
            hay_datos_despues = 1 if not siguiente.empty and siguiente.notna().sum().sum() > 0 else 0
            puntaje = textos * 2 + no_vacios + hay_datos_despues

            if no_vacios >= 2 and textos >= 1 and puntaje > mejor_puntaje:
                mejor_fila = i
                mejor_puntaje = puntaje

        columnas_originales = bruto.iloc[mejor_fila].tolist()
        columnas = normalizar_columnas(columnas_originales)

        # Toma solamente el primer bloque de datos debajo del encabezado.
        # Esto evita traer matrices auxiliares o notas que estén debajo de la tabla principal.
        filas = []
        for j in range(mejor_fila + 1, len(bruto)):
            fila = bruto.iloc[j]
            if fila.isna().all():
                if filas:
                    break
                continue
            filas.append(fila)

        if not filas:
            return pd.DataFrame(columns=columnas)

        df = pd.DataFrame(filas).copy()
        df.columns = columnas
        df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
        df = convertir_numericas_posibles(df)
        return df.reset_index(drop=True)

    if nombre.endswith(".csv"):
        uploaded_file.seek(0)

        try:
            df = pd.read_csv(uploaded_file, sep=None, engine="python", encoding="utf-8-sig")
        except Exception:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep=";", engine="python", encoding="utf-8-sig")

        if df.shape[1] == 1:
            primera_columna = str(df.columns[0])
            if ";" in primera_columna:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep=";", engine="python", encoding="utf-8-sig")

        df.columns = normalizar_columnas(df.columns)
        return convertir_numericas_posibles(df)

    if nombre.endswith(".xlsx") or nombre.endswith(".xls"):
        return leer_excel_con_encabezado_automatico(uploaded_file)

    raise ValueError("Formato no soportado. Usa CSV o Excel.")


def graficar_serie(data, pronosticos=None):
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(data["periodo"], data["valor"], marker="o", label="Datos históricos")

    if pronosticos is not None:
        x_futuro = np.arange(data["periodo"].max() + 1, data["periodo"].max() + len(pronosticos) + 1)
        for nombre in pronosticos.columns:
            if nombre != "Periodo futuro":
                ax.plot(x_futuro, pronosticos[nombre], marker="o", linestyle="--", label=nombre)

    ax.set_xlabel("Periodo")
    ax.set_ylabel("Valor")
    ax.set_title("Serie histórica y pronósticos")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    st.pyplot(fig)


def graficar_residuos(fitted, residuos):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(fitted, residuos)
    ax.axhline(0, linestyle="--")
    ax.set_xlabel("Valores ajustados")
    ax.set_ylabel("Residuos")
    ax.set_title("Residuos vs valores ajustados")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)




def graficar_diagnostico_regresion_completo(resultado):
    """
    Muestra los 4 gráficos clásicos de diagnóstico de regresión.
    """
    modelo = resultado["modelo"]
    fitted = np.asarray(resultado["fitted"], dtype=float)
    residuos = np.asarray(resultado["residuos"], dtype=float)

    try:
        influencia = modelo.get_influence()
        residuos_std = influencia.resid_studentized_internal
        leverage = influencia.hat_matrix_diag
    except Exception:
        residuos_std = (residuos - np.nanmean(residuos)) / (np.nanstd(residuos) if np.nanstd(residuos) != 0 else 1)
        leverage = np.zeros_like(residuos_std)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    # 1. Residuals vs Fitted
    axes[0, 0].scatter(fitted, residuos, alpha=0.75)
    axes[0, 0].axhline(0, linestyle="--")
    axes[0, 0].set_title("Residuals vs Fitted")
    axes[0, 0].set_xlabel("Valores ajustados")
    axes[0, 0].set_ylabel("Residuos")
    axes[0, 0].grid(True, alpha=0.25)

    # 2. Q-Q Residuals
    stats.probplot(residuos_std, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title("Q-Q Residuals")
    axes[0, 1].grid(True, alpha=0.25)

    # 3. Scale-Location
    axes[1, 0].scatter(fitted, np.sqrt(np.abs(residuos_std)), alpha=0.75)
    axes[1, 0].set_title("Scale-Location")
    axes[1, 0].set_xlabel("Valores ajustados")
    axes[1, 0].set_ylabel("√|residuos estandarizados|")
    axes[1, 0].grid(True, alpha=0.25)

    # 4. Residuals vs Leverage
    axes[1, 1].scatter(leverage, residuos_std, alpha=0.75)
    axes[1, 1].axhline(0, linestyle="--")
    axes[1, 1].set_title("Residuals vs Leverage")
    axes[1, 1].set_xlabel("Leverage")
    axes[1, 1].set_ylabel("Residuos estandarizados")
    axes[1, 1].grid(True, alpha=0.25)

    fig.tight_layout()
    st.pyplot(fig)


def graficar_scatter_presentacion(df, x_col, y_col):
    data = df[[x_col, y_col]].copy()
    data[x_col] = pd.to_numeric(data[x_col], errors="coerce")
    data[y_col] = pd.to_numeric(data[y_col], errors="coerce")
    data = data.dropna()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(data[x_col], data[y_col], alpha=0.75)

    if len(data) >= 2 and data[x_col].nunique() > 1:
        b, a = np.polyfit(data[x_col], data[y_col], 1)
        xs = np.linspace(data[x_col].min(), data[x_col].max(), 100)
        ax.plot(xs, a + b * xs)

    ax.set_title(f"Relación entre {x_col} y {y_col}")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.grid(True, alpha=0.25)
    st.pyplot(fig)


def graficar_boxplot_presentacion(df, cat_col, y_col):
    data = df[[cat_col, y_col]].copy().dropna()
    grupos = []
    etiquetas = []
    for valor, grupo in data.groupby(cat_col):
        y = pd.to_numeric(grupo[y_col], errors="coerce").dropna().values
        if len(y) > 0:
            grupos.append(y)
            etiquetas.append(str(valor))

    if not grupos:
        st.info("No hay datos suficientes para el boxplot.")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot(grupos, labels=etiquetas)
    ax.set_title(f"{y_col} según {cat_col}")
    ax.set_xlabel(cat_col)
    ax.set_ylabel(y_col)
    ax.grid(True, alpha=0.25)
    st.pyplot(fig)


def graficar_pares_presentacion(df, columnas):
    data = df[columnas].apply(pd.to_numeric, errors="coerce").dropna()
    if data.shape[1] < 2 or data.shape[0] < 3:
        st.info("No hay datos suficientes para el gráfico de pares.")
        return

    axes = pd.plotting.scatter_matrix(data, figsize=(10, 10), diagonal="kde")
    fig = axes[0, 0].get_figure()
    fig.suptitle("Gráfico de pares", y=1.02)
    st.pyplot(fig)


def graficar_descomposicion_serie(tabla_descomp):
    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(tabla_descomp["periodo"], tabla_descomp["observado"])
    axes[0].set_title("Serie original")

    axes[1].plot(tabla_descomp["periodo"], tabla_descomp["tendencia"])
    axes[1].set_title("Tendencia")

    axes[2].plot(tabla_descomp["periodo"], tabla_descomp["estacionalidad"])
    axes[2].set_title("Estacionalidad")

    axes[3].plot(tabla_descomp["periodo"], tabla_descomp["residuo"])
    axes[3].set_title("Residuo")
    axes[3].set_xlabel("Periodo")

    for ax in axes:
        ax.grid(True, alpha=0.25)

    fig.tight_layout()
    st.pyplot(fig)


def graficar_correlacion(corr):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr.values)
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.index)

    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center")

    ax.set_title("Matriz de correlación")
    fig.colorbar(im)
    st.pyplot(fig)


def seleccionar_columna_tiempo(df):
    tipos = detectar_tipos(df)
    if tipos["posibles_fechas"]:
        return tipos["posibles_fechas"][0]

    nombres = ["fecha", "date", "tiempo", "periodo", "año", "anio", "mes", "dia", "día"]
    for col in df.columns:
        col_limpia = str(col).lower()
        if any(palabra in col_limpia for palabra in nombres):
            return col

    return df.columns[0]


def seleccionar_variable_objetivo(numericas):
    """
    Sugiere la variable objetivo Y.
    Prioriza nombres típicos de lo que se quiere predecir.
    Si no encuentra coincidencias, usa la última columna numérica,
    porque en muchos ejercicios de clase la Y aparece al final.
    """
    prioridad = [
        # Variables objetivo típicas de los ejercicios del curso.
        # En BostonHousing, por ejemplo, la Y correcta suele ser medv y no b.
        "salario_actual",
        "medv",
        "valor_medio",
        "valor vivienda",
        "valor_vivienda",
        "tiempo ms",
        "tiempo",
        "respuesta",
        "latencia",
        "consumo_kwh",
        "consumo",
        "ventas",
        "demanda",
        "precio",
        "costo",
        "rendimiento",
        "nota",
        "puntaje",
        "ingresos",
        "valor"
    ]

    columnas_limpias = {col: str(col).lower().strip() for col in numericas}

    for palabra in prioridad:
        for col, nombre in columnas_limpias.items():
            if palabra in nombre:
                return col

    return numericas[-1] if numericas else None


def es_columna_control(col, serie=None):
    """
    Evita sugerir como X columnas que normalmente no explican el modelo:
    id, índice, intercepto, constante o columnas sin variación.
    """
    nombre = str(col).lower().strip()
    palabras_control = ["id", "index", "indice", "índice", "intercepto", "const", "codigo", "código"]

    if any(p == nombre or p in nombre for p in palabras_control):
        return True

    if serie is not None:
        try:
            if serie.nunique(dropna=True) <= 1:
                return True
        except Exception:
            pass

    return False


def mostrar_ayuda_breve_regresion(y_col, x_cols):
    """
    Ayuda corta y cerrada por defecto. No es una guía larga.
    """
    with st.expander("Ayuda breve", expanded=False):
        st.write(f"Y a predecir: `{y_col}`")
        st.write("X seleccionadas:", x_cols)
        st.write("Revisa en orden: correlación → modelo → VIF → predicción.")


def mostrar_diagnostico(df):
    resumen = diagnostico_basico(df)

    c1, c2, c3 = st.columns(3)
    c1.metric("Filas", resumen["filas"])
    c2.metric("Columnas", resumen["columnas"])
    c3.metric("Variables numéricas", len(resumen["numericas"]))

    st.write("**Columnas numéricas:**", resumen["numericas"])
    st.write("**Columnas categóricas/texto:**", resumen["categoricas"])
    st.write("**Posibles columnas de fecha:**", resumen["posibles_fechas"])

    if len(resumen["numericas"]) > 0:
        st.subheader("Estadística descriptiva")
        st.dataframe(df[resumen["numericas"]].describe().T, use_container_width=True)

    st.subheader("Recomendación")
    for rec in recomendacion_modulo(df):
        st.info(rec)

    return resumen






def sugerir_variables_tecnicas(df, resumen=None):
    """
    Sugiere columnas para cada requisito del examen sin ejecutar cálculos.
    Sirve como guía técnica rápida:
    - Series: tiempo + valor.
    - Regresión: Y numérica + X explicativas.
    - Clasificación: objetivo alto/bajo + predictoras.
    """
    if resumen is None:
        resumen = diagnostico_basico(df)

    numericas = resumen.get("numericas", [])
    posibles_fechas = resumen.get("posibles_fechas", [])
    columnas = df.columns.tolist()

    # Series de tiempo: eje X = tiempo/fecha/periodo; eje Y = valor numérico.
    col_tiempo = seleccionar_columna_tiempo(df) if len(columnas) > 0 else None
    col_valor = seleccionar_variable_objetivo(numericas) if len(numericas) > 0 else None

    # Regresión múltiple: Y = variable numérica objetivo; X = numéricas útiles, evitando id/constantes.
    y_reg = seleccionar_variable_objetivo(numericas) if len(numericas) > 0 else None
    x_reg = []
    for col in numericas:
        if col == y_reg:
            continue
        if es_columna_control(col, df[col]):
            continue
        x_reg.append(col)
    x_reg = x_reg[:8]

    # Clasificación/matriz: primero busca objetivo binario; si no hay, sugiere convertir una numérica a Alto/Bajo.
    binarias = []
    for col in columnas:
        try:
            if df[col].dropna().nunique() == 2 and not es_columna_control(col, df[col]):
                binarias.append(col)
        except Exception:
            pass

    if binarias:
        objetivo_clasif = binarias[0]
        forma_clasif = "binaria existente"
    else:
        objetivo_clasif = y_reg
        forma_clasif = "convertir numérica a Alto/Bajo usando mediana o promedio"

    x_clasif = []
    for col in numericas:
        if col == objetivo_clasif:
            continue
        if es_columna_control(col, df[col]):
            continue
        x_clasif.append(col)
    x_clasif = x_clasif[:8]

    return {
        "series_tiempo": col_tiempo,
        "series_valor": col_valor,
        "regresion_y": y_reg,
        "regresion_x": x_reg,
        "clasificacion_objetivo": objetivo_clasif,
        "clasificacion_forma": forma_clasif,
        "clasificacion_x": x_clasif,
        "posibles_fechas": posibles_fechas,
        "numericas": numericas,
    }


def mostrar_mapa_tecnico_examen(df, resumen=None, expanded=False):
    """
    Muestra una guía pequeña de qué columna elegir según el tipo de pregunta.
    No cambia los cálculos del sistema; solo orienta al usuario.
    """
    sugerencias = sugerir_variables_tecnicas(df, resumen)

    with st.expander("Mapa técnico: qué elegir según la data", expanded=expanded):
        st.caption("Guía rápida para no confundirte al elegir X, Y, tiempo y clase.")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**Series de tiempo / pronóstico**")
            st.write("Eje X / tiempo:", f"`{sugerencias['series_tiempo']}`" if sugerencias["series_tiempo"] is not None else "No detectado")
            st.write("Eje Y / valor:", f"`{sugerencias['series_valor']}`" if sugerencias["series_valor"] is not None else "No detectado")
            st.caption("Usa este módulo cuando tengas fecha, año, mes, día o periodo + una variable numérica.")

        with c2:
            st.markdown("**Regresión lineal múltiple**")
            st.write("Y numérica a predecir:", f"`{sugerencias['regresion_y']}`" if sugerencias["regresion_y"] is not None else "No detectada")
            st.write("X sugeridas:")
            if sugerencias["regresion_x"]:
                st.code(", ".join(map(str, sugerencias["regresion_x"])))
            else:
                st.caption("No hay suficientes X numéricas útiles.")
            st.caption("Y es lo que quieres predecir; X son las variables que ayudan a explicarla.")

        with c3:
            st.markdown("**Matriz de confusión**")
            st.write("Objetivo/clase:", f"`{sugerencias['clasificacion_objetivo']}`" if sugerencias["clasificacion_objetivo"] is not None else "No detectado")
            st.write("Regla:", sugerencias["clasificacion_forma"])
            st.write("Predictoras numéricas:")
            if sugerencias["clasificacion_x"]:
                st.code(", ".join(map(str, sugerencias["clasificacion_x"])))
            else:
                st.caption("No hay suficientes predictoras numéricas útiles.")
            st.caption("Si la variable es numérica, el sistema puede convertirla a Alto/Bajo.")

        st.info(
            "Recuerda: estas son sugerencias automáticas. Si el profesor indica una Y, una fecha o una clase específica, "
            "usa lo que indique el enunciado del examen."
        )

def resolver_series_automatico(df, horizonte=5, ventana=3, periodo_estacional=7, alpha=0.30):
    tipos = detectar_tipos(df)
    numericas = tipos["numericas"]

    if len(numericas) == 0:
        st.warning("No hay columnas numéricas para series de tiempo.")
        return

    col_tiempo = seleccionar_columna_tiempo(df)
    col_valor = seleccionar_variable_objetivo(numericas)

    st.subheader("📈 Resolución automática: Series de tiempo y pronósticos")
    st.write(f"**Columna de tiempo elegida:** `{col_tiempo}`")
    st.write(f"**Variable numérica elegida:** `{col_valor}`")

    serie = preparar_serie(df, col_tiempo, col_valor)

    st.write("**Serie preparada:**")
    st.dataframe(serie.head(20), use_container_width=True)

    tabla_pron, _, info_tendencia = generar_pronosticos(
        serie["valor"].values,
        horizonte=int(horizonte),
        ventana=int(ventana),
        periodo_estacional=int(periodo_estacional),
        alpha=float(alpha)
    )

    st.write("**Tabla de pronósticos:**")
    st.dataframe(tabla_pron, use_container_width=True)

    st.write("**Gráfico de la serie con pronósticos:**")
    graficar_serie(serie, tabla_pron)

    metricas = evaluar_pronosticos(
        serie["valor"].values,
        ventana=int(ventana),
        periodo_estacional=int(periodo_estacional),
        alpha=float(alpha)
    )

    st.write("**Exactitud de pronóstico:**")
    if len(metricas) > 0:
        st.dataframe(metricas, use_container_width=True)
        mejor = metricas.iloc[0]["Método"]
        st.success(f"Mejor método según RMSE: **{mejor}**")
    else:
        st.warning("No hay suficientes datos para comparar la exactitud con prueba interna.")

    st.write("**Tendencia lineal:**")
    st.code(f"ŷ = {info_tendencia['a_intercepto']:.4f} + {info_tendencia['b_pendiente']:.4f}x")

    st.write("**Conclusión:**")
    st.success(interpretar_serie(serie["valor"].values, info_tendencia, metricas))


def resolver_regresion_automatico(df):
    numericas = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numericas) < 2:
        st.warning("No hay suficientes variables numéricas para regresión múltiple.")
        return

    y_col = seleccionar_variable_objetivo(numericas)
    x_cols = [c for c in df.columns if c != y_col]
    x_cols = x_cols[: min(8, len(x_cols))]

    st.subheader("📉 Resolución automática: Regresión múltiple")
    st.write(f"**Variable dependiente Y elegida:** `{y_col}`")
    st.write("**Variables independientes X elegidas:**", x_cols)

    resultado = ajustar_regresion_multiple(df, y_col, x_cols)
    diag = resultado["diagnostico"]

    c1, c2, c3 = st.columns(3)
    c1.metric("R²", diag["r2"])
    c2.metric("R² ajustado", diag["r2_ajustado"])
    c3.metric("p-value modelo", diag["p_value_modelo"])

    st.write("**Ecuación estimada:**")
    st.code(construir_ecuacion(resultado["coeficientes"], y_col))

    st.write("**Coeficientes:**")
    st.dataframe(resultado["coeficientes"], use_container_width=True)

    st.write("**Matriz de correlación:**")
    cols_corr = [y_col] + [x for x in x_cols if x in numericas]
    if len(cols_corr) >= 2:
        corr = df[cols_corr].corr(numeric_only=True)
        st.dataframe(corr, use_container_width=True)
        graficar_correlacion(corr)

    st.write("**Multicolinealidad VIF:**")
    if len(resultado["vif"]) > 0:
        st.dataframe(resultado["vif"], use_container_width=True)
    else:
        st.info("No se pudo calcular VIF.")

    st.write("**Diagnóstico de supuestos:**")
    st.write(f"Normalidad de residuos, Shapiro p-value: {diag['shapiro_p_residuos']}")
    st.write(f"Homocedasticidad, Breusch-Pagan p-value: {diag['breusch_pagan_p']}")
    st.write(f"Independencia, Durbin-Watson: {diag['durbin_watson']}")
    graficar_residuos(resultado["fitted"], resultado["residuos"])

    st.write("**Conclusión:**")
    st.success(interpretar_regresion(resultado, y_col))


def resolver_matriz_automatica(df):
    columnas = df.columns.tolist()
    numericas = df.select_dtypes(include=[np.number]).columns.tolist()

    binarias = []
    for col in columnas:
        valores = df[col].dropna().unique()
        if len(valores) == 2:
            binarias.append(col)

    st.subheader("✅ Resolución automática: Matriz de confusión")

    try:
        if len(binarias) >= 1:
            objetivo = binarias[0]
            x_cols = [c for c in columnas if c != objetivo]
            st.write(f"**Variable objetivo binaria elegida:** `{objetivo}`")
            st.write("**Variables predictoras elegidas:**", x_cols[:8])
            resultado = entrenar_logistica_para_matriz(df, objetivo, x_cols[:8], "mediana")
        elif len(numericas) >= 2:
            objetivo = seleccionar_variable_objetivo(numericas)
            x_cols = [c for c in columnas if c != objetivo]
            st.write(f"**Variable convertida a alto/bajo:** `{objetivo}`")
            st.write("**Variables predictoras elegidas:**", x_cols[:8])
            resultado = entrenar_logistica_para_matriz(df, objetivo, x_cols[:8], "mediana")
        else:
            st.warning("No hay suficientes columnas para construir una matriz de confusión automática.")
            return

        st.info(resultado["explicacion_objetivo"])

        st.write("**Matriz de confusión:**")
        st.dataframe(resultado["matriz"], use_container_width=True)

        st.write("**Métricas:**")
        st.dataframe(pd.DataFrame([resultado["metricas"]]), use_container_width=True)

        st.write("**Interpretación:**")
        st.success(interpretar_matriz(resultado["metricas"]))

    except Exception as e:
        st.warning(f"No se pudo resolver matriz de confusión automáticamente: {e}")


# ============================================================
# BARRA LATERAL
# ============================================================

st.sidebar.header("Carga y modo")

archivo = st.sidebar.file_uploader(
    "Sube tu archivo CSV o Excel",
    type=["csv", "xlsx", "xls"]
)

modulo = st.sidebar.radio(
    "Selecciona el modo",
    [
        "Calculadora automática",
        "Diagnóstico automático",
        "Series de tiempo y pronósticos",
        "Regresión lineal múltiple",
        "Clasificación y matriz de confusión",
        "Guía rápida para el examen",
    ]
)

if modulo in ["Calculadora automática", "Series de tiempo y pronósticos"]:
    st.sidebar.subheader("Parámetros rápidos")
    horizonte_auto = st.sidebar.number_input("Horizonte del pronóstico", min_value=1, max_value=100, value=5)
    ventana_auto = st.sidebar.number_input("Ventana media móvil", min_value=1, max_value=50, value=3)
    periodo_auto = st.sidebar.number_input("Periodo estacional", min_value=1, max_value=500, value=7)
    alpha_auto = st.sidebar.slider("Alpha suavizamiento", min_value=0.01, max_value=0.99, value=0.30)
else:
    horizonte_auto = 5
    ventana_auto = 3
    periodo_auto = 7
    alpha_auto = 0.30


# ============================================================
# SI NO HAY ARCHIVO
# ============================================================

if archivo is None:
    st.info("Sube una data en la barra lateral para que la calculadora empiece a resolver.")

    if modulo == "Guía rápida para el examen":
        st.header("📌 Guía rápida para elegir el método")
        st.markdown(
            """
            **Series de tiempo:** eje X = fecha/año/mes/día/periodo; eje Y = valor numérico a pronosticar.  
            **Regresión lineal múltiple:** Y = variable numérica a predecir; X = variables numéricas/categóricas que explican.  
            **Matriz de confusión:** clase real/predicha o conversión de variable numérica a Alto/Bajo.  
            **Calculadora automática:** intenta resolver lo que se pueda con la estructura de la data.
            """
        )

    st.stop()


# ============================================================
# CARGA Y LIMPIEZA
# ============================================================

try:
    df_original = leer_archivo(archivo)
    df, reporte_limpieza = limpiar_dataframe(df_original)
except Exception as e:
    st.error(f"No se pudo leer el archivo: {e}")
    st.stop()

st.success("Archivo cargado correctamente.")

with st.expander("Vista previa de la data", expanded=True):
    st.dataframe(df.head(20), use_container_width=True)

with st.expander("Reporte de limpieza automática", expanded=False):
    for item in reporte_limpieza:
        st.write(f"- {item}")

# Mapa pequeño para elegir columnas según lo que pida el examen.
# No ejecuta modelos; solo orienta: tiempo+valor, Y+X, alto/bajo.
resumen_mapa = diagnostico_basico(df)
mostrar_mapa_tecnico_examen(df, resumen_mapa, expanded=False)


# ============================================================
# MODO 0: CALCULADORA AUTOMÁTICA
# ============================================================

if modulo == "Calculadora automática":
    st.header("🧮 Resolución automática")
    st.write("El programa revisa la estructura de la data y resuelve los análisis que sí aplican.")

    resumen = mostrar_diagnostico(df)


    if len(resumen["posibles_fechas"]) >= 1 and len(resumen["numericas"]) >= 1:
        with st.expander("Resultado: Series de tiempo y pronósticos", expanded=True):
            try:
                resolver_series_automatico(df, horizonte_auto, ventana_auto, periodo_auto, alpha_auto)
            except Exception as e:
                st.error(f"No se pudo resolver series de tiempo: {e}")

    elif len(resumen["numericas"]) >= 1 and df.shape[0] >= 3:
        with st.expander("Resultado: Series de tiempo por orden de filas", expanded=True):
            try:
                df_temp = df.copy()
                df_temp["periodo_automatico"] = np.arange(1, len(df_temp) + 1)
                resolver_series_automatico(df_temp, horizonte_auto, ventana_auto, periodo_auto, alpha_auto)
            except Exception as e:
                st.error(f"No se pudo resolver series por orden de filas: {e}")

    if len(resumen["numericas"]) >= 2 and df.shape[0] >= 8:
        with st.expander("Resultado: Regresión múltiple", expanded=True):
            try:
                resolver_regresion_automatico(df)
            except Exception as e:
                st.error(f"No se pudo resolver regresión múltiple: {e}")

    if df.shape[0] >= 10 and len(df.columns) >= 2:
        with st.expander("Resultado: Clasificación y matriz de confusión", expanded=False):
            resolver_matriz_automatica(df)


# ============================================================
# MÓDULO 1: DIAGNÓSTICO AUTOMÁTICO
# ============================================================

elif modulo == "Diagnóstico automático":
    st.header("🔎 Diagnóstico automático")
    mostrar_diagnostico(df)


# ============================================================
# MÓDULO 2: SERIES DE TIEMPO
# ============================================================

elif modulo == "Series de tiempo y pronósticos":
    st.header("📈 Series de tiempo y pronósticos")
    st.write("Analiza una variable ordenada en el tiempo y genera pronósticos simples.")

    columnas = df.columns.tolist()
    numericas = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numericas) == 0:
        st.error("No se encontraron columnas numéricas para pronosticar.")
        st.stop()

    col_tiempo_sugerida = seleccionar_columna_tiempo(df)
    col_valor_sugerida = seleccionar_variable_objetivo(numericas)

    col_tiempo = st.selectbox(
        "Columna de tiempo, fecha, año o periodo",
        columnas,
        index=columnas.index(col_tiempo_sugerida) if col_tiempo_sugerida in columnas else 0
    )
    col_valor = st.selectbox(
        "Variable numérica a pronosticar",
        numericas,
        index=numericas.index(col_valor_sugerida) if col_valor_sugerida in numericas else 0
    )

    with st.expander("Ayuda breve", expanded=False):
        st.write(f"Eje X / tiempo seleccionado: `{col_tiempo}`")
        st.write(f"Eje Y / valor a pronosticar: `{col_valor}`")
        st.write("El horizonte se interpreta como cantidad de periodos futuros. Ejemplo: si la data es mensual, 3 = trimestre, 6 = semestre y 12 = año.")
        st.write("Revisa en orden: serie preparada → pronósticos → comparación → tendencia → descomposición.")

    if st.button("Resolver series de tiempo", type="primary"):
        st.session_state["serie_col_tiempo"] = col_tiempo
        st.session_state["serie_col_valor"] = col_valor
        st.session_state["serie_calcular"] = True

    if not st.session_state.get("serie_calcular", False):
        st.info("Selecciona la columna de tiempo, la variable numérica y presiona el botón para resolver.")
        st.stop()

    col_tiempo = st.session_state.get("serie_col_tiempo", col_tiempo)
    col_valor = st.session_state.get("serie_col_valor", col_valor)

    try:
        serie = preparar_serie(df, col_tiempo, col_valor)
        tabla_pron, _, info_tendencia = generar_pronosticos(
            serie["valor"].values,
            horizonte=int(horizonte_auto),
            ventana=int(ventana_auto),
            periodo_estacional=int(periodo_auto),
            alpha=float(alpha_auto)
        )
        metricas = evaluar_pronosticos(
            serie["valor"].values,
            ventana=int(ventana_auto),
            periodo_estacional=int(periodo_auto),
            alpha=float(alpha_auto)
        )
    except Exception as e:
        st.error(f"Ocurrió un error en series de tiempo: {e}")
        st.stop()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Serie preparada",
        "Pronósticos",
        "Comparación",
        "Tendencia",
        "Descomposición"
    ])

    with tab1:
        st.subheader("Serie preparada")
        st.dataframe(serie.head(50), use_container_width=True)
        st.write("Gráfico de datos históricos")
        graficar_serie(serie)

    with tab2:
        st.subheader("Pronósticos generados")
        st.dataframe(tabla_pron, use_container_width=True)
        st.write("Gráfico de la serie con pronósticos")
        graficar_serie(serie, tabla_pron)

        with st.expander("Qué significa cada método", expanded=False):
            st.markdown(
                """
                - **Método ingenuo:** repite el último valor observado.
                - **Método de la media:** usa el promedio histórico.
                - **Media móvil:** usa el promedio de los últimos datos según la ventana.
                - **Deriva:** proyecta una línea entre el primer y último valor.
                - **Tendencia lineal:** ajusta una recta `ŷ = a + bx`.
                - **Ingenuo estacional:** repite el patrón del último ciclo estacional.
                """
            )

    with tab3:
        st.subheader("Comparación de métodos")
        if len(metricas) > 0:
            st.dataframe(metricas, use_container_width=True)
            mejor = metricas.iloc[0]["Método"]
            rmse = metricas.iloc[0]["RMSE"]
            st.success(f"Mejor método según RMSE: {mejor} con RMSE = {rmse}.")
        else:
            st.warning("No hay suficientes datos para evaluar la exactitud con separación entrenamiento/prueba.")

        st.write("Conclusión automática")
        st.success(interpretar_serie(serie["valor"].values, info_tendencia, metricas))

    with tab4:
        st.subheader("Proyección de tendencia")
        st.write("La tendencia lineal ajusta una recta para proyectar la serie hacia el futuro.")
        st.code(f"ŷ = {info_tendencia['a_intercepto']:.4f} + {info_tendencia['b_pendiente']:.4f}x")

        pendiente = info_tendencia["b_pendiente"]
        if pendiente > 0:
            st.success("La pendiente es positiva: la serie presenta tendencia creciente.")
        elif pendiente < 0:
            st.warning("La pendiente es negativa: la serie presenta tendencia decreciente.")
        else:
            st.info("La pendiente es cercana a cero: la serie parece estable.")

    with tab5:
        st.subheader("Descomposición de la serie")
        st.write("Separa la serie en: observado, tendencia, estacionalidad y residuo.")
        st.write(f"Periodo estacional usado: {int(periodo_auto)}")

        try:
            tabla_descomp = descomponer_serie(
                serie["valor"].values,
                periodo_estacional=int(periodo_auto),
                modelo="aditivo"
            )
            st.dataframe(tabla_descomp, use_container_width=True)
            graficar_descomposicion_serie(tabla_descomp)
            st.success(interpretar_descomposicion(tabla_descomp))
        except Exception as e:
            st.warning(f"No se pudo descomponer la serie: {e}")
            st.info("Prueba con un periodo estacional menor o usa una data con más observaciones.")


# ============================================================
# MÓDULO 3: REGRESIÓN MÚLTIPLE
# ============================================================

elif modulo == "Regresión lineal múltiple":
    st.header("Regresión Lineal Múltiple")
    st.write("Predice una variable numérica Y usando varias variables explicativas X.")

    numericas = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numericas) < 2:
        st.error("Se necesitan al menos dos variables numéricas para aplicar regresión lineal múltiple.")
        st.stop()

    with st.expander("Vista previa y resumen", expanded=False):
        st.dataframe(df.head(20), use_container_width=True)
        st.dataframe(df[numericas].describe().T, use_container_width=True)

    y_sugerida = seleccionar_variable_objetivo(numericas)

    y_col = st.selectbox(
        "Variable dependiente Y, es decir, lo que quieres predecir",
        numericas,
        index=numericas.index(y_sugerida) if y_sugerida in numericas else 0
    )

    posibles_x = [c for c in df.columns if c != y_col]
    x_sugeridas = [
        c for c in posibles_x
        if not es_columna_control(c, df[c])
    ]

    x_cols = st.multiselect(
        "Variables independientes X, es decir, las que ayudan a predecir",
        posibles_x,
        default=x_sugeridas[:min(5, len(x_sugeridas))]
    )

    mostrar_ayuda_breve_regresion(y_col, x_cols)

    if len(x_cols) == 0:
        st.warning("Selecciona al menos una variable independiente X.")
        st.stop()

    if st.button("Calcular regresión lineal múltiple", type="primary"):
        st.session_state["rlm_y_col"] = y_col
        st.session_state["rlm_x_cols"] = x_cols
        st.session_state["rlm_calcular"] = True

    if not st.session_state.get("rlm_calcular", False):
        st.info("Selecciona Y, selecciona X y presiona el botón para calcular.")
        st.stop()

    y_col = st.session_state.get("rlm_y_col", y_col)
    x_cols = st.session_state.get("rlm_x_cols", x_cols)

    try:
        resultado = ajustar_regresion_multiple(df, y_col, x_cols)
        diag = resultado["diagnostico"]
    except Exception as e:
        st.error(f"No se pudo ajustar el modelo: {e}")
        st.stop()

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Correlación",
        "Modelo",
        "Modelo optimizado",
        "Diagnóstico",
        "Presentación",
        "Predicción"
    ])

    with tab1:
        st.subheader("Matriz de correlación")
        cols_corr = [y_col] + [x for x in x_cols if x in numericas]

        if len(cols_corr) >= 2:
            corr = df[cols_corr].corr(numeric_only=True)
            st.dataframe(corr, use_container_width=True)
            graficar_correlacion(corr)

            st.write("Interpretación rápida")
            for col in cols_corr:
                if col == y_col:
                    continue
                valor_corr = corr.loc[y_col, col]
                if pd.isna(valor_corr):
                    continue
                fuerza = "alta" if abs(valor_corr) >= 0.70 else "moderada" if abs(valor_corr) >= 0.40 else "baja" if abs(valor_corr) >= 0.20 else "muy baja"
                direccion = "positiva" if valor_corr > 0 else "negativa"
                st.write(f"- `{col}` tiene correlación {fuerza} {direccion} con `{y_col}`: {valor_corr:.2f}")
        else:
            st.info("No hay suficientes variables numéricas seleccionadas para calcular correlación.")

    with tab2:
        st.subheader("Resultados del modelo")

        c1, c2, c3 = st.columns(3)
        c1.metric("R²", diag["r2"])
        c2.metric("R² ajustado", diag["r2_ajustado"])
        c3.metric("p-value modelo", diag["p_value_modelo"])

        st.write("Ecuación estimada")
        st.code(construir_ecuacion(resultado["coeficientes"], y_col))

        st.write("Coeficientes")
        st.dataframe(resultado["coeficientes"], use_container_width=True)

        st.write("Interpretación")
        st.success(interpretar_regresion(resultado, y_col))

    with tab3:
        st.subheader("Modelo optimizado")
        st.write("El modelo optimizado elimina variables con bajo aporte estadístico usando p-value.")

        try:
            resultado_opt = optimizar_regresion_por_pvalue(df, y_col, x_cols, alpha=0.05)
            diag_opt = resultado_opt["diagnostico"]

            c1, c2, c3 = st.columns(3)
            c1.metric("R² optimizado", diag_opt["r2"])
            c2.metric("R² ajustado optimizado", diag_opt["r2_ajustado"])
            c3.metric("p-value modelo", diag_opt["p_value_modelo"])

            st.write("Variables finales")
            st.write(resultado_opt.get("variables_finales", []))

            eliminadas = resultado_opt.get("variables_eliminadas", pd.DataFrame())
            if len(eliminadas) > 0:
                st.write("Variables eliminadas")
                st.dataframe(eliminadas, use_container_width=True)
            else:
                st.info("No se eliminaron variables con el criterio p-value > 0.05.")

            st.write("Ecuación optimizada")
            st.code(construir_ecuacion(resultado_opt["coeficientes"], y_col))

            st.write("Coeficientes del modelo optimizado")
            st.dataframe(resultado_opt["coeficientes"], use_container_width=True)

            st.write("Interpretación")
            st.success(interpretar_modelo_optimizado(resultado_opt, y_col))
        except Exception as e:
            st.warning(f"No se pudo generar el modelo optimizado: {e}")

    with tab4:
        st.subheader("Multicolinealidad y diagnóstico completo")

        st.write("VIF")
        if len(resultado["vif"]) > 0:
            st.dataframe(resultado["vif"], use_container_width=True)

            if resultado["vif"]["VIF"].replace([np.inf, -np.inf], np.nan).dropna().gt(10).any() or np.isinf(resultado["vif"]["VIF"]).any():
                st.warning("Hay posible multicolinealidad alta. Revisa variables muy relacionadas o repetidas.")
            else:
                st.success("No se observa multicolinealidad alta con el criterio VIF > 10.")
        else:
            st.info("No se pudo calcular VIF.")

        st.write("Supuestos básicos")
        st.write(f"Normalidad de residuos, Shapiro p-value: {diag['shapiro_p_residuos']}")
        st.write(f"Homocedasticidad, Breusch-Pagan p-value: {diag['breusch_pagan_p']}")
        st.write(f"Independencia, Durbin-Watson: {diag['durbin_watson']}")

        st.write("Gráficos clásicos de diagnóstico")
        graficar_diagnostico_regresion_completo(resultado)

    with tab5:
        st.subheader("Gráficos automáticos para presentación")

        numeric_x = [x for x in x_cols if x in numericas]
        if len(numeric_x) > 0:
            corr_presentacion = df[[y_col] + numeric_x].corr(numeric_only=True)[y_col].drop(labels=[y_col], errors="ignore")
            if len(corr_presentacion) > 0:
                x_principal = corr_presentacion.abs().idxmax()
                st.write(f"Relación principal detectada: `{x_principal}` vs `{y_col}`")
                graficar_scatter_presentacion(df, x_principal, y_col)

        cats = []
        for col in x_cols:
            try:
                n_unicos = df[col].nunique(dropna=True)
                if col not in numericas and n_unicos <= 10:
                    cats.append(col)
                elif col in numericas and n_unicos <= 5:
                    cats.append(col)
            except Exception:
                pass

        if cats:
            st.write("Boxplot automático")
            cat_col = st.selectbox("Variable para boxplot", cats)
            graficar_boxplot_presentacion(df, cat_col, y_col)
        else:
            st.info("No se encontraron variables categóricas o binarias adecuadas para boxplot automático.")

        st.write("Gráfico de pares")
        cols_pares = [y_col] + numeric_x[:3]
        if len(cols_pares) >= 2:
            graficar_pares_presentacion(df, cols_pares)
        else:
            st.info("No hay suficientes variables numéricas para el gráfico de pares.")

    with tab6:
        st.subheader("Predicción")
        st.write("Ingresa valores para las X y el programa estimará Y.")

        valores_nuevos = {}
        for col in x_cols:
            if col in numericas:
                valor_default = float(pd.to_numeric(df[col], errors="coerce").mean())
                valores_nuevos[col] = st.number_input(f"Valor para {col}", value=valor_default)
            else:
                opciones = df[col].dropna().astype(str).unique().tolist()
                valores_nuevos[col] = st.selectbox(f"Valor para {col}", opciones) if opciones else ""

        if st.button("Calcular predicción"):
            try:
                nuevo = pd.DataFrame([valores_nuevos])

                for col in nuevo.columns:
                    if col in numericas:
                        nuevo[col] = pd.to_numeric(nuevo[col], errors="coerce")

                nuevo_dummies = pd.get_dummies(nuevo, drop_first=True, dtype=float)
                nuevo_dummies = nuevo_dummies.reindex(columns=resultado["X"].columns, fill_value=0)

                nuevo_const = sm.add_constant(nuevo_dummies, has_constant="add")
                nuevo_const = nuevo_const.reindex(columns=resultado["X_const"].columns, fill_value=0)

                if "const" in nuevo_const.columns:
                    nuevo_const["const"] = 1.0

                prediccion = resultado["modelo"].predict(nuevo_const)[0]
                st.success(f"Predicción estimada de `{y_col}`: {prediccion:.4f}")

            except Exception as e:
                st.error(f"No se pudo realizar la predicción: {e}")


# ============================================================
# MÓDULO 4: CLASIFICACIÓN Y MATRIZ DE CONFUSIÓN
# ============================================================

elif modulo == "Clasificación y matriz de confusión":
    st.header("✅ Clasificación y matriz de confusión")
    st.info("Para cumplir la regla del examen: si la variable objetivo es numérica, el sistema puede convertirla a clase Alto/Bajo usando mediana o promedio antes de calcular VP, FP, FN, VN y métricas.")

    modo = st.radio(
        "Elige cómo calcular la matriz",
        [
            "Ya tengo columna real y columna predicha",
            "Crear modelo automático alto/bajo con regresión logística",
        ]
    )

    if modo == "Ya tengo columna real y columna predicha":
        col_real = st.selectbox("Columna real", df.columns)
        col_pred = st.selectbox("Columna predicha", df.columns)

        if st.button("Calcular matriz de confusión"):
            try:
                y_real = pd.factorize(df[col_real])[0] if df[col_real].dtype == "object" else df[col_real]
                y_pred = pd.factorize(df[col_pred])[0] if df[col_pred].dtype == "object" else df[col_pred]

                temp = pd.DataFrame({"real": y_real, "pred": y_pred}).dropna()
                matriz, metricas = calcular_matriz_metricas(temp["real"], temp["pred"])

                st.subheader("Matriz de confusión")
                st.dataframe(matriz, use_container_width=True)

                st.subheader("Métricas")
                st.dataframe(pd.DataFrame([metricas]), use_container_width=True)

                st.subheader("Interpretación")
                st.success(interpretar_matriz(metricas))

            except Exception as e:
                st.error(f"No se pudo calcular la matriz: {e}")

    else:
        columnas = df.columns.tolist()
        objetivo = st.selectbox("Variable objetivo para convertir/clasificar", columnas)
        x_cols = st.multiselect("Variables predictoras X", [c for c in columnas if c != objetivo])
        metodo_umbral = st.radio("Si la variable objetivo es numérica, convertir usando:", ["mediana", "promedio"])

        if st.button("Entrenar y calcular matriz"):
            if len(x_cols) == 0:
                st.warning("Selecciona al menos una variable predictora.")
                st.stop()

            try:
                resultado = entrenar_logistica_para_matriz(df, objetivo, x_cols, metodo_umbral)

                st.info(resultado["explicacion_objetivo"])

                st.subheader("Matriz de confusión")
                st.dataframe(resultado["matriz"], use_container_width=True)

                st.subheader("Métricas")
                st.dataframe(pd.DataFrame([resultado["metricas"]]), use_container_width=True)

                st.subheader("Interpretación")
                st.success(interpretar_matriz(resultado["metricas"]))

            except Exception as e:
                st.error(f"No se pudo entrenar el modelo de clasificación: {e}")


# ============================================================
# MÓDULO 5: GUÍA RÁPIDA
# ============================================================

elif modulo == "Guía rápida para el examen":
    st.header("🧠 Guía rápida para el examen")
    st.markdown(
        """
        ### 1. Series de tiempo
        Usa este módulo cuando tengas una columna de tiempo y una variable numérica.
        Calcula método ingenuo, media, media móvil, deriva, suavizamiento exponencial,
        tendencia lineal, ingenuo estacional y descomposición de la serie.

        ### 2. Regresión múltiple
        Usa este módulo cuando quieras explicar una variable numérica Y usando varias X.
        Revisa R², R² ajustado, p-value, coeficientes, modelo optimizado, VIF, diagnóstico y gráficos de presentación.

        ### 3. Matriz de confusión
        Usa este módulo cuando tengas clases real/predicha o cuando quieras convertir una variable numérica en alto/bajo.

        ### 4. Para el examen
        Lo más rápido es usar **Calculadora automática**. Luego, si el profesor pide un método específico,
        entra al módulo correspondiente y ajusta las columnas manualmente.
        """
    )
