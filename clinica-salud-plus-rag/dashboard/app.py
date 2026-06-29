"""
dashboard/app.py
Dashboard de observabilidad para el agente de Clínica Salud Plus.

Lee directamente el archivo de logs generado por observability/logger.py
(observability/logs/agent_logs.jsonl) y visualiza las métricas de
precisión, latencia, consistencia y anomalías exigidas en la EP3
(IE1, IE2, IE3, IE4, IE5).

Cómo ejecutar (desde la carpeta clinica-salud-plus-rag):
    streamlit run dashboard/app.py
"""
import json
import os

import pandas as pd
import plotly.express as px
import streamlit as st

# ── Configuración de la página ───────────────────────────────────────────────
st.set_page_config(
    page_title="Observabilidad — Clínica Salud Plus",
    page_icon="📊",
    layout="wide",
)

# Ruta del log: relativa a este archivo, para que funcione sin importar
# desde qué carpeta se invoque "streamlit run".
RUTA_LOG = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "observability", "logs", "agent_logs.jsonl"
)


# ── Carga y procesamiento de datos ───────────────────────────────────────────
@st.cache_data(ttl=30)
def cargar_logs(ruta_log: str):
    """Lee el .jsonl y separa los eventos CONSULTA_FIN y HERRAMIENTA_USADA."""
    consultas, herramientas = [], []
    if not os.path.exists(ruta_log):
        return consultas, herramientas

    with open(ruta_log, encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if not linea:
                continue
            try:
                registro = json.loads(linea)
            except json.JSONDecodeError:
                continue  # ignora líneas corruptas, no rompe el dashboard
            if registro.get("evento") == "CONSULTA_FIN":
                consultas.append(registro)
            elif registro.get("evento") == "HERRAMIENTA_USADA":
                herramientas.append(registro)
    return consultas, herramientas


def construir_dataframe(consultas: list) -> pd.DataFrame:
    df = pd.DataFrame(consultas)
    if df.empty:
        return df
    df["timestamp_fin"] = pd.to_datetime(df["timestamp_fin"])
    df["cantidad_anomalias"] = df["anomalias"].apply(len)
    df = df.sort_values("timestamp_fin").reset_index(drop=True)
    df["n_consulta"] = df.index + 1  # orden secuencial para el eje X de latencia
    return df


# ── Carga real de datos ───────────────────────────────────────────────────────
consultas, herramientas = cargar_logs(RUTA_LOG)
df = construir_dataframe(consultas)

st.title("📊 Observabilidad — Agente Clínica Salud Plus")
st.caption(
    "Datos leídos en vivo desde `observability/logs/agent_logs.jsonl`. "
    "Este dashboard no usa datos de ejemplo: si el archivo no existe o está vacío, "
    "se muestra un aviso en vez de gráficos simulados."
)

if df.empty:
    st.warning(
        "No se encontraron registros en el archivo de logs. "
        "Corre primero `python -m tests.test_agent` o `python src/agent_core.py` "
        "para generar datos, luego recarga esta página."
    )
    st.stop()


# ── KPIs principales ──────────────────────────────────────────────────────────
total_consultas = len(df)
tasa_exito = 100 * df["exitoso"].mean()
latencia_promedio = df["latencia_seg"].mean()
total_anomalias = int(df["cantidad_anomalias"].sum())

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total de consultas", total_consultas)
col2.metric("Tasa de éxito", f"{tasa_exito:.1f}%")
col3.metric("Latencia promedio", f"{latencia_promedio:.2f} s")
col4.metric("Anomalías detectadas", total_anomalias)

st.divider()


# ── Fila 1: latencia por consulta + distribución por tipo ───────────────────
col_a, col_b = st.columns([2, 1])

with col_a:
    st.subheader("Latencia por consulta (orden cronológico)")
    fig_latencia = px.bar(
        df,
        x="n_consulta",
        y="latencia_seg",
        color="tipo_consulta",
        hover_data=["pregunta", "exitoso"],
        labels={"n_consulta": "N° de consulta", "latencia_seg": "Latencia (s)", "tipo_consulta": "Tipo"},
    )
    fig_latencia.add_hline(
        y=df["latencia_seg"].mean(),
        line_dash="dash",
        annotation_text="Promedio",
        annotation_position="top left",
    )
    st.plotly_chart(fig_latencia, use_container_width=True)

with col_b:
    st.subheader("Distribución por tipo de consulta")
    conteo_tipos = df["tipo_consulta"].value_counts().reset_index()
    conteo_tipos.columns = ["tipo_consulta", "cantidad"]
    fig_pie = px.pie(conteo_tipos, names="tipo_consulta", values="cantidad", hole=0.4)
    st.plotly_chart(fig_pie, use_container_width=True)


# ── Fila 2: latencia promedio por tipo + anomalías ───────────────────────────
col_c, col_d = st.columns(2)

with col_c:
    st.subheader("Latencia promedio por tipo de consulta")
    latencia_por_tipo = df.groupby("tipo_consulta")["latencia_seg"].mean().reset_index()
    fig_lat_tipo = px.bar(
        latencia_por_tipo,
        x="tipo_consulta",
        y="latencia_seg",
        labels={"tipo_consulta": "Tipo de consulta", "latencia_seg": "Latencia promedio (s)"},
        color="tipo_consulta",
    )
    st.plotly_chart(fig_lat_tipo, use_container_width=True)

with col_d:
    st.subheader("Anomalías detectadas por tipo")
    anomalias_explotadas = df.explode("anomalias")["anomalias"].dropna()
    if anomalias_explotadas.empty:
        st.info("No se registraron anomalías en los datos actuales.")
    else:
        conteo_anomalias = anomalias_explotadas.value_counts().reset_index()
        conteo_anomalias.columns = ["anomalia", "cantidad"]
        fig_anomalias = px.bar(
            conteo_anomalias,
            x="anomalia",
            y="cantidad",
            labels={"anomalia": "Tipo de anomalía", "cantidad": "Ocurrencias"},
            color="anomalia",
        )
        st.plotly_chart(fig_anomalias, use_container_width=True)


st.divider()


# ── Fila 3: herramientas usadas ───────────────────────────────────────────────
st.subheader("Uso de herramientas (tools) por el agente")
if herramientas:
    df_tools = pd.DataFrame(herramientas)
    conteo_tools = df_tools["herramienta"].value_counts().reset_index()
    conteo_tools.columns = ["herramienta", "cantidad"]
    fig_tools = px.bar(
        conteo_tools,
        x="herramienta",
        y="cantidad",
        labels={"herramienta": "Herramienta", "cantidad": "Veces usada"},
    )
    st.plotly_chart(fig_tools, use_container_width=True)
else:
    st.info("No hay eventos de uso de herramientas registrados todavía.")


st.divider()


# ── Consultas con error (para IE3 / IE4: trazabilidad de fallos) ────────────
st.subheader("🔴 Consultas con error registrado")
df_errores = df[df["exitoso"] == False][["timestamp_fin", "session_id", "pregunta", "error"]]
if df_errores.empty:
    st.success("No se registraron errores en los datos actuales.")
else:
    st.dataframe(df_errores, use_container_width=True, hide_index=True)


st.divider()


# ── Tabla de detalle completa ─────────────────────────────────────────────────
st.subheader("Detalle de todas las consultas registradas")
columnas_mostrar = [
    "n_consulta", "timestamp_fin", "session_id", "tipo_consulta",
    "latencia_seg", "exitoso", "cantidad_anomalias", "pregunta", "respuesta_preview",
]
st.dataframe(df[columnas_mostrar], use_container_width=True, hide_index=True)

st.caption(
    f"Archivo de logs: `{os.path.abspath(RUTA_LOG)}` · "
    f"Última actualización de esta vista: cada 30 segundos (cache_data ttl=30)."
)
