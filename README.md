#Clínica Salud Plus — Agente de Atención al Paciente con IA

# Clínica Salud Plus — Agente de Atención al Paciente con IA

Agente conversacional de atención al paciente para una clínica ficticia, construido con **LangGraph** (arquitectura ReAct), **RAG** sobre **ChromaDB** y **memoria dual** (corto y largo plazo). Este repositorio corresponde a la **Evaluación Parcial N°3 (EP3)** — *Implementación de Observabilidad* — de la asignatura ISY0101, Ingeniería de Soluciones con IA.

> 📄 El informe técnico completo (hallazgos, métricas, recomendaciones) está en `EP3_Informe_Observabilidad_ClinicaSaludPlus_1.pdf`, en la raíz del repositorio.

---

## ¿Qué hace el agente?

El agente recibe una consulta en lenguaje natural y la clasifica automáticamente en una de cuatro categorías antes de responder:

| Tipo de consulta | Comportamiento |
|---|---|
| **INFORMATIVA** | Busca la respuesta en la base de conocimiento de la clínica (RAG) y responde solo con esa información. |
| **URGENCIA** | Prioriza la derivación inmediata al número de emergencia (131), sin necesidad de consultar el RAG. |
| **AGENDAMIENTO** | Entrega instrucciones para agendar hora (teléfono / app web de la clínica). |
| **DERIVACIÓN** | Deriva a recepción cuando la información solicitada no está disponible. |

El agente nunca inventa horarios, datos de médicos ni información médica que no esté en `data/info_clinica.txt`.

---

## Arquitectura

```
clinica-salud-plus-rag/
├── src/
│   └── agent_core.py        # Orquestación del agente (LangGraph ReAct) + integración del logger
├── tools/
│   ├── rag_tool.py           # Recuperación semántica sobre ChromaDB (con índice persistido en disco)
│   └── classification_tool.py# Clasificación de la consulta (INFORMATIVA / URGENCIA / AGENDAMIENTO / DERIVACIÓN)
├── memory/
│   └── long_term_memory.py   # Memoria semántica entre sesiones (ChromaDB)
├── observability/
│   ├── logger.py             # Logger de observabilidad (latencia, tipo de consulta, errores, anomalías)
│   └── logs/
│       └── agent_logs.jsonl  # Logs reales generados por las ejecuciones del agente (formato JSON Lines)
├── dashboard/
│   └── app.py                # Dashboard de monitoreo en Streamlit (lee agent_logs.jsonl)
├── data/
│   └── info_clinica.txt      # Base de conocimiento de la clínica (fuente del RAG)
├── tests/
│   └── test_agent.py         # Batería de 20 casos de prueba (informativas, urgencias, agendamiento, multietapa)
├── chroma_rag_store/         # Índice vectorial persistido en disco (se genera automáticamente, no editar a mano)
├── memory_store/             # Persistencia de la memoria de largo plazo (se genera automáticamente)
├── requirements.txt
└── .env                      # Variables de entorno (no subir credenciales reales a un repo público)
```

### Componente de observabilidad (EP3)

El módulo `observability/logger.py` se integra directamente en `src/agent_core.py` (función `consultar()`), y registra automáticamente, para cada interacción:

- **Latencia** de la consulta completa, vía un context manager (`trazar_consulta`).
- **Tipo de consulta** detectado (INFORMATIVA / URGENCIA / AGENDAMIENTO / DERIVACIÓN).
- **Uso de herramientas** (RAG, clasificador) en eventos `HERRAMIENTA_USADA`.
- **Errores reales**, con su traza completa, sin interrumpir el registro del log.
- **Anomalías** detectadas automáticamente (latencia crítica, urgencia detectada, error en agente, ausencia de herramientas usadas).

Todo se almacena en `observability/logs/agent_logs.jsonl` (formato JSON Lines), que alimenta directamente el dashboard.

---

## Cómo ejecutar el proyecto

### 1. Requisitos previos

- Python 3.10+
- Un **GitHub Token** con permiso de cuenta `Models: Read-only` ([GitHub Models](https://github.com/marketplace/models)), o alternativamente una **API Key de OpenAI**.

### 2. Clonar el repositorio e instalar dependencias

```bash
git clone https://github.com/Pipe-25a/Cl-nica-Salud-Plus-Eva_3.git
cd Cl-nica-Salud-Plus-Eva_3/clinica-salud-plus-rag
pip install -r requirements.txt
```

### 3. Configurar las variables de entorno

Crea un archivo `.env` en la carpeta `clinica-salud-plus-rag/` con:

```env
GITHUB_TOKEN=tu_github_token_aqui
GITHUB_BASE_URL=https://models.inference.ai.azure.com
```

(Si en cambio usas OpenAI directo, define `OPENAI_API_KEY` en lugar de `GITHUB_TOKEN`.)

### 4. Ejecutar el agente

**Modo demo (preguntas de ejemplo):**
```bash
python src/agent_core.py
```

**Batería completa de pruebas (20 casos, genera logs reales):**
```bash
python -m tests.test_agent
```

> ℹ️ La primera ejecución genera el índice vectorial del RAG y lo guarda en `chroma_rag_store/` (consume cuota de la API de embeddings solo esa vez). Las ejecuciones siguientes reutilizan ese índice desde disco.

### 5. Levantar el dashboard de observabilidad

```bash
streamlit run dashboard/app.py
```

Esto abre el navegador en `http://localhost:8501` con KPIs, latencia por consulta, distribución por tipo, anomalías, uso de herramientas y la tabla de errores registrados — todo leído en vivo desde `observability/logs/agent_logs.jsonl`. Si el archivo de logs no existe aún, el dashboard lo indica y no inventa datos de ejemplo.

---

## Limitaciones conocidas

- El plan gratuito de **GitHub Models** limita a 150 solicitudes por día; en sesiones de prueba extensas puede agotarse la cuota (ver informe, Hallazgo 3).
- El `score_threshold` del recuperador RAG puede descartar fragmentos relevantes en consultas atípicas (ver informe, Hallazgo 5).
- Los logs actuales no anonimizan el contenido de las preguntas; no usar este sistema con datos reales de pacientes sin antes implementar las recomendaciones de privacidad descritas en el informe (sección E).

---

## Stack técnico

LangGraph · LangChain · ChromaDB · OpenAI / GitHub Models (gpt-4o-mini, text-embedding-3-small) · Streamlit · Plotly · Pandas
