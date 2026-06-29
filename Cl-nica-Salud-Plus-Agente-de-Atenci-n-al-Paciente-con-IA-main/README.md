# 🏥 Clínica Salud Plus — Agente de Atención al Paciente con IA

> **Evaluación Parcial N°2 — ISY0101 Ingeniería de Soluciones con IA — DuocUC**  
> Integrantes: Felipe Pérez S. · Ignacio Naum F.
## Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/...
cd clinica-salud-plus-rag

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Configurar credenciales
cp .env.example .env
```

Sistema de atención al paciente basado en **LangChain Agents (ReAct)**, RAG semántico y memoria dual de corto y largo plazo.
## Requisitos
- Python 3.10 o superior
- GitHub Token o OpenAI API Key
- 
## Ejecución
# Ejecutar el agente con consultas de prueba
python src/agent_core.py
# Ejecutar la suite de 20 pruebas
python tests/test_agent.py

## Estructura del Proyecto
clinica-salud-plus-rag/
├── src/
│   └── agent_core.py           # Orquestador principal del agente ReAct
├── tools/
│   ├── rag_tool.py             # Herramienta de búsqueda semántica (RAG)
│   └── classification_tool.py # Herramienta de clasificación de intenciones
├── memory/
│   └── long_term_memory.py    # Memoria de largo plazo con ChromaDB
├── tests/
│   └── test_agent.py          # Suite de 20 pruebas
├── data/
│   └── info_clinica.txt       # Base de conocimiento de la clínica
├── .env.example
├── requirements.txt
└── README.md

## Diagrama de componetes

<img width="937" height="837" alt="Captura de pantalla 2026-05-30 172606" src="https://github.com/user-attachments/assets/93d3804f-0c87-4e7f-a13b-784eea0ed644" />

## Descripción de Módulos

| Módulo | Descripción |
|--------|-------------|
| `src/agent_core.py` | Crea el agente ReAct, configura la memoria dual y expone la función `consultar()` que orquesta el flujo completo por cada turno. |
| `tools/rag_tool.py` | Carga `info_clinica.txt`, divide en chunks (400 chars / 60 overlap), genera embeddings y busca los 3 fragmentos más relevantes en ChromaDB. |
| `tools/classification_tool.py` | Detecta urgencias médicas por palabras clave y clasifica la intención del paciente antes de cualquier búsqueda. |
| `memory/long_term_memory.py` | Persiste interacciones en ChromaDB en disco y las recupera por similitud semántica al inicio de cada nueva sesión. |
| `tests/test_agent.py` | 20 pruebas que cubren consultas informativas, urgencias, agendamiento y consultas multietapa. |

##Diagrama de Flujos
<img width="934" height="1294" alt="image" src="https://github.com/user-attachments/assets/7e25d203-d377-4544-bff3-6f0510e92340" />
