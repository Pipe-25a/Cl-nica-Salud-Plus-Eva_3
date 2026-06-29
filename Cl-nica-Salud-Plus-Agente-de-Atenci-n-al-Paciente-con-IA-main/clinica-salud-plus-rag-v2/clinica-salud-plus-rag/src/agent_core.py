"""
src/agent_core.py
Agente principal de Clinica Salud Plus.
Implementa el esquema ReAct usando LangGraph (compatible con LangChain >= 1.0)
con memoria de corto plazo y largo plazo (ChromaDB semantico).
"""
import os
import sys
 
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
 
from dotenv import load_dotenv
load_dotenv()
 
token    = os.environ.get("GITHUB_TOKEN") or os.environ.get("OPENAI_API_KEY", "")
base_url = os.environ.get("GITHUB_BASE_URL", "https://models.inference.ai.azure.com")
 
os.environ["OPENAI_API_KEY"]  = token
os.environ["OPENAI_API_BASE"] = base_url
 
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
 
from tools.rag_tool import buscar_informacion_clinica
from tools.classification_tool import clasificar_consulta
from memory.long_term_memory import LongTermMemory

# NUEVO: AgentLogger registra cada consulta (inicio, fin, latencia, errores)
# y cada uso de herramienta (RAG, clasificador) en un archivo .jsonl,
# para poder medir observabilidad (IE1, IE2) y luego analizar logs (IE3, IE4).
from observability.logger import AgentLogger
 
 
SYSTEM_PROMPT = """Eres el asistente virtual de la Clinica Salud Plus.
Tu rol es ayudar a pacientes respondiendo sus consultas de forma precisa, empatica y confiable.
 
PROTOCOLO OBLIGATORIO — sigue estas etapas en orden:
1. Clasifica la consulta usando la herramienta clasificar_consulta.
2. Si el resultado es URGENCIA: deriva de inmediato al 131 o urgencias. No uses otras herramientas.
3. Si el resultado es AGENDAMIENTO o DERIVACION: entrega las instrucciones correspondientes.
4. Si el resultado es INFORMATIVA: usa buscar_informacion_clinica para recuperar informacion.
5. Formula una respuesta clara y humana basada SOLO en la informacion recuperada.
 
REGLAS CRITICAS:
- Nunca inventes informacion medica, horarios, ni datos de medicos.
- Si la informacion no esta disponible, indica que no tienes esa informacion y deriva a recepcion (600 123 4567).
- Ante cualquier sintoma de urgencia, prioriza la derivacion inmediata sobre todo lo demas.
- Responde siempre en espanol, con tono amable y profesional."""


# NUEVO: instancia unica del logger para todo el agente.
# IMPORTANTE: le damos una ruta CON carpeta (no solo un nombre de archivo),
# porque el logger internamente hace os.makedirs(carpeta_del_archivo) para
# asegurarse de que la carpeta exista. Si el nombre de archivo no tiene
# ninguna carpeta delante (ej: "agent_logs.jsonl" solo), os.path.dirname(...)
# devuelve "" y os.makedirs("") falla en Windows con FileNotFoundError.
# Por eso apuntamos explicitamente a "observability/logs/agent_logs.jsonl".
# Se crea UNA sola instancia (a nivel de modulo) para que todas las
# consultas y herramientas escriban en el mismo archivo de logs.
_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "observability", "logs", "agent_logs.jsonl")
logger = AgentLogger(log_file=_LOG_PATH)
 
 
def crear_agente(session_id: str = "default"):
    llm = ChatOpenAI(
        api_key=token,
        base_url=base_url,
        model="gpt-4o-mini",
        temperature=0,
    )
 
    tools = [clasificar_consulta, buscar_informacion_clinica]
 
    # Memoria de corto plazo (IE3): MemorySaver guarda historial por thread_id
    memoria_corto_plazo = MemorySaver()
 
    # Memoria de largo plazo (IE4): ChromaDB semantico
    memoria_largo_plazo = LongTermMemory(persist_dir="./memory_store")
 
    # Agente ReAct con LangGraph
    agente = create_react_agent(
        model=llm,
        tools=tools,
        checkpointer=memoria_corto_plazo,
        prompt=SYSTEM_PROMPT,
    )
 
    return agente, memoria_largo_plazo
 
 
def consultar(ejecutor, memoria_lp: LongTermMemory, pregunta: str, session_id: str = "default") -> str:
    # Recuperar contexto historico de sesiones anteriores
    historial = memoria_lp.recuperar_contexto_previo(pregunta)

    input_text = pregunta
    if historial:
        input_text = f"{pregunta}\n\n[Contexto previo: {historial}]"

    config = {"configurable": {"thread_id": session_id}}

    # NUEVO: "with logger.trazar_consulta(...)" abre un bloque que:
    #   - al ENTRAR: escribe en el .jsonl un evento "CONSULTA_INICIO" con timestamp
    #   - mientras tanto: nos entrega "ctx" (un diccionario) para que vayamos
    #     guardando datos (tipo de consulta, respuesta, herramientas usadas)
    #   - al SALIR (sea con exito o con error): calcula la latencia total,
    #     detecta anomalias (latencia alta, sin herramientas, error, urgencia)
    #     y escribe el evento "CONSULTA_FIN" con todo eso en el .jsonl.
    # Si algo lanza una excepcion dentro del "with", el logger la registra
    # como error en el log y luego la vuelve a lanzar (no la "esconde").
    with logger.trazar_consulta(pregunta, session_id=session_id) as ctx:

        # NUEVO: clasificamos la pregunta ANTES de invocar al agente,
        # llamando directamente a la misma herramienta que usa el agente
        # (clasificar_consulta). Esto es solo para que el log quede con
        # el tipo de consulta (URGENCIA / INFORMATIVA / AGENDAMIENTO / etc),
        # sin depender de que el agente decida usarla como tool call.
        # ".invoke({...})" es la forma estandar de ejecutar un @tool de LangChain.
        clasificacion = clasificar_consulta.invoke({"mensaje": pregunta})
        if "TIPO: URGENCIA" in clasificacion:
            ctx["tipo_consulta"] = "URGENCIA"
        elif "TIPO: AGENDAMIENTO" in clasificacion:
            ctx["tipo_consulta"] = "AGENDAMIENTO"
        elif "TIPO: DERIVACION" in clasificacion:
            ctx["tipo_consulta"] = "DERIVACION"
        else:
            ctx["tipo_consulta"] = "INFORMATIVA"

        # Llamada real al agente (sin cambios respecto a la version original)
        resultado = ejecutor.invoke(
            {"messages": [HumanMessage(content=input_text)]},
            config=config,
        )

        respuesta = resultado["messages"][-1].content

        # NUEVO: guardamos la respuesta final dentro de "ctx" para que el
        # logger la incluya (truncada a 300 caracteres) en el CONSULTA_FIN.
        ctx["respuesta"] = respuesta

        # NUEVO: revisamos los mensajes intermedios que devolvio LangGraph
        # para saber que herramientas uso el agente en esta consulta
        # (ej: "clasificar_consulta", "buscar_informacion_clinica").
        # Cada mensaje de tipo AIMessage puede traer "tool_calls" si el
        # agente decidio usar una herramienta en ese paso.
        herramientas_usadas = []
        for m in resultado["messages"]:
            tool_calls = getattr(m, "tool_calls", None)
            if tool_calls:
                herramientas_usadas.extend([tc["name"] for tc in tool_calls])
        ctx["herramientas"] = herramientas_usadas

        # NUEVO: por cada herramienta detectada, registramos un evento
        # "HERRAMIENTA_USADA" individual en el log. No tenemos el tiempo
        # exacto que tardo cada herramienta por separado (LangGraph no lo
        # expone aqui), asi que registramos duracion=0.0 como marcador;
        # si se quiere medir el tiempo real de cada tool, hay que instrumentar
        # directamente dentro de rag_tool.py y classification_tool.py.
        for nombre_tool in herramientas_usadas:
            logger.log_herramienta(
                trace_id=ctx["trace_id"],
                nombre=nombre_tool,
                input_data=pregunta,
                output_data=respuesta,
                duracion=0.0,
            )

    # Memoria de largo plazo se guarda fuera del bloque del logger,
    # porque no es parte de la "ejecucion del agente" en si misma.
    memoria_lp.guardar_interaccion(pregunta, respuesta, session_id=session_id)
    return respuesta
 
 
if __name__ == "__main__":
    print("=" * 60)
    print("  CLINICA SALUD PLUS - Asistente Virtual IA")
    print("  Agente LangGraph ReAct con Memoria Dual")
    print("=" * 60)
 
    ejecutor, memoria_lp = crear_agente(session_id="test_session")
 
    preguntas = [
        "Cuales son los horarios de atencion de la clinica?",
        "Como me preparo para un examen de sangre?",
        "Que especialidades medicas tienen disponibles?",
        "Tengo un dolor fuerte en el pecho, que hago?",
        "Puedo agendar una hora con el Dr. Carlos Rojas?",
        "Aceptan Fonasa?",
        "Cuanto demoran los resultados de los examenes?",
    ]
 
    for pregunta in preguntas:
        print(f"\n{'-'*55}")
        print(f"PACIENTE: {pregunta}")
        print("-" * 55)
        respuesta = consultar(ejecutor, memoria_lp, pregunta, session_id="test_session")
        print(f"AGENTE: {respuesta}")
 
    print(f"\n{'=' * 60}")
    print("  Pruebas completadas.")
    print("=" * 60)

    # NUEVO: al terminar las pruebas manuales de este archivo, mostramos
    # el resumen estadistico de TODOS los logs acumulados hasta ahora
    # (latencia promedio/maxima/minima, tasa de exito, tipos de consulta,
    # anomalias detectadas). Util para verificar rapido que el logger
    # esta funcionando, sin tener que abrir el .jsonl a mano.
    print("\n📊 RESUMEN DE OBSERVABILIDAD (logger.resumen_logs()):")
    import json as _json
    print(_json.dumps(logger.resumen_logs(), indent=2, ensure_ascii=False))