"""
tools/classification_tool.py
Herramienta de clasificación y razonamiento de consultas.
Analiza la intención del paciente y detecta situaciones de urgencia.
"""
from langchain.tools import tool

# Palabras clave que indican urgencia médica
PALABRAS_URGENCIA = [
    "dolor en el pecho", "dolor pecho", "infarto", "no puedo respirar",
    "dificultad para respirar", "falta de aire", "perdida de conciencia",
    "desmayo", "convulsion", "convulsiones", "sangrado severo",
    "hemorragia", "trauma craneal", "golpe en la cabeza", "accidente",
    "emergencia", "urgente", "urgencia", "muy grave", "crisis",
    "no reacciona", "sin pulso", "intoxicacion", "envenenamiento",
]


@tool
def clasificar_consulta(mensaje: str) -> str:
    """
    Clasifica el tipo de consulta del paciente para decidir cómo responder.
    Detecta situaciones de urgencia médica, consultas informativas estándar
    y solicitudes de derivación o agendamiento.
    Devuelve: URGENCIA, INFORMATIVA, AGENDAMIENTO, o DERIVACION.
    """
    mensaje_lower = mensaje.lower()

    # Detección de urgencia (máxima prioridad)
    for palabra in PALABRAS_URGENCIA:
        if palabra in mensaje_lower:
            return (
                "TIPO: URGENCIA\n"
                "ACCION REQUERIDA: Derivar INMEDIATAMENTE al servicio de urgencias. "
                "Indicar al paciente que llame al 131 o acuda de inmediato a urgencias "
                "(entrada lateral, calle Salud 200). NO continuar con consulta informativa."
            )

    # Detección de intención de agendar
    palabras_agendamiento = ["agendar", "reservar", "pedir hora", "solicitar hora", "quiero una hora"]
    if any(p in mensaje_lower for p in palabras_agendamiento):
        return (
            "TIPO: AGENDAMIENTO\n"
            "ACCION REQUERIDA: Indicar que para agendar horas el paciente debe llamar "
            "al 600 123 4567 o usar la aplicación web de la clínica."
        )

    # Detección de intención de derivación
    palabras_derivacion = ["hablar con", "comunicar con", "recepcion", "recepcionista", "persona humana"]
    if any(p in mensaje_lower for p in palabras_derivacion):
        return (
            "TIPO: DERIVACION\n"
            "ACCION REQUERIDA: Derivar al paciente a recepción llamando al 600 123 4567."
        )

    # Consulta informativa estándar
    return (
        "TIPO: INFORMATIVA\n"
        "ACCION REQUERIDA: Buscar información relevante en la base de conocimiento "
        "y responder directamente al paciente."
    )
