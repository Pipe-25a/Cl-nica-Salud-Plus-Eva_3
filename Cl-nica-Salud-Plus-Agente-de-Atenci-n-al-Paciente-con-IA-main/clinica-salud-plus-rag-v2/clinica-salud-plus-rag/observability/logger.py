import json
import os
import time
import uuid
import traceback
from datetime import datetime
from contextlib import contextmanager
from typing import Optional

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
LOG_FILE =  os.path.join(LOG_DIR, "agent_logs.jsonl")

UMBRAL_LATENCIA_ALTA = 10
UMBRAL_LATENCIA_CRITICA = 20

class AgentLogger:
    """
    Logger estructurado para el agente de clinica Salud Plus.
    Guarda cada evento en formato JSONL.
    """

    def __init__(self, log_file: str = LOG_FILE):
        self.log_file = log_file
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)


    def _escribir(self, registro: dict) -> None:
        """escribe un registro JSON  en el archivo de logs."""
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(registro, ensure_ascii=False) + "\n")

    @contextmanager
    def trazar_consulta(self, pregunta:str,session_id:str = "default"):
        """registra toda trazabilidad de una consulta."""

        trace_id = str(uuid.uuid4())[:8]
        inicio = time.time()
        timestamp =datetime.now().isoformat()

        ctx = {
            "trace_id": trace_id,
            "session_id": session_id,
            "pregunta": pregunta,
            "tipo_consulta": None,
            "respuesta": None,
            "herramientas": [],
            "error": None
        }

        self._escribir({
            "evento":   "CONSULTA_INICIO",
            "trace_id": trace_id,
            "session_id": session_id,
            "timestamp": timestamp,
            "pregunta": pregunta
        })

        try:
            yield ctx
        except Exception as e:
            ctx["error"] = {
                "tipo": type(e).__name__,
                "mensaje":str(e),
                "detalle": traceback.format_exc()
            }
            raise
        finally:
            latencia = round(time.time() - inicio,3)
            anomalias = self._detectar_anomalias(latencia,ctx)

            self._escribir({
                "evento": "CONSULTA_FIN",
                "trace_id": ctx["trace_id"],
                "session_id": ctx["session_id"],
                "timestamp_fin": datetime.now().isoformat(),
                "pregunta": ctx["pregunta"],
                "tipo_consulta": ctx["tipo_consulta"],
                "latencia_seg": latencia,
                "exitoso": ctx["error"] is None,
                "error": ctx["error"],
                "anomalias": anomalias,
                "respuesta_preview": (ctx["respuesta"] or "")[:300]
            })
    def log_herramienta(self, trace_id: str, nombre: str, input_data: str, output_data: str, duracion: float) -> None:
        """Registra el uso de una herramienta especifica dentro de una consulta."""
            
        self._escribir({
            "evento": "HERRAMIENTA_USADA",
            "trace_id": trace_id,
            "timestamp": datetime.now().isoformat(),
            "herramienta": nombre,
            "input": input_data[:200],
            "output": output_data[:400],
            "duracion_seg": round(duracion,3)
        })
    def _detectar_anomalias(self,latencia: float, ctx:dict) -> list:
        """Analiza una consulta finalizada y retorna una lista de anomalias"""

        anomalias = []

        if latencia >= UMBRAL_LATENCIA_CRITICA:
            anomalias.append("LATENCIA_CRITICA")
        elif latencia >= UMBRAL_LATENCIA_ALTA:
            anomalias.append("LATENCIA_ALTA")
        
        if ctx.get("error") is not None:
            anomalias.append("ERROR_EN_AGENTE")
        if not ctx.get("herramientas"):
            anomalias.append("SIN_HERRAMIENTAS")
        if ctx.get("tipo_consulta") == "URGENCIA":
            anomalias.append("URGENCIA_DETECTADA")
        return anomalias
    
    def leer_logs(self) -> list:
        """lee todos los registros CONSULTA_FIN del archivo."""
        if not os.path.exists(self.log_file):
            return []
        
        registros = []

        with open(self.log_file, "r", encoding="utf-8") as f:
            for linea in f:
                linea = linea.strip()
                
                if not linea:
                    continue
                try:
                    registro = json.loads(linea)
                    if registro.get("evento") == "CONSULTA_FIN":
                        registros.append(registro)
                except json.JSONDecodeError:
                    continue
        return registros
    
    def resumen_logs(self) -> dict:
        """Genera estadisticas de todos los logs almacenados."""

        registros = self.leer_logs()

        if not registros:
            return {"mensaje": "no hay logs registrados aun."}
        
        total = len(registros)
        exitosos = sum(1 for r in registros if r.get("exitoso", False))
        latencias = [r["latencia_seg"] for r in registros if "latencia_seg" in r]

        tipos = {}
        for r in registros:
            tipo = r.get("tipo_consulta") or "DESCONOCIDO"
            tipos[tipo] = tipos.get(tipo, 0) + 1

        anomalias_conteo = {}
        total_anomalias = 0

        for r in registros:
            for anomalia in r.get("anomalias", []):
                anomalias_conteo[anomalia] = anomalias_conteo.get(anomalia,0) + 1
                total_anomalias += 1
        return{
            "total_consultas": total,
            "tasa_exito": round(exitosos / total * 100,1),
            "latencia_promedio": round(sum(latencias) / len(latencias),2),
            "latencia_maxima": round(max(latencias),2),
            "latencia_minima": round(min(latencias),2),
            "tipos_consulta": tipos,
            "total_anomalias": total_anomalias,
            "anomalias_detalle": anomalias_conteo 
        }