"""
memory/long_term_memory.py
Memoria de largo plazo basada en ChromaDB semántico (IE4).
Almacena y recupera interacciones relevantes entre sesiones.
"""
import os
import uuid
from datetime import datetime

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


def _get_embeddings() -> OpenAIEmbeddings:
    token    = os.environ.get("GITHUB_TOKEN") or os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("GITHUB_BASE_URL", "https://models.inference.ai.azure.com")
    return OpenAIEmbeddings(
        api_key=token,
        base_url=base_url,
        model="text-embedding-3-small",
    )


class LongTermMemory:
    """
    Memoria semántica de largo plazo para el agente.
    Persiste interacciones relevantes entre sesiones usando ChromaDB.
    """

    def __init__(self, persist_dir: str = "./memory_store"):
        self.persist_dir = persist_dir
        self.embeddings  = _get_embeddings()
        self._store: Chroma | None = None

    def _get_store(self) -> Chroma:
        """Obtiene o crea el vectorstore de memoria persistente."""
        if self._store is None:
            self._store = Chroma(
                collection_name="long_term_memory",
                embedding_function=self.embeddings,
                persist_directory=self.persist_dir,
            )
        return self._store

    def guardar_interaccion(self, pregunta: str, respuesta: str, session_id: str = "default") -> None:
        """
        Guarda una interacción pregunta-respuesta en la memoria de largo plazo.
        Solo guarda interacciones con respuestas sustanciales (>50 chars).
        """
        if len(respuesta.strip()) < 50:
            return

        store    = self._get_store()
        contenido = f"Pregunta: {pregunta}\nRespuesta: {respuesta}"
        metadatos = {
            "session_id": session_id,
            "timestamp":  datetime.now().isoformat(),
            "tipo":       "interaccion",
        }
        store.add_texts(
            texts=[contenido],
            metadatas=[metadatos],
            ids=[str(uuid.uuid4())],
        )

    def recuperar_contexto_previo(self, consulta: str, top_k: int = 2) -> str:
        """
        Recupera las interacciones previas más similares semánticamente a la consulta actual.
        Devuelve un string formateado para inyectar como contexto histórico.
        """
        store = self._get_store()

        try:
            resultados = store.similarity_search_with_score(consulta, k=top_k)
        except Exception:
            return ""

        if not resultados:
            return ""

        # Filtrar por relevancia mínima (cosine similarity > 0.5)
        relevantes = [(doc, score) for doc, score in resultados if score < 0.7]  # Chroma usa distancia
        if not relevantes:
            return ""

        lineas = ["[Contexto de sesiones anteriores]"]
        for doc, _ in relevantes:
            lineas.append(doc.page_content)

        return "\n\n".join(lineas)
