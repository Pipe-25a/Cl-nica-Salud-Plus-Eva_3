"""
tools/rag_tool.py
Herramienta RAG de consulta semántica sobre la base de conocimiento de Clínica Salud Plus.
Recupera fragmentos relevantes desde ChromaDB usando embeddings de OpenAI.
"""
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.tools import tool


# NUEVO: carpeta donde se guarda el indice vectorial en DISCO (no solo en RAM).
# Antes, _build_vectorstore() creaba el vectorstore solo en memoria, asi que
# CADA VEZ que se ejecutaba el script (proceso nuevo de Python), se volvian a
# generar embeddings para todos los chunks del documento, gastando cuota de
# la API de embeddings solo para arrancar. Con persist_directory, Chroma
# guarda el indice en esta carpeta la primera vez, y en las siguientes
# ejecuciones simplemente lo carga desde disco sin llamar a la API.
_PERSIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "chroma_rag_store")


# ── Configuración de embeddings ─────────────────────────────────────────────
def _get_embeddings() -> OpenAIEmbeddings:
    """Inicializa el modelo de embeddings con GitHub Models / OpenAI."""
    token    = os.environ.get("GITHUB_TOKEN") or os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("GITHUB_BASE_URL", "https://models.inference.ai.azure.com")
    return OpenAIEmbeddings(
        api_key=token,
        base_url=base_url,
        model="text-embedding-3-small",
    )


# ── Construcción del vectorstore (singleton en módulo) ───────────────────────
_vectorstore: Chroma | None = None


def _build_vectorstore(data_path: str = "data/info_clinica.txt") -> Chroma:
    """
    Carga o construye el vectorstore ChromaDB de la clinica.

    NUEVO comportamiento (antes solo creaba en memoria):
      1. Si ya existe una instancia en memoria de este mismo proceso
         (_vectorstore no es None), la reutiliza -> 0 llamadas a la API.
      2. Si no, intenta abrir el indice ya GUARDADO EN DISCO en
         _PERSIST_DIR (de una ejecucion anterior) -> 0 llamadas a la API
         de embeddings, solo lee archivos locales.
      3. Si la carpeta de persistencia esta vacia (primera vez que se
         corre el proyecto), recien ahi carga el .txt, lo divide en
         chunks y genera embeddings nuevos, GUARDANDOLOS en disco para
         que la proxima ejecucion entre por el paso 2 y no gaste cuota.
    """
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    embeddings = _get_embeddings()

    # NUEVO: Paso 2 -> ¿ya existe un indice persistido de una corrida anterior?
    # Chroma guarda sus archivos internos (sqlite, parquet, etc.) dentro de
    # _PERSIST_DIR. Si la carpeta existe y tiene contenido, asumimos que el
    # indice ya fue construido antes y lo abrimos directamente sin volver
    # a llamar a la API de embeddings para generar nada.
    indice_ya_existe = os.path.isdir(_PERSIST_DIR) and len(os.listdir(_PERSIST_DIR)) > 0

    if indice_ya_existe:
        _vectorstore = Chroma(
            persist_directory=_PERSIST_DIR,
            embedding_function=embeddings,
        )
        return _vectorstore

    # Paso 3 -> primera vez: construir el indice desde el documento fuente.
    # 1. Cargar documento
    loader     = TextLoader(data_path, encoding="utf-8")
    documentos = loader.load()

    # 2. Dividir en chunks (IL1.3: texto con solapamiento para mantener contexto)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=60,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(documentos)

    # 3. Crear vectorstore y GUARDARLO EN DISCO (persist_directory) para que
    # la siguiente ejecucion del programa lo reutilice en vez de regenerarlo.
    _vectorstore = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=_PERSIST_DIR,
    )
    return _vectorstore


def get_retriever(top_k: int = 3):
    """Retorna el retriever configurado con top_k resultados."""
    vs = _build_vectorstore()
    return vs.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": top_k, "score_threshold": 0.3},
    )


# ── Herramienta LangChain ────────────────────────────────────────────────────
@tool
def buscar_informacion_clinica(consulta: str) -> str:
    """
    Busca información relevante sobre la Clínica Salud Plus.
    Úsala para responder preguntas sobre horarios, médicos, especialidades,
    preparación de exámenes, convenios, pagos y preguntas frecuentes.
    Devuelve los fragmentos más relevantes encontrados en la base de conocimiento.
    """
    retriever = get_retriever(top_k=3)
    docs = retriever.invoke(consulta)

    if not docs:
        return (
            "No se encontró información específica sobre esa consulta en la base "
            "de conocimiento de la clínica. Considera solicitar más detalles al paciente "
            "o derivar a la recepción llamando al 600 123 4567."
        )

    fragmentos = []
    for i, doc in enumerate(docs, 1):
        fragmentos.append(f"[Fragmento {i}]\n{doc.page_content.strip()}")

    return "\n\n".join(fragmentos)