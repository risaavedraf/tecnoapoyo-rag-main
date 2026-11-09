# -*- coding: utf-8 -*-
"""Configuración central del asistente TecnoApoyo.

Define rutas, parámetros del LLM y el punto de conexión (endpoint compatible
con OpenAI). Los valores por defecto apuntan a la API de Mistral, el proveedor
de la corrida documentada; el endpoint y los modelos se pueden sustituir por
variables de entorno porque cualquier proveedor compatible con OpenAI usa el
mismo stack LangChain del ramo. La clave no tiene valor por defecto: hay que
definir LLM_API_KEY en .env.
"""
from pathlib import Path
import os

from dotenv import load_dotenv

RAIZ = Path(__file__).resolve().parents[1]

# Cargar variables de entorno desde .env (si existe)
load_dotenv(RAIZ / ".env")

# --- Rutas ---------------------------------------------------------------
DIR_CORPUS = RAIZ / "data" / "corpus"
DIR_PROCEDIMIENTOS = DIR_CORPUS / "procedimientos"
DIR_FAQ = DIR_CORPUS / "faq"
DIR_OUTPUT = RAIZ / "data" / "output"
RUTA_INDICE = DIR_OUTPUT / "indice_faiss"
RUTA_DOCUMENTOS = DIR_OUTPUT / "documentos.json"
RUTA_EVAL = RAIZ / "evaluation" / "dataset_evaluacion.json"
DIR_EVAL_RESULTADOS = RAIZ / "evaluation" / "results"

# --- Parámetros del LLM --------------------------------------------------
# Endpoint compatible con OpenAI. El valor por defecto es la API de Mistral, el
# proveedor de la corrida documentada; se puede sustituir por cualquier otro
# endpoint compatible con OpenAI sin tocar el código.
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.mistral.ai/v1")
# Sin clave por defecto: cada proveedor exige la suya y una credencial
# inventada solo produce un 401 diferido. Sin LLM_API_KEY el cliente falla al
# construirse, antes de emitir ninguna petición.
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
# Se aceptan ambos nombres porque el material del curso (RA1/IL1.1) declara
# LLM_MODEL y el proyecto usa LLM_MODELO: antes, un .env con LLM_MODEL era
# ignorado en silencio y el motor arrancaba con el modelo por defecto. Prevalece
# LLM_MODELO; el valor por defecto es el modelo de chat de la corrida
# documentada.
LLM_MODELO = (os.getenv("LLM_MODELO")
              or os.getenv("LLM_MODEL")
              or "ministral-8b-latest")
# El índice FAISS de data/output/indice_faiss queda atado al modelo de
# embeddings con el que se construyó: cambiar este valor obliga a regenerar el
# índice ejecutando python scripts/build_index.py.
EMBEDDING_MODELO = os.getenv("EMBEDDING_MODELO", "mistral-embed")
TEMPERATURA = float(os.getenv("TEMPERATURA", "0.1"))
# Tope de espera de una llamada al proveedor, en segundos. Sin este valor el
# cliente hereda el default del proveedor (600 s) y una consulta se queda
# minutos en silencio, sin forma de distinguir "lento" de "colgado".
LLM_TIMEOUT_SEG = float(os.getenv("LLM_TIMEOUT_SEG", "60"))
# Reintentos del cliente ante 429/5xx/timeout. Un valor negativo no tiene
# sentido y el cliente lo rechazaría, así que se acota a 0.
LLM_MAX_REINTENTOS = max(0, int(os.getenv("LLM_MAX_REINTENTOS", "1")))
# Respuesta en streaming por la consola: el texto final se escribe a medida que
# llega en lugar de esperar el panel completo. Solo 0/false/no/off lo apagan.
_LLM_STREAM = os.getenv("LLM_STREAM", "1").strip().lower()
LLM_STREAM = _LLM_STREAM not in ("0", "false", "no", "off")

# --- Parámetros de chunking ----------------------------------------------
# Configuraciones que `scripts/build_index.py` compara midiendo la forma de los
# trozos, no su recuperación: la elegida es la que deja `longitud_media` más
# cerca de su propio `chunk_size`.
CONFIGURACIONES_CHUNKING = [
    {"nombre": "chunk200-overlap20", "chunk_size": 200, "chunk_overlap": 20},
    {"nombre": "chunk350-overlap50", "chunk_size": 350, "chunk_overlap": 50},
    {"nombre": "chunk500-overlap75", "chunk_size": 500, "chunk_overlap": 75},
]

# Recuperación
TOP_K = int(os.getenv("TOP_K", "3"))
# Similitud mínima (score normalizado 0-1) para intentar responder; por debajo
# el motor se abstiene. Debe recalibrarse cada vez que cambia
# EMBEDDING_MODELO: cada modelo de embeddings produce su propia distribución de
# scores de relevancia, así que un umbral ajustado para uno no es necesariamente
# válido para otro.
UMBRAL_ABSTENCION = float(os.getenv("UMBRAL_ABSTENCION", "0.45"))

# --- Capa de agentes (RA2 / IL2.1-IL2.4) ---------------------------------
# Estas variables no cambian el comportamiento del motor RAG: solo alimentan
# los módulos de agente, casos, memoria y trazas. Los valores por defecto
# permiten arrancar sin tocar el .env existente.


def _ruta_desde_entorno(nombre: str, defecto: Path) -> Path:
    """Resuelve una ruta configurable contra la raíz del proyecto.

    Un valor relativo se interpreta respecto de RAIZ y no del directorio de
    trabajo: el proyecto se ejecuta como `python -m src.app` desde la raíz,
    pero también desde tests con otro cwd, y en ambos casos la ruta debe
    apuntar al mismo artefacto.
    """
    valor = os.getenv(nombre)
    if not valor:
        return defecto
    ruta = Path(valor).expanduser()
    return ruta if ruta.is_absolute() else (RAIZ / ruta)


# Tope duro de iteraciones del ciclo de herramientas (IL2.1). Un valor <= 0
# dejaría al agente sin poder ejecutar ninguna herramienta.
AGENTE_MAX_PASOS = max(1, int(os.getenv("AGENTE_MAX_PASOS", "6")))
# Modelo que decide las llamadas a herramientas. Se separa del modelo de
# redacción del RAG para poder usar uno más barato en la selección.
AGENTE_MODELO = os.getenv("AGENTE_MODELO") or LLM_MODELO
# Modelo que produce el plan en modo planificado. Se separa del modelo del
# agente para poder usar uno más capaz solo al planificar; vacío = AGENTE_MODELO.
PLAN_MODELO = os.getenv("PLAN_MODELO") or AGENTE_MODELO

# Estrategias de memoria de corto plazo (IL2.2): buffer completo, ventana de
# los últimos N turnos o resumen incremental.
ESTRATEGIAS_MEMORIA = ("buffer", "ventana", "resumen")
_estrategia = os.getenv("MEMORIA_ESTRATEGIA", "ventana").strip().lower()
# Un valor no reconocido no detiene la aplicación: se cae al default y queda
# marcado para que la interfaz pueda avisar en vez de fallar en silencio.
MEMORIA_ESTRATEGIA_INVALIDA = _estrategia not in ESTRATEGIAS_MEMORIA
MEMORIA_ESTRATEGIA = ("ventana" if MEMORIA_ESTRATEGIA_INVALIDA else _estrategia)
# Turnos conservados por la estrategia de ventana; mínimo 1 para no dejar el
# historial vacío con un valor accidental (0, negativo).
MEMORIA_VENTANA = max(1, int(os.getenv("MEMORIA_VENTANA", "6")))

# Artefactos de la capa de agente. Todo bajo data/output/, que está ignorado
# por git: son datos de ejecución, no entregables.
RUTA_CASOS = _ruta_desde_entorno("RUTA_CASOS", DIR_OUTPUT / "casos.db")
RUTA_MEMORIA = _ruta_desde_entorno("RUTA_MEMORIA", DIR_OUTPUT / "memoria.db")
RUTA_TRAZAS = _ruta_desde_entorno("RUTA_TRAZAS", DIR_OUTPUT / "trazas")
# Semilla versionada de casos de soporte (sí es entregable).
RUTA_CASOS_SEMILLA = RAIZ / "data" / "casos_semilla.json"

# --- Memoria: resumidor de la estrategia 'resumen' (IL2.2) ---------------
# 'llm' resume con el proveedor y cae al resumidor local si el LLM falla;
# 'local' usa siempre el fallback, útil sin red o sin cuota.
RESUMIDORES_MEMORIA = ("llm", "local")
_resumidor = os.getenv("MEMORIA_RESUMIDOR", "llm").strip().lower()
MEMORIA_RESUMIDOR_INVALIDO = _resumidor not in RESUMIDORES_MEMORIA
MEMORIA_RESUMIDOR = "llm" if MEMORIA_RESUMIDOR_INVALIDO else _resumidor

# --- Planificación (IL2.3) ------------------------------------------------
# Dos modos comparables: el bucle reactivo actual y el planificador.
MODOS_AGENTE = ("reactivo", "planificado")
_modo = os.getenv("AGENTE_MODO", "planificado").strip().lower()
AGENTE_MODO_INVALIDO = _modo not in MODOS_AGENTE
AGENTE_MODO = "planificado" if AGENTE_MODO_INVALIDO else _modo
# Tope de pasos de un plan y tope de re-planificaciones por consulta.
PLAN_MAX_PASOS = max(1, int(os.getenv("PLAN_MAX_PASOS", "6")))
PLAN_MAX_REPLANES = max(0, int(os.getenv("PLAN_MAX_REPLANES", "1")))

