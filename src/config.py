# -*- coding: utf-8 -*-
"""Configuración central del asistente TecnoApoyo.

Define rutas, parámetros del LLM y el punto de conexión (endpoint compatible
con OpenAI). Por defecto usa Ollama en local (gratuito y sin claves); basta
definir LLM_BASE_URL y LLM_API_KEY en .env para apuntar a un proveedor externo
(por ejemplo Mistral) manteniendo el mismo stack LangChain del ramo, porque
ambos exponen una API compatible con OpenAI.
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
# Endpoint compatible con OpenAI. Ollama expone su API en localhost:11434/v1
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")  # clave arbitraria para Ollama
# Se aceptan ambos nombres porque el material del curso (RA1/IL1.1) declara
# LLM_MODEL y el proyecto usa LLM_MODELO: antes, un .env con LLM_MODEL era
# ignorado en silencio y el motor arrancaba con llama3.2:3b. Prevalece
# LLM_MODELO; el valor por defecto corresponde a un Ollama local.
LLM_MODELO = (os.getenv("LLM_MODELO")
              or os.getenv("LLM_MODEL")
              or "llama3.2:3b")
# El índice FAISS de data/output/indice_faiss queda atado al modelo de
# embeddings con el que se construyó: cambiar este valor obliga a regenerar el
# índice ejecutando python scripts/build_index.py.
EMBEDDING_MODELO = os.getenv("EMBEDDING_MODELO", "nomic-embed-text")
TEMPERATURA = float(os.getenv("TEMPERATURA", "0.1"))

# --- Parámetros de chunking ----------------------------------------------
# Pruebas de chunking (se comparan y se elige la mejor por recuperación)
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