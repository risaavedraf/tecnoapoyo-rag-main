"""Defaults de configuración sin `.env`: el proyecto ya no cae a un proveedor local.

Estos tests fijan la regla del cambio: sin variables de entorno, `src.config`
resuelve el endpoint y los modelos de Mistral, y no queda ningún valor heredado
de Ollama. Se recarga el módulo con `dotenv.load_dotenv` neutralizado porque el
`.env` del desarrollador (si existe) definiría las mismas variables y ocultaría
los valores por defecto.
"""
import importlib

import pytest

from src import config

# Variables que el módulo resuelve y que el `.env` puede definir.
VARIABLES = (
    "LLM_BASE_URL",
    "LLM_API_KEY",
    "LLM_MODELO",
    "LLM_MODEL",
    "EMBEDDING_MODELO",
)


@pytest.fixture
def config_sin_entorno(monkeypatch):
    """`src.config` recargado sin `.env` y sin variables del proceso."""
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: False)
    for nombre in VARIABLES:
        monkeypatch.delenv(nombre, raising=False)
    importlib.reload(config)
    try:
        yield config
    finally:
        # Restaurar entorno y recargar antes de soltar el monkeypatch: si se
        # recargara con el entorno limpio, el resto de la suite heredaría los
        # defaults y no la configuración real.
        monkeypatch.undo()
        importlib.reload(config)


def test_sin_entorno_los_defaults_apuntan_a_mistral(config_sin_entorno):
    assert config_sin_entorno.LLM_BASE_URL == "https://api.mistral.ai/v1"
    assert config_sin_entorno.LLM_MODELO == "ministral-8b-latest"
    assert config_sin_entorno.EMBEDDING_MODELO == "mistral-embed"


def test_sin_entorno_no_queda_ningun_fallback_local(config_sin_entorno):
    valores = (
        config_sin_entorno.LLM_BASE_URL,
        config_sin_entorno.LLM_API_KEY,
        config_sin_entorno.LLM_MODELO,
        config_sin_entorno.EMBEDDING_MODELO,
    )
    texto = " ".join(valores).lower()
    assert "ollama" not in texto
    assert "localhost" not in texto
    assert "11434" not in texto
    # Sin clave por defecto: el cliente falla al construirse en vez de mandar
    # una credencial inventada a un endpoint remoto.
    assert config_sin_entorno.LLM_API_KEY == ""
