"""Contrato del motor RAG: abstención por umbral y no-llamada al LLM.

Ningún test de este módulo toca la red. `RAGEngine` se instancia con
`carga_indice=False` y se le inyectan un vector store y una cadena falsos:
así se ejercita la lógica real de `recuperar()` y `consultar()` sin Mistral,
sin FAISS y sin claves.
"""
from types import SimpleNamespace

import pytest

from src import config
from src.rag import RAGEngine

FRASE_ABSTENCION = (
    "No tengo información suficiente en la documentación de TecnoApoyo "
    "para responder eso."
)


class VectorStoreFalso:
    """Devuelve los pares (documento, score) que se le entreguen."""

    def __init__(self, resultados):
        self._resultados = resultados
        self.k_pedidos = []

    def similarity_search_with_relevance_scores(self, pregunta, k):
        self.k_pedidos.append(k)
        return self._resultados[:k]


class CadenaFalsa:
    """Cadena RetrievalQA falsa: registra las invocaciones recibidas."""

    def __init__(self, respuesta=None, excepcion=None):
        self.respuesta = respuesta
        self.excepcion = excepcion
        self.llamadas = []

    def invoke(self, entrada, config=None):
        self.llamadas.append({"entrada": entrada, "config": config})
        if self.excepcion is not None:
            raise self.excepcion
        return {"result": self.respuesta}


def documento(document_id="PROC-001", titulo="Plazo de diagnóstico"):
    return SimpleNamespace(
        page_content="El diagnóstico demora hasta 48 horas hábiles.",
        metadata={
            "document_id": document_id,
            "titulo": titulo,
            "tema": "procedimientos",
            "version": "1.0",
            "fecha": "2026-01-01",
        },
    )


@pytest.fixture
def motor(monkeypatch):
    """RAGEngine sin índice, sin red y con umbral fijo en 0.45."""
    # Sin clave real: el endpoint por defecto es un Ollama local y ninguna
    # prueba llega a emitir una petición HTTP.
    monkeypatch.setattr(config, "LLM_API_KEY", "clave-de-prueba")
    monkeypatch.setattr(config, "UMBRAL_ABSTENCION", 0.45)
    motor = RAGEngine(carga_indice=False)
    return motor


def test_abstiene_por_umbral_sin_llamar_al_llm(motor):
    motor.vector_store = VectorStoreFalso([(documento(), 0.20)])
    motor.qa = CadenaFalsa(respuesta="no debería usarse")

    resultado = motor.consultar("¿cuánto demora el diagnóstico?")

    assert resultado.abstencion is True
    assert resultado.respuesta == FRASE_ABSTENCION
    # El corazón del contrato: sin evidencia no se gasta una llamada al LLM.
    assert motor.qa.llamadas == []
    assert resultado.tokens_total == 0
    assert resultado.tokens_entrada == 0 and resultado.tokens_salida == 0
    assert resultado.error is None
    # Trazabilidad: el documento evaluado se informa aunque se abstenga.
    assert len(resultado.documentos_recuperados) == 1
    assert resultado.documentos_recuperados[0] == {
        "texto": "El diagnóstico demora hasta 48 horas hábiles.",
        "score": 0.2,
        "document_id": "PROC-001",
        "titulo": "Plazo de diagnóstico",
        "tema": "procedimientos",
        "version": "1.0",
        "fecha": "2026-01-01",
    }


def test_abstiene_si_la_recuperacion_viene_vacia(motor):
    motor.vector_store = VectorStoreFalso([])
    motor.qa = CadenaFalsa(respuesta="no debería usarse")

    resultado = motor.consultar("¿hay soporte los domingos?")

    assert resultado.abstencion is True
    assert resultado.respuesta == FRASE_ABSTENCION
    assert motor.qa.llamadas == []
    assert resultado.documentos_recuperados == []


def test_responde_cuando_supera_el_umbral(motor):
    motor.vector_store = VectorStoreFalso([(documento(), 0.81)])
    motor.qa = CadenaFalsa(respuesta="  El diagnóstico demora 48 horas hábiles.  ")

    resultado = motor.consultar("¿cuánto demora el diagnóstico?")

    assert resultado.abstencion is False
    assert resultado.respuesta == "El diagnóstico demora 48 horas hábiles."
    assert len(motor.qa.llamadas) == 1
    assert motor.qa.llamadas[0]["entrada"] == {"query": "¿cuánto demora el diagnóstico?"}
    assert resultado.error is None


def test_abstencion_declarada_por_el_modelo_se_marca_como_abstencion(motor):
    motor.vector_store = VectorStoreFalso([(documento(), 0.90)])
    motor.qa = CadenaFalsa(respuesta=FRASE_ABSTENCION)

    resultado = motor.consultar("¿reparan tablets?")

    assert resultado.abstencion is True
    assert resultado.respuesta == FRASE_ABSTENCION
    assert len(motor.qa.llamadas) == 1


def test_umbral_explicito_manda_sobre_la_configuracion(motor):
    motor.vector_store = VectorStoreFalso([(documento(), 0.50)])
    motor.qa = CadenaFalsa(respuesta="no debería usarse")

    resultado = motor.consultar("¿cuánto demora?", umbral_abstencion=0.90)

    assert resultado.abstencion is True
    assert motor.qa.llamadas == []


def test_fallo_del_proveedor_no_se_reporta_como_abstencion(motor):
    motor.vector_store = VectorStoreFalso([(documento(), 0.90)])
    motor.qa = CadenaFalsa(excepcion=RuntimeError("429 demasiadas solicitudes"))

    resultado = motor.consultar("¿cuánto demora el diagnóstico?")

    # Un fallo del proveedor es un error, no falta de evidencia: mezclarlos
    # falsearía las métricas de abstención de la evaluación.
    assert resultado.abstencion is False
    assert resultado.error == "RuntimeError: 429 demasiadas solicitudes"
    assert resultado.tokens_total == 0


def test_historial_registra_el_turno_con_abstencion(motor):
    motor.vector_store = VectorStoreFalso([(documento(), 0.10)])
    motor.qa = CadenaFalsa()

    motor.consultar("¿precio de formateo?")

    assert motor.historial == [{
        "pregunta": "¿precio de formateo?",
        "respuesta": FRASE_ABSTENCION,
        "abstencion": True,
        "error": None,
    }]


def test_top_k_por_defecto_sale_de_la_configuracion(motor):
    store = VectorStoreFalso([])
    motor.vector_store = store
    motor.qa = CadenaFalsa()

    motor.recuperar("cualquier cosa")

    assert store.k_pedidos == [config.TOP_K]
