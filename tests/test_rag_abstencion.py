"""Contrato del motor RAG: abstención por umbral y no-llamada al LLM.

Ningún test de este módulo toca la red. `RAGEngine` se instancia con
`carga_indice=False` y se le inyectan un vector store y una cadena falsos:
así se ejercita la lógica real de `recuperar()` y `consultar()` sin Mistral,
sin FAISS y sin claves.
"""
from types import SimpleNamespace

import pytest

from src import config
from src.rag import (
    ALIAS_FUERA_DE_ALCANCE,
    PRODUCTOS_FUERA_DE_ALCANCE,
    RAGEngine,
    menciona_producto_fuera_de_alcance,
)

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
    # Sin clave real: la configuración no trae una por defecto y el cliente no
    # se construye sin ella. Ninguna prueba de este módulo emite una petición
    # HTTP: la cadena y los vectores se sustituyen por dobles.
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


# --- S12: una sola definición de la frase canónica -----------------------
def test_la_frase_canonica_tiene_una_sola_definicion():
    """`rag.FRASE_ABSTENCION` es la fuente; prefijo, prompt y agente derivan.

    El literal de este módulo es la expectativa independiente: si alguien edita
    una sola copia —la respuesta del motor, el prefijo del guard, el prompt o la
    constante del agente— este test lo señala.
    """
    from src import agente, rag

    assert rag.FRASE_ABSTENCION == FRASE_ABSTENCION
    assert rag.FRASE_ABSTENCION in rag.PROMPT_FUNDAMENTADO
    # El prefijo del guard es la apertura de la frase, no otra redacción.
    assert rag.PREFIJO_ABSTENCION == "No tengo información suficiente"
    assert rag.FRASE_ABSTENCION.startswith(rag.PREFIJO_ABSTENCION)
    # `src.agente` no redeclara ninguna de las dos: importa las mismas cadenas.
    assert agente.FRASE_ABSTENCION is rag.FRASE_ABSTENCION
    assert agente.PREFIJO_ABSTENCION is rag.PREFIJO_ABSTENCION


def test_la_respuesta_por_umbral_es_la_frase_canonica(motor):
    from src import rag

    motor.vector_store = VectorStoreFalso([(documento(), 0.20)])
    motor.qa = CadenaFalsa()

    resultado = motor.consultar("¿cuánto demora el diagnóstico?")

    # La misma cadena que el agente reconoce como abstención: la respuesta del
    # motor no puede ser una frase que su guard no identifique.
    assert resultado.respuesta == rag.FRASE_ABSTENCION == FRASE_ABSTENCION


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

    # Producto cubierto: la abstención que se prueba aquí es la que declara el
    # modelo sobre evidencia recuperada, no la de la frontera de dominio.
    resultado = motor.consultar("¿reparan impresoras?")

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


# --- Frontera de dominio -------------------------------------------------
# La recuperación siempre devuelve los k fragmentos más parecidos, aunque la
# pregunta no esté cubierta. En la evaluación real, "¿Cuánto cuesta reparar un
# iPhone 15?" alcanzó 0.7499 y "¿Reparan tablets?" 0.7139 con un umbral de
# 0.45, y el corpus no menciona ninguno de los dos productos: ese score mide
# cercanía de vocabulario, no evidencia documental. Por eso el motor corta
# antes de recuperar y deja el veredicto en "sin evidencia".


@pytest.mark.parametrize("pregunta", [
    "¿Cuánto cuesta reparar un iPhone 15?",
    "¿Reparan tablets?",
    "¿REPARAN TABLETS?",
    "¿Arreglan smartphones?",
    "¿reparan consolas PlayStation?",
    "¿venden televisores?",
    "¿tienen proyector?",
    "¿reparan un smartwatch?",
])
def test_producto_fuera_de_alcance_no_tiene_evidencia(motor, pregunta):
    store = VectorStoreFalso([(documento(), 0.75)])
    motor.vector_store = store
    motor.qa = CadenaFalsa(respuesta="no debería usarse")

    assert motor.recuperar(pregunta) == []

    resultado = motor.consultar(pregunta)

    assert resultado.abstencion is True
    assert resultado.respuesta == FRASE_ABSTENCION
    assert resultado.documentos_recuperados == []
    assert motor.qa.llamadas == []
    # El corte es de dominio, no de score: el vector store ni se consulta, así
    # que ningún fragmento genérico puede reabrir la respuesta.
    assert store.k_pedidos == []


@pytest.mark.parametrize("pregunta", [
    "¿cuánto demora el diagnóstico de un notebook?",
    "¿reparan impresoras?",
    "¿el monitor tiene garantía?",
    "¿cuánto demora el diagnóstico?",
    "¿el computador queda activo durante la reparación?",
])
def test_pregunta_cubierta_no_dispara_la_frontera(motor, pregunta):
    motor.vector_store = VectorStoreFalso([(documento(), 0.81)])
    motor.qa = CadenaFalsa(respuesta="  El diagnóstico demora 48 horas hábiles.  ")

    resultado = motor.consultar(pregunta)

    assert resultado.abstencion is False
    assert resultado.respuesta == "El diagnóstico demora 48 horas hábiles."
    assert len(resultado.documentos_recuperados) == 1
    assert len(motor.qa.llamadas) == 1


# --- El teléfono es un canal de contacto, no un producto -----------------
# `PRODUCTOS_FUERA_DE_ALCANCE` enumera equipos que el corpus no documenta. El
# teléfono no es uno de ellos: el corpus lo documenta como canal de contacto
# (FAQ-001, FAQ-029) y el dataset de evaluación mide "¿En qué horario atienden
# por teléfono?" como consulta respondible. Mientras `telefono`/`telefonos`
# estuvieron entre los alias, el guard cortó esa pregunta legítima antes de
# recuperar y el motor se abstuvo sin haber buscado evidencia.


@pytest.mark.parametrize("pregunta", [
    "¿En qué horario atienden por teléfono?",
    "¿Atienden por teléfonos?",
])
def test_un_canal_de_contacto_no_dispara_la_frontera(motor, pregunta):
    store = VectorStoreFalso([(documento("FAQ-001", "Horario de atención"), 0.84)])
    motor.vector_store = store
    motor.qa = CadenaFalsa(respuesta="Atendemos de lunes a viernes, de 9:00 a 18:00.")

    assert menciona_producto_fuera_de_alcance(pregunta) is False

    resultado = motor.consultar(pregunta)

    assert resultado.abstencion is False
    assert resultado.respuesta == "Atendemos de lunes a viernes, de 9:00 a 18:00."
    assert len(resultado.documentos_recuperados) == 1
    # El guard no reemplaza a la recuperación: la pregunta llega al índice.
    assert store.k_pedidos == [config.TOP_K]
    assert len(motor.qa.llamadas) == 1


@pytest.mark.parametrize(("producto", "pregunta"), [
    ("tablet", "¿Reparan tablets?"),
    ("iphone", "¿Cuánto cuesta reparar un iPhone 15?"),
    ("celular", "¿Atienden celulares?"),
    ("consola", "¿Reparan consolas PlayStation?"),
    ("televisor", "¿Venden televisores?"),
    ("proyector", "¿Tienen proyector?"),
    ("smartwatch", "¿Reparan un smartwatch?"),
])
def test_cada_producto_declarado_sigue_fuera_de_alcance(motor, producto, pregunta):
    """Sacar el teléfono no afloja ningún otro producto declarado."""
    assert producto in PRODUCTOS_FUERA_DE_ALCANCE
    assert menciona_producto_fuera_de_alcance(pregunta) is True

    store = VectorStoreFalso([(documento(), 0.75)])
    motor.vector_store = store
    motor.qa = CadenaFalsa(respuesta="no debería usarse")

    assert motor.recuperar(pregunta) == []
    assert store.k_pedidos == []


def test_el_alias_de_telefono_ya_no_esta_declarado():
    """Regresión: el teléfono no vuelve a la lista y el resto de los alias sigue."""
    assert "telefono" not in ALIAS_FUERA_DE_ALCANCE
    assert "telefonos" not in ALIAS_FUERA_DE_ALCANCE
    assert PRODUCTOS_FUERA_DE_ALCANCE["celular"] == (
        "celular", "celulares", "smartphone", "smartphones"
    )
    assert set(PRODUCTOS_FUERA_DE_ALCANCE) == {
        "tablet", "iphone", "celular", "consola", "televisor", "proyector",
        "smartwatch",
    }


def test_top_k_por_defecto_sale_de_la_configuracion(motor):
    store = VectorStoreFalso([])
    motor.vector_store = store
    motor.qa = CadenaFalsa()

    motor.recuperar("cualquier cosa")

    assert store.k_pedidos == [config.TOP_K]


# --- Clasificación del modelo: el prefijo es la raíz, no la mención -------
@pytest.mark.parametrize("redaccion", [
    "No tengo información suficiente en la documentación disponible.",
    "No tengo información suficiente en la documentación del cliente.",
])
def test_una_redaccion_mas_corta_del_modelo_sigue_siendo_abstencion(
        motor, redaccion):
    """A1: la clasificación del motor no puede ser más estricta que la de `master`.

    El prefijo con que el motor etiqueta una respuesta ya emitida es la raíz de
    la frase canónica —«…en la documentación»—, no la mención completa del
    producto. Un modelo que corta ahí sigue declarando la misma abstención, y
    contarla como respuesta inventaría respuestas que no existen.
    """
    motor.vector_store = VectorStoreFalso([(documento(), 0.90)])
    motor.qa = CadenaFalsa(respuesta=redaccion)

    resultado = motor.consultar("¿reparan impresoras?")

    assert resultado.abstencion is True
    assert resultado.respuesta == redaccion
    assert len(motor.qa.llamadas) == 1
