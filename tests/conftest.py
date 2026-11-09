"""Configuración común de la suite.

Los tests corren desde la raíz del proyecto con `python -m pytest -q`, pero
también desde otros cwd (IDE, tarea programada). La raíz se agrega al
sys.path para que `import src...` funcione en ambos casos.

Aquí viven además los dobles del modelo y del motor RAG y los ayudantes que
arman un turno. Son una sola definición para los tres módulos del agente y su
contrato es el superconjunto de lo que esos módulos ya observaban: ningún
módulo pierde una observación por dejar de tener su copia.
"""
import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

RAIZ_PROYECTO = Path(__file__).resolve().parents[1]
if str(RAIZ_PROYECTO) not in sys.path:
    sys.path.insert(0, str(RAIZ_PROYECTO))

from src import config  # noqa: E402 - la raíz debe entrar antes al sys.path
from src.agente import AgenteTecnoApoyo  # noqa: E402
from src.casos import RepositorioCasos  # noqa: E402
from src.herramientas import crear_herramientas  # noqa: E402


class MotorFalso:
    """Doble del motor RAG: devuelve los fragmentos dados y cuenta consultas."""

    def __init__(self, fragmentos=None):
        self.fragmentos = list(fragmentos or [])
        # Cuántas recuperaciones se pidieron de verdad: permite afirmar que una
        # herramienta no se volvió a ejecutar, no solo que el paso quedó en False.
        self.llamadas = 0

    def recuperar(self, pregunta, k=None):
        self.llamadas += 1
        return self.fragmentos[:k] if k else self.fragmentos


class ModeloFalso:
    """Doble de ChatOpenAI: devuelve el guion y recuerda lo recibido.

    Cada `invoke` guarda una instantánea de los mensajes recibidos. `mensajes`
    y `mensajes_recibidos` son el mismo registro con dos nombres porque los tres
    módulos observan uno u otro, y la instantánea —no la lista viva— es lo que
    hace observable qué recibió el modelo en cada turno.
    """

    def __init__(self, guion):
        self.guion = list(guion)
        self.herramientas_recibidas = []
        self.mensajes_recibidos = []
        self.mensajes = self.mensajes_recibidos

    def bind_tools(self, herramientas, **kwargs):
        self.herramientas_recibidas = list(herramientas)
        return self

    def invoke(self, mensajes, config=None):
        self.mensajes_recibidos.append(list(mensajes))
        if not self.guion:
            raise AssertionError("El guion del modelo falso se agotó")
        return self.guion.pop(0)


def llamada(nombre, argumentos, id_llamada="call_1"):
    return {"name": nombre, "args": argumentos, "id": id_llamada, "type": "tool_call"}


def mensaje_tool(*llamadas):
    return AIMessage(content="", tool_calls=list(llamadas))


def mensaje_final(texto):
    return AIMessage(content=texto)


@pytest.fixture
def repo(tmp_path):
    return RepositorioCasos(ruta=tmp_path / "casos.db")


@pytest.fixture
def modo_reactivo_para_los_guiones(monkeypatch):
    """Fija el bucle reactivo para los módulos cuyos guiones lo ejercitan.

    No es autouse: el módulo del modo planificado no la pide, y aplicarla a
    toda la suite le cambiaría el modo a los tests que no la declaran.
    """
    monkeypatch.setattr(config, "AGENTE_MODO", "reactivo")


def construir(repo, motor=None, guion=None, confirmador=None, max_pasos=6,
              modo=None, memoria=None):
    """Arma un agente con el doble del modelo y devuelve `(agente, modelo)`.

    `motor` es el doble del motor RAG; sin él se usa un `MotorFalso` sin
    fragmentos. `modo=None` deja que el agente siga `config.AGENTE_MODO` y
    `max_pasos=6` es el tope con el que los guiones se escribieron, que cada
    test puede bajar.
    """
    modelo = ModeloFalso(guion or [])
    herramientas = crear_herramientas(
        repo, MotorFalso() if motor is None else motor, confirmador
    )
    agente = AgenteTecnoApoyo(
        herramientas=herramientas, llm=modelo, max_pasos=max_pasos,
        modo=modo, memoria=memoria,
    )
    return agente, modelo
