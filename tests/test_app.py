"""Contratos de la consola que no dependen del terminal ni del proveedor.

`obtener_agente` cachea el agente de la sesión. El agente y las tools leen el
cliente con el callable de la sesión en cada turno, así que el cliente no es
parte de la identidad de lo construido; el modo sí, porque queda grabado en el
orquestador. Reconstruir por cliente descartaría el `ChatOpenAI` y su limitador
de cuota sin cambiar ninguna respuesta.
"""
import pytest

from src import app as app_mod
from src import config


@pytest.fixture
def construidos(monkeypatch):
    """Aísla `obtener_agente` del repositorio, el motor y el proveedor."""
    agentes: list = []

    class AgenteFalso:
        def __init__(self, **argumentos):
            self.argumentos = argumentos
            agentes.append(self)

    monkeypatch.setattr(app_mod, "AgenteTecnoApoyo", AgenteFalso)
    monkeypatch.setattr(app_mod, "RepositorioCasos", lambda: object())
    monkeypatch.setattr(
        app_mod, "crear_herramientas",
        lambda repositorio, motor, confirmador, **opciones: [],
    )
    monkeypatch.setattr(app_mod, "engine", object())
    monkeypatch.setattr(app_mod, "memoria", object())
    # El agente y el modo son estado de módulo: cada test arranca sin agente.
    monkeypatch.setattr(app_mod, "_agente", None)
    monkeypatch.setattr(app_mod, "_agente_modo", None)
    monkeypatch.setattr(config, "AGENTE_MODO", "reactivo")
    monkeypatch.setattr(config, app_mod.CLAVE_CLIENTE_SESION, None,
                        raising=False)
    return agentes


def test_el_agente_se_reutiliza_cuando_solo_cambia_el_cliente(construidos):
    """m3: cambiar de cliente no descarta el modelo ni su limitador.

    El cliente se lee por turno con el mismo callable que reciben las tools, así
    que el objeto construido sirve para los dos clientes.
    """
    app_mod.establecer_cliente("María Fernanda Rojas")
    primero = app_mod.obtener_agente()

    app_mod.establecer_cliente("Ricardo")
    segundo = app_mod.obtener_agente()

    assert segundo is primero
    assert len(construidos) == 1
    # El lector viaja vivo: es la razón por la que el cambio de cliente no
    # necesita una reconstrucción.
    assert primero.argumentos["cliente_actual"] is app_mod.cliente_actual


def test_el_agente_se_reconstruye_cuando_cambia_el_modo(construidos, monkeypatch):
    """m3: el modo sí queda grabado en el orquestador construido."""
    primero = app_mod.obtener_agente()

    monkeypatch.setattr(config, "AGENTE_MODO", "planificado")
    segundo = app_mod.obtener_agente()

    assert segundo is not primero
    assert [agente.argumentos["modo"] for agente in construidos] == [
        "reactivo", "planificado",
    ]
