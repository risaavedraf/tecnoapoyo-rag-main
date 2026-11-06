"""Modo planificado del agente (RA2 / IL2.3): plan, síntesis y guard.

El modelo se reemplaza por un doble con guion: primero emite el JSON del plan,
después la síntesis. Sin red.
"""
import json

import pytest
from langchain_core.messages import AIMessage

from src import config
from src.agente import FRASE_ABSTENCION, AgenteTecnoApoyo
from src.casos import RepositorioCasos
from src.herramientas import crear_herramientas


class MotorFalso:
    def __init__(self, fragmentos=None):
        self.fragmentos = fragmentos or []

    def recuperar(self, pregunta, k=None):
        return self.fragmentos[:k] if k else self.fragmentos


class ModeloFalso:
    def __init__(self, guion):
        self.guion = list(guion)
        self.mensajes = []

    def bind_tools(self, herramientas, **kwargs):
        return self

    def invoke(self, mensajes, config=None):
        self.mensajes.append(mensajes)
        if not self.guion:
            raise AssertionError("El guion del modelo falso se agotó")
        return self.guion.pop(0)


def plan_json(pasos, objetivo="resolver la consulta"):
    return AIMessage(content=json.dumps(
        {"objetivo": objetivo, "pasos": pasos, "version": 1}, ensure_ascii=False
    ))


def paso(id_paso, herramienta, argumentos, depende_de=None):
    return {"id": id_paso, "herramienta": herramienta,
            "argumentos": argumentos, "depende_de": depende_de or []}


@pytest.fixture
def repo(tmp_path):
    return RepositorioCasos(ruta=tmp_path / "casos.db")


FRAGMENTO = {
    "texto": "El diagnóstico demora hasta 48 horas hábiles.",
    "score": 0.88,
    "document_id": "PROC-004",
    "titulo": "Política de presupuestos y diagnósticos",
    "tema": "procedimientos",
    "version": "1.0",
    "fecha": "2026-01-01",
}


def construir(repo, motor, guion, modo="planificado", confirmador=None):
    modelo = ModeloFalso(guion)
    agente = AgenteTecnoApoyo(
        herramientas=crear_herramientas(repo, motor, confirmador),
        llm=modelo,
        modo=modo,
    )
    return agente, modelo


def test_el_modo_por_defecto_es_reactivo(repo):
    modelo = ModeloFalso([AIMessage(content="Hola.")])
    agente = AgenteTecnoApoyo(
        herramientas=crear_herramientas(repo, MotorFalso()), llm=modelo)
    assert agente.modo == config.AGENTE_MODO
    assert agente.modo == "reactivo"


def test_un_modo_desconocido_cae_a_reactivo(repo):
    agente, _ = construir(repo, MotorFalso(), [AIMessage(content="Hola.")],
                          modo="telepático")
    assert agente.modo == "reactivo"


def test_modo_planificado_ejecuta_el_plan_y_sintetiza(repo):
    agente, _ = construir(repo, MotorFalso([dict(FRAGMENTO)]), [
        plan_json([paso("p1", "buscar_documentacion",
                        {"pregunta": "¿cuánto demora el diagnóstico?"})]),
        AIMessage(content="El diagnóstico demora hasta 48 horas hábiles."),
    ])
    respuesta = agente.consultar("¿cuánto demora el diagnóstico?")
    assert respuesta.error is None
    assert respuesta.abstencion is False
    assert respuesta.respuesta == "El diagnóstico demora hasta 48 horas hábiles."
    assert respuesta.plan is not None
    assert [paso_plan.id for paso_plan in respuesta.plan.pasos] == ["p1"]
    assert len(respuesta.pasos) == 1
    assert respuesta.pasos[0].herramienta == "buscar_documentacion"
    assert respuesta.replanes == 0


def test_modo_planificado_encadena_placeholder(repo):
    # p1 lista casos y p2 consume el id real del primer caso por placeholder.
    agente, _ = construir(repo, MotorFalso([dict(FRAGMENTO)]), [
        plan_json([
            paso("p1", "listar_casos", {"limite": 1}),
            paso("p2", "consultar_caso",
                 {"id_caso": "{{p1.datos.casos.0.id}}"}, depende_de=["p1"]),
        ]),
        AIMessage(content="Tu caso está cerrado."),
    ])
    respuesta = agente.consultar("¿en qué estado está mi caso?")
    assert respuesta.error is None
    assert [p.herramienta for p in respuesta.pasos] == ["listar_casos", "consultar_caso"]
    assert all(p.ok for p in respuesta.pasos)
    assert respuesta.respuesta == "Tu caso está cerrado."


def test_abstencion_forzada_en_modo_planificado(repo):
    agente, _ = construir(repo, MotorFalso([]), [
        plan_json([paso("p1", "buscar_documentacion", {"pregunta": "¿reparan tablets?"})]),
        AIMessage(content="Sí, reparamos tablets por 30.000 pesos."),
    ])
    respuesta = agente.consultar("¿reparan tablets?")
    assert respuesta.abstencion is True
    assert respuesta.respuesta == FRASE_ABSTENCION


def test_un_plan_invalido_no_ejecuta_nada(repo):
    plan_malo = plan_json([paso("p1", "borrar_todo", {})])
    agente, _ = construir(repo, MotorFalso([dict(FRAGMENTO)]), [
        plan_malo, plan_malo,  # planificación + el único reintento
    ])
    respuesta = agente.consultar("borra todo")
    assert respuesta.pasos == []
    assert respuesta.respuesta == ""
    assert respuesta.error is not None
    assert respuesta.abstencion is False


def test_error_del_proveedor_en_modo_planificado_no_propaga(repo):
    class ModeloQueFalla:
        def bind_tools(self, herramientas, **kwargs):
            return self

        def invoke(self, mensajes, config=None):
            raise RuntimeError("429 demasiadas solicitudes")

    agente = AgenteTecnoApoyo(
        herramientas=crear_herramientas(repo, MotorFalso()),
        llm=ModeloQueFalla(), modo="planificado")
    respuesta = agente.consultar("¿cuánto demora?")
    assert respuesta.respuesta == ""
    assert respuesta.error is not None and "429" in respuesta.error
    assert respuesta.abstencion is False


def test_un_paso_fallido_dispara_un_replan_y_completa(repo):
    agente, modelo = construir(repo, MotorFalso([dict(FRAGMENTO)]), [
        plan_json([paso("p1", "consultar_caso", {"id_caso": "CASO-9999"})]),
        plan_json([paso("p1", "buscar_documentacion",
                        {"pregunta": "¿cuánto demora el diagnóstico?"})]),
        AIMessage(content="No encontré ese caso, pero la documentación dice 48 horas."),
    ])
    respuesta = agente.consultar("¿dónde está mi caso?")
    assert respuesta.pasos[0].ok is False
    assert respuesta.pasos[-1].ok is True
    assert respuesta.replanes == 1
    assert respuesta.error is None
    assert respuesta.respuesta == (
        "No encontré ese caso, pero la documentación dice 48 horas."
    )
    assert len(modelo.mensajes) == 3


def test_el_plan_se_acota_con_plan_max_pasos(repo, monkeypatch):
    """PLAN_MAX_PASOS gobierna el plan; AGENTE_MAX_PASOS gobierna el bucle."""
    from src import config

    monkeypatch.setattr(config, "PLAN_MAX_PASOS", 1)
    plan_largo = plan_json([
        paso("p1", "listar_casos", {"limite": 1}),
        paso("p2", "listar_casos", {"limite": 2}, depende_de=["p1"]),
    ])
    agente, modelo = construir(repo, MotorFalso([dict(FRAGMENTO)]), [
        plan_largo, plan_largo, plan_largo,  # los MAX_INTENTOS del planificador
    ])
    assert agente.max_pasos == config.AGENTE_MAX_PASOS  # el bucle no cambia
    respuesta = agente.consultar("lista dos veces")
    assert respuesta.pasos == []
    assert respuesta.error is not None
    assert "tope" in respuesta.error.lower() or "pasos" in respuesta.error.lower()
    assert len(modelo.mensajes) == 3
