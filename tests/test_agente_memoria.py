"""Integración agente ↔ memoria (RA2 / IL2.2).

Sin red: el modelo es un doble y la memoria usa SQLite temporal. Lo que se fija
aquí es que la memoria no sea un adorno del prompt, sino un canal con tres
efectos observables: bloque en el prompt, herramienta `recordar` y turnos
persistidos.
"""
import json

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from src.agente import AgenteTecnoApoyo
from src.casos import RepositorioCasos
from src.herramientas import crear_herramienta_recordar, crear_herramientas, por_nombre
from src.memoria import Memoria, MemoriaCortoPlazo, MemoriaLargoPlazo


class MotorFalso:
    def recuperar(self, pregunta, k=None):
        return []


class ModeloFalso:
    def __init__(self, guion):
        self.guion = list(guion)
        self.herramientas_recibidas = []
        self.mensajes_recibidos = []

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
def memoria(tmp_path):
    return Memoria(
        corto=MemoriaCortoPlazo(estrategia="buffer"),
        largo=MemoriaLargoPlazo(ruta=tmp_path / "memoria.db", usuario="ana"),
    )


def construir(repo, memoria, guion):
    modelo = ModeloFalso(guion)
    herramientas = crear_herramientas(repo, MotorFalso())
    agente = AgenteTecnoApoyo(herramientas=herramientas, llm=modelo,
                              memoria=memoria)
    return agente, modelo


# --- prompt --------------------------------------------------------------
def test_sin_hechos_el_prompt_no_lleva_bloque_de_memoria(repo, memoria):
    agente, modelo = construir(repo, memoria, [mensaje_final("Hola.")])
    agente.consultar("¿cuánto demora el diagnóstico?")
    prompt = modelo.mensajes_recibidos[0][0].content
    assert "Memoria relevante del usuario" not in prompt


def test_los_hechos_relevantes_llegan_al_prompt(repo, memoria):
    memoria.recordar("equipo", "notebook Lenovo IdeaPad")
    agente, modelo = construir(repo, memoria, [mensaje_final("Tu equipo es un Lenovo.")])
    agente.consultar("¿qué notebook tengo?")
    prompt = modelo.mensajes_recibidos[0][0].content
    assert "Memoria relevante del usuario" in prompt
    assert "Lenovo" in prompt


def test_sin_memoria_inyectada_el_prompt_es_el_de_siempre(repo):
    modelo = ModeloFalso([mensaje_final("ok")])
    agente = AgenteTecnoApoyo(
        herramientas=crear_herramientas(repo, MotorFalso()), llm=modelo)
    agente.consultar("hola")
    assert "Memoria relevante del usuario" not in modelo.mensajes_recibidos[0][0].content


# --- herramienta recordar -----------------------------------------------
def test_con_memoria_el_agente_expone_la_tool_recordar(repo, memoria):
    _, modelo = construir(repo, memoria, [mensaje_final("ok")])
    nombres = {herramienta.name for herramienta in modelo.herramientas_recibidas}
    assert "recordar" in nombres


def test_sin_memoria_no_existe_la_tool_recordar(repo):
    modelo = ModeloFalso([mensaje_final("ok")])
    AgenteTecnoApoyo(herramientas=crear_herramientas(repo, MotorFalso()), llm=modelo)
    nombres = {herramienta.name for herramienta in modelo.herramientas_recibidas}
    assert "recordar" not in nombres


def test_el_modelo_puede_guardar_un_hecho_con_recordar(repo, memoria):
    agente, _ = construir(repo, memoria, [
        mensaje_tool(llamada("recordar", {
            "clave": "correo", "valor": "ana@example.com"}, "call_1")),
        mensaje_final("Anotado."),
    ])
    respuesta = agente.consultar("mi correo es ana@example.com")
    assert respuesta.pasos[0].ok is True
    hechos = {h["clave"]: h["valor"] for h in memoria.listar_hechos()}
    assert hechos["correo"] == "ana@example.com"


def test_recordar_sobrevive_al_reinicio(repo, tmp_path):
    ruta = tmp_path / "memoria.db"
    memoria = Memoria(
        corto=MemoriaCortoPlazo(estrategia="buffer"),
        largo=MemoriaLargoPlazo(ruta=ruta, usuario="ana"),
    )
    agente, _ = construir(repo, memoria, [
        mensaje_tool(llamada("recordar", {"clave": "ciudad", "valor": "Ñuñoa"})),
        mensaje_final("Anotado."),
    ])
    agente.consultar("vivo en Ñuñoa")

    otra = Memoria(
        corto=MemoriaCortoPlazo(estrategia="buffer"),
        largo=MemoriaLargoPlazo(ruta=ruta, usuario="ana"),
    )
    assert {h["clave"]: h["valor"] for h in otra.listar_hechos()}["ciudad"] == "Ñuñoa"


def test_recordar_invalido_no_rompe_el_bucle(repo, memoria):
    agente, _ = construir(repo, memoria, [
        mensaje_tool(llamada("recordar", {"clave": "a"})),  # falta valor
        mensaje_final("No pude anotarlo."),
    ])
    respuesta = agente.consultar("anota algo")
    assert respuesta.pasos[0].ok is False
    assert memoria.listar_hechos() == []
    assert respuesta.respuesta == "No pudo anotarlo." or respuesta.respuesta == "No pude anotarlo."


# --- turnos persistidos --------------------------------------------------
def test_los_turnos_se_persisten_al_consultar(repo, memoria):
    id_sesion = memoria.abrir_sesion()
    agente, _ = construir(repo, memoria, [mensaje_final("Hasta 48 horas.")])
    agente.consultar("¿cuánto demora el diagnóstico?")
    turnos = memoria.largo.cargar_turnos(id_sesion)
    assert [t["rol"] for t in turnos] == ["usuario", "asistente"]
    assert turnos[0]["contenido"] == "¿cuánto demora el diagnóstico?"
    assert turnos[1]["contenido"] == "Hasta 48 horas."


def test_no_se_persiste_un_turno_de_asistente_vacio(repo, memoria):
    id_sesion = memoria.abrir_sesion()
    agente, _ = construir(repo, memoria, [mensaje_final("")])
    agente.consultar("algo")
    turnos = memoria.largo.cargar_turnos(id_sesion)
    assert [t["rol"] for t in turnos] == ["usuario"]


def test_el_corto_plazo_recibe_el_turno(repo, memoria):
    agente, _ = construir(repo, memoria, [mensaje_final("Hasta 48 horas.")])
    agente.consultar("¿cuánto demora?")
    assert [t["contenido"] for t in memoria.corto.turnos()] == [
        "¿cuánto demora?", "Hasta 48 horas.",
    ]


def test_una_falla_de_memoria_no_tumba_la_consulta(repo):
    class MemoriaRota(Memoria):
        def bloque_para(self, consulta):
            return ""

        def registrar(self, rol, contenido):
            raise RuntimeError("disco lleno")

    memoria = MemoriaRota(
        corto=MemoriaCortoPlazo(estrategia="buffer"),
        largo=MemoriaLargoPlazo(ruta=":memory:", usuario="ana"),
    )
    agente, _ = construir(repo, memoria, [mensaje_final("Respondo igual.")])
    respuesta = agente.consultar("hola")
    assert respuesta.respuesta == "Respondo igual."
    assert respuesta.error is None


# --- mensaje de tool serializable ---------------------------------------
def test_el_toolmessage_de_recordar_es_json(repo, memoria):
    agente, modelo = construir(repo, memoria, [
        mensaje_tool(llamada("recordar", {"clave": "equipo", "valor": "Lenovo"})),
        mensaje_final("Anotado."),
    ])
    agente.consultar("mi equipo es un Lenovo")
    segundo_turno = modelo.mensajes_recibidos[1]
    contenido = [m for m in segundo_turno if isinstance(m, ToolMessage)][0].content
    assert isinstance(contenido, str)
    assert json.loads(contenido)["ok"] is True


def test_por_nombre_incluye_recordar_solo_con_memoria(repo, memoria):
    herramienta = crear_herramienta_recordar(memoria)
    assert por_nombre([herramienta])["recordar"].name == "recordar"
