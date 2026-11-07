"""Integración agente ↔ memoria (RA2 / IL2.2).

Sin red: el modelo es un doble y la memoria usa SQLite temporal. Lo que se fija
aquí es que la memoria no sea un adorno del prompt, sino un canal con tres
efectos observables: bloque en el prompt, herramienta `recordar` y turnos
persistidos.
"""
import json

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from src import config
from src.agente import LIMITE_HISTORIAL, AgenteTecnoApoyo
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


@pytest.fixture(autouse=True)
def modo_reactivo_para_los_guiones(monkeypatch):
    """Estos guiones prueban el canal de memoria del agente reactivo."""
    monkeypatch.setattr(config, "AGENTE_MODO", "reactivo")


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


def memoria_con(tmp_path, estrategia, ventana=1):
    """Memoria con una estrategia de corto plazo concreta, sin tocar la red."""
    return Memoria(
        corto=MemoriaCortoPlazo(estrategia=estrategia, ventana=ventana),
        largo=MemoriaLargoPlazo(
            ruta=tmp_path / f"memoria-{estrategia}.db", usuario="ana"
        ),
    )


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


# --- historial en el prompt (seguimiento) --------------------------------
def test_el_seguimiento_ve_lo_dicho_en_el_turno_anterior(repo, memoria):
    agente, modelo = construir(repo, memoria, [
        mensaje_final("Anoté el CASO-0004."),
        mensaje_final("Sigue en diagnóstico."),
    ])
    agente.consultar("mi caso es CASO-0004")
    agente.consultar("¿en qué estado está?")

    prompt = modelo.mensajes_recibidos[1][0].content
    assert "Historial reciente" in prompt
    # El turno anterior llega completo: la pregunta de seguimiento tiene contexto.
    assert "usuario: mi caso es CASO-0004" in prompt
    assert "asistente: Anoté el CASO-0004." in prompt
    # El turno en curso no se duplica: se registra después de responder.
    assert "¿en qué estado está?" not in prompt


def test_cada_estrategia_de_corto_plazo_se_nota_en_el_prompt(repo, tmp_path):
    def prompt_con(estrategia):
        mem = memoria_con(tmp_path, estrategia, ventana=1)
        for rol, contenido in [
            ("usuario", "turno uno del usuario"),
            ("asistente", "respuesta uno"),
            ("usuario", "turno dos del usuario"),
            ("asistente", "respuesta dos"),
        ]:
            mem.registrar(rol, contenido)
        agente, modelo = construir(repo, mem, [mensaje_final("ok")])
        agente.consultar("¿y ahora?")
        return modelo.mensajes_recibidos[0][0].content

    buffer = prompt_con("buffer")
    ventana = prompt_con("ventana")
    resumen = prompt_con("resumen")

    # buffer: todo el historial.
    assert "turno uno del usuario" in buffer
    # ventana=1: solo el ultimo turno vigente, lo viejo se descarta.
    assert "turno uno del usuario" not in ventana
    assert "asistente: respuesta dos" in ventana
    # resumen: lo viejo se pliega en el resumen en vez de perderse.
    assert "turno uno del usuario" in resumen
    assert "Resumen de turnos anteriores:" in resumen

    assert len({buffer, ventana, resumen}) == 3


def test_el_bloque_de_historial_respeta_el_tope_de_caracteres(repo, memoria):
    for indice in range(8):
        memoria.registrar("usuario", f"turno-{indice:02d} " + "x" * 700)
    memoria.registrar("asistente", "MARCA-FINAL")

    agente, modelo = construir(repo, memoria, [mensaje_final("ok")])
    agente.consultar("¿y el estado?")
    prompt = modelo.mensajes_recibidos[0][0].content

    assert "Historial reciente" in prompt
    # Acotado: el prompt no crece con la sesion, solo hasta el tope del bloque.
    assert len(prompt) <= len(agente.prompt_sistema) + LIMITE_HISTORIAL + 200
    # El recorte conserva la cola: lo mas reciente sigue a la vista.
    assert "MARCA-FINAL" in prompt


def test_sin_memoria_el_prompt_no_lleva_ningun_bloque_nuevo(repo):
    modelo = ModeloFalso([mensaje_final("ok")])
    agente = AgenteTecnoApoyo(
        herramientas=crear_herramientas(repo, MotorFalso()), llm=modelo)
    agente.consultar("mi caso es CASO-0004")
    prompt = modelo.mensajes_recibidos[0][0].content
    assert prompt == agente.prompt_sistema
    assert "Historial reciente" not in prompt
    assert "Memoria relevante del usuario" not in prompt


def test_el_contexto_del_planificador_lleva_historial_y_hechos(repo, memoria):
    memoria.recordar("equipo", "notebook Lenovo")
    memoria.registrar("usuario", "mi caso es CASO-0004")
    agente, _ = construir(repo, memoria, [mensaje_final("ok")])

    contexto = agente._contexto_de_memoria("¿qué notebook y qué caso tengo?")
    assert "Historial reciente" in contexto
    assert "Memoria relevante del usuario" in contexto
    assert "notebook Lenovo" in contexto
    # Mismo orden que el prompt: primero el historial, después los hechos.
    assert contexto.index("Historial reciente") < contexto.index(
        "Memoria relevante del usuario")


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
