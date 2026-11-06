"""Bucle del agente (RA2 / IL2.1): tools, tope, pasos y guard de abstención.

Ningún test toca la red: el modelo se reemplaza por `ModeloFalso`, que
devuelve un guion de mensajes predefinidos y registra lo que recibe. Así se
ejercita el bucle real —`bind_tools`, ejecución de herramientas, `ToolMessage`,
tope de pasos— sin depender de Mistral.
"""
import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src import rag
from src.agente import (
    FRASE_ABSTENCION,
    AgenteTecnoApoyo,
    PasoAgente,
    RespuestaAgente,
)
from src.casos import RepositorioCasos
from src.herramientas import crear_herramientas

FRAGMENTO_BUENO = {
    "texto": "El diagnóstico demora hasta 48 horas hábiles.",
    "score": 0.88,
    "document_id": "PROC-004",
    "titulo": "Política de presupuestos y diagnósticos",
    "tema": "procedimientos",
    "version": "1.0",
    "fecha": "2026-01-01",
}


class MotorFalso:
    def __init__(self, fragmentos):
        self.fragmentos = fragmentos
        # Cuántas recuperaciones se pidieron de verdad: permite afirmar que una
        # herramienta no se volvió a ejecutar, no solo que el paso quedó en False.
        self.llamadas = 0

    def recuperar(self, pregunta, k=None):
        self.llamadas += 1
        return self.fragmentos[:k] if k else self.fragmentos


class ModeloFalso:
    """Doble de ChatOpenAI: devuelve el guion y recuerda lo recibido."""

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


def motor_con_evidencia():
    return MotorFalso([dict(FRAGMENTO_BUENO)])


def motor_sin_evidencia():
    return MotorFalso([])


def construir(repo, motor, guion, confirmador=None, max_pasos=6):
    modelo = ModeloFalso(guion)
    herramientas = crear_herramientas(repo, motor, confirmador)
    agente = AgenteTecnoApoyo(herramientas=herramientas, llm=modelo,
                              max_pasos=max_pasos)
    return agente, modelo


# --- contrato de la frase canónica ---------------------------------------
def test_la_frase_de_abstencion_es_la_del_prompt_del_rag():
    # Sin importar un literal distinto: si el prompt del motor cambia, este
    # test señala la deriva antes de que el agente responda otra frase.
    assert FRASE_ABSTENCION in rag.PROMPT_FUNDAMENTADO


def test_bind_tools_recibe_las_ocho_herramientas(repo):
    agente, modelo = construir(repo, motor_con_evidencia(),
                               [mensaje_final("Hola.")])
    nombres = {herramienta.name for herramienta in modelo.herramientas_recibidas}
    assert nombres == {
        "buscar_documentacion", "consultar_caso", "listar_casos",
        "crear_caso", "agregar_nota", "actualizar_estado",
        "escalar_a_humano", "calcular_plazo",
    }
    assert agente.max_pasos == 6


# --- respuesta sin herramientas ------------------------------------------
def test_responde_sin_pasos_cuando_el_modelo_no_usa_herramientas(repo):
    agente, _ = construir(repo, motor_con_evidencia(),
                          [mensaje_final("El horario es de 9 a 18.")])
    respuesta = agente.consultar("¿cuál es el horario?")
    assert isinstance(respuesta, RespuestaAgente)
    assert respuesta.respuesta == "El horario es de 9 a 18."
    assert respuesta.abstencion is False
    assert respuesta.pasos == []
    assert respuesta.iteraciones == 1
    assert respuesta.error is None


# --- ejecución de herramientas -------------------------------------------
def test_ejecuta_la_tool_y_entrega_toolmessage_al_modelo(repo):
    agente, modelo = construir(repo, motor_con_evidencia(), [
        mensaje_tool(llamada("buscar_documentacion",
                             {"pregunta": "¿cuánto demora el diagnóstico?"})),
        mensaje_final("El diagnóstico demora hasta 48 horas hábiles."),
    ])
    respuesta = agente.consultar("¿cuánto demora el diagnóstico?")

    assert respuesta.respuesta == "El diagnóstico demora hasta 48 horas hábiles."
    assert len(respuesta.pasos) == 1
    paso = respuesta.pasos[0]
    assert isinstance(paso, PasoAgente)
    assert paso.herramienta == "buscar_documentacion"
    assert paso.ok is True
    # El modelo, en su segunda llamada, ve el resultado como ToolMessage
    # serializable: es lo que exige el bucle para continuar el razonamiento.
    segundo_turno = modelo.mensajes_recibidos[1]
    tool_messages = [m for m in segundo_turno if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].tool_call_id == "call_1"
    contenido = tool_messages[0].content
    assert isinstance(contenido, str)
    payload = json.loads(contenido)
    assert payload["ok"] is True
    assert payload["datos"]["evidencia_suficiente"] is True


def test_varias_herramientas_en_un_mismo_turno(repo):
    agente, _ = construir(repo, motor_con_evidencia(), [
        mensaje_tool(
            llamada("buscar_documentacion", {"pregunta": "¿cuánto demora?"}, "call_1"),
            llamada("consultar_caso", {"id_caso": "CASO-0001"}, "call_2"),
            llamada("calcular_plazo", {"tipo_servicio": "diagnostico",
                                       "fecha_inicio": "2026-03-06"}, "call_3"),
        ),
        mensaje_final("Listo."),
    ])
    respuesta = agente.consultar("¿cuánto demora y cuándo termina?")
    assert [paso.herramienta for paso in respuesta.pasos] == [
        "buscar_documentacion", "consultar_caso", "calcular_plazo",
    ]
    assert all(paso.ok for paso in respuesta.pasos)


# --- guard duro de abstención -------------------------------------------
def test_guard_fuerza_la_abstencion_si_no_hay_evidencia(repo):
    agente, _ = construir(repo, motor_sin_evidencia(), [
        mensaje_tool(llamada("buscar_documentacion", {"pregunta": "¿reparan tablets?"})),
        mensaje_final("Sí, reparamos tablets con un costo de 30.000 pesos."),
    ])
    respuesta = agente.consultar("¿reparan tablets?")
    # El modelo intentó responder igual: el ejecutor impone la abstención.
    assert respuesta.abstencion is True
    assert respuesta.respuesta == FRASE_ABSTENCION
    assert len(respuesta.pasos) == 1


def test_guard_no_actua_cuando_la_evidencia_alcanza(repo):
    agente, _ = construir(repo, motor_con_evidencia(), [
        mensaje_tool(llamada("buscar_documentacion", {"pregunta": "¿cuánto demora?"})),
        mensaje_final("Demora hasta 48 horas hábiles."),
    ])
    respuesta = agente.consultar("¿cuánto demora el diagnóstico?")
    assert respuesta.abstencion is False
    assert respuesta.respuesta == "Demora hasta 48 horas hábiles."


def test_guard_no_actua_si_no_hubo_ninguna_recuperacion(repo):
    agente, _ = construir(repo, motor_sin_evidencia(),
                          [mensaje_final("El horario es de 9 a 18.")])
    respuesta = agente.consultar("¿cuál es el horario?")
    assert respuesta.abstencion is False


def test_guard_usa_la_ultima_recuperacion(repo):
    # Primero sin evidencia, después con evidencia: manda la última.
    motor = MotorFalso([])
    agente, modelo = construir(repo, motor, [
        mensaje_tool(llamada("buscar_documentacion", {"pregunta": "primera"}, "call_1")),
        mensaje_tool(llamada("buscar_documentacion", {"pregunta": "segunda"}, "call_2")),
        mensaje_final("Demora 48 horas."),
    ])
    motor.fragmentos = [dict(FRAGMENTO_BUENO)]
    respuesta = agente.consultar("¿cuánto demora?")
    assert respuesta.abstencion is False


def test_el_modelo_puede_abstenerse_por_su_cuenta(repo):
    agente, _ = construir(repo, motor_con_evidencia(), [
        mensaje_final(FRASE_ABSTENCION),
    ])
    respuesta = agente.consultar("¿algo fuera del corpus?")
    assert respuesta.abstencion is True
    assert respuesta.respuesta == FRASE_ABSTENCION


# --- tope de iteraciones -------------------------------------------------
def test_tope_de_pasos_detiene_el_bucle(repo):
    # Argumentos distintos en cada paso: el bucle debe agotarse por presupuesto
    # y no adelantarse el detector de repeticiones, que corta con otro error.
    guion = [
        mensaje_tool(llamada("buscar_documentacion",
                             {"pregunta": f"consulta número {i}"}, f"call_{i}"))
        for i in range(10)
    ]
    agente, modelo = construir(repo, motor_con_evidencia(), guion, max_pasos=3)
    respuesta = agente.consultar("lista casos por siempre")
    assert respuesta.iteraciones == 3
    assert len(modelo.mensajes_recibidos) == 3
    assert len(respuesta.pasos) == 3
    assert respuesta.error is not None
    assert "tope" in respuesta.error.lower()
    # Nunca se inventa una respuesta final cuando el bucle se corta por tope.
    assert respuesta.respuesta == ""


# --- errores de herramienta no rompen el bucle ---------------------------
def test_herramienta_desconocida_se_reporta_como_paso_con_error(repo):
    agente, _ = construir(repo, motor_con_evidencia(), [
        mensaje_tool(llamada("borrar_todo", {"confirmado": True})),
        mensaje_final("No pude hacer eso."),
    ])
    respuesta = agente.consultar("borra todo")
    assert respuesta.pasos[0].ok is False
    assert respuesta.pasos[0].error is not None
    assert "borrar_todo" in respuesta.pasos[0].error
    assert respuesta.respuesta == "No pude hacer eso."


def test_argumentos_invalidos_no_escriben_y_se_reportan(repo):
    antes = repo.contar()
    agente, _ = construir(repo, motor_con_evidencia(), [
        mensaje_tool(llamada("crear_caso", {
            "cliente": "Cliente Prueba", "categoria": "recepcion",
            "descripcion": "Notebook no enciende.", "confirmado": True,
        })),
        mensaje_final("No se pudo crear."),
    ])
    respuesta = agente.consultar("crea un caso")
    assert respuesta.pasos[0].ok is False
    assert repo.contar() == antes


def test_el_agente_no_escribe_sin_confirmador(repo):
    antes = repo.contar()
    agente, _ = construir(repo, motor_con_evidencia(), [
        mensaje_tool(llamada("crear_caso", {
            "cliente": "Cliente Prueba", "categoria": "recepcion",
            "descripcion": "Notebook no enciende.",
        })),
        mensaje_final("Solicité autorización."),
    ])
    respuesta = agente.consultar("crea un caso")
    assert respuesta.pasos[0].ok is False
    assert repo.contar() == antes


def test_el_agente_escribe_con_confirmador_aprobado(repo):
    agente, _ = construir(repo, motor_con_evidencia(), [
        mensaje_tool(llamada("crear_caso", {
            "cliente": "Cliente Prueba", "categoria": "recepcion",
            "descripcion": "Notebook no enciende.",
        })),
        mensaje_final("Caso creado."),
    ], confirmador=lambda accion, resumen: True)
    respuesta = agente.consultar("crea un caso")
    assert respuesta.pasos[0].ok is True
    assert repo.contar() == 11


# --- pasos intermedios ---------------------------------------------------
def test_return_intermediate_steps_false_no_devuelve_pasos(repo):
    agente, _ = construir(repo, motor_con_evidencia(), [
        mensaje_tool(llamada("listar_casos", {})),
        mensaje_final("Hay 10 casos."),
    ])
    respuesta = agente.consultar("¿cuántos casos hay?", return_intermediate_steps=False)
    assert respuesta.pasos == []
    assert respuesta.respuesta == "Hay 10 casos."


def test_el_prompt_del_sistema_y_la_pregunta_viajan_al_modelo(repo):
    agente, modelo = construir(repo, motor_con_evidencia(),
                               [mensaje_final("ok")])
    agente.consultar("¿cuánto demora el diagnóstico?")
    primer_turno = modelo.mensajes_recibidos[0]
    assert primer_turno[0].type == "system"
    assert "abstenerte" in primer_turno[0].content.lower()
    assert isinstance(primer_turno[-1], HumanMessage)
    assert primer_turno[-1].content == "¿cuánto demora el diagnóstico?"


# --- F1c: robustez del bucle --------------------------------------------
class ModeloQueFalla:
    """Doble que revienta al invocar, como un 429 o un timeout del proveedor."""

    def __init__(self, guion_previo=None):
        self.guion_previo = list(guion_previo or [])

    def bind_tools(self, herramientas, **kwargs):
        return self

    def invoke(self, mensajes, config=None):
        if self.guion_previo:
            return self.guion_previo.pop(0)
        raise RuntimeError("429 demasiadas solicitudes")


def test_error_del_proveedor_no_propaga_y_queda_trazado(repo):
    agente = AgenteTecnoApoyo(
        herramientas=crear_herramientas(repo, motor_con_evidencia()),
        llm=ModeloQueFalla(),
    )
    # La sesión no debe morir: el mismo criterio que ya usa el motor RAG.
    respuesta = agente.consultar("¿cuánto demora el diagnóstico?")
    assert respuesta.respuesta == ""
    assert respuesta.abstencion is False
    assert respuesta.error == "RuntimeError: 429 demasiadas solicitudes"
    assert respuesta.iteraciones == 1
    assert respuesta.tokens_total == 0
    assert isinstance(respuesta.latencia_seg, float)


def test_error_del_proveedor_conserva_los_pasos_ya_ejecutados(repo):
    agente = AgenteTecnoApoyo(
        herramientas=crear_herramientas(repo, motor_con_evidencia()),
        llm=ModeloQueFalla([
            mensaje_tool(llamada("listar_casos", {})),
        ]),
    )
    respuesta = agente.consultar("¿cuántos casos hay?")
    assert len(respuesta.pasos) == 1
    assert respuesta.pasos[0].herramienta == "listar_casos"
    assert respuesta.error is not None and "429" in respuesta.error


def test_respuesta_final_vacia_se_reporta_como_error(repo):
    agente, _ = construir(repo, motor_con_evidencia(), [mensaje_final("")])
    respuesta = agente.consultar("hola")
    assert respuesta.respuesta == ""
    assert respuesta.abstencion is False
    # Una respuesta exitosa que no dice nada es un fallo, no un éxito vacío.
    assert respuesta.error == "El modelo no devolvió una respuesta final."


def test_consultar_caso_limpia_un_veredicto_previo_sin_evidencia(repo):
    agente, _ = construir(repo, motor_sin_evidencia(), [
        mensaje_tool(llamada("buscar_documentacion",
                             {"pregunta": "estado de mi orden"}, "call_1")),
        mensaje_tool(llamada("consultar_caso", {"id_caso": "CASO-0001"}, "call_2")),
        mensaje_final("Tu caso CASO-0001 está cerrado."),
    ])
    respuesta = agente.consultar("¿en qué estado está mi caso?")
    # Hay datos válidos del caso: abstenerse aquí sería un falso positivo.
    assert respuesta.abstencion is False
    assert respuesta.respuesta == "Tu caso CASO-0001 está cerrado."


def test_listar_casos_limpia_un_veredicto_previo_sin_evidencia(repo):
    agente, _ = construir(repo, motor_sin_evidencia(), [
        mensaje_tool(llamada("buscar_documentacion", {"pregunta": "mis casos"}, "call_1")),
        mensaje_tool(llamada("listar_casos", {"estado": "abierto"}, "call_2")),
        mensaje_final("Tienes 3 casos abiertos."),
    ])
    respuesta = agente.consultar("¿qué casos tengo abiertos?")
    assert respuesta.abstencion is False
    assert respuesta.respuesta == "Tienes 3 casos abiertos."


def test_un_caso_inexistente_no_limpia_el_veredicto(repo):
    agente, _ = construir(repo, motor_sin_evidencia(), [
        mensaje_tool(llamada("buscar_documentacion", {"pregunta": "mi caso"}, "call_1")),
        mensaje_tool(llamada("consultar_caso", {"id_caso": "CASO-9999"}, "call_2")),
        mensaje_final("El caso no existe."),
    ])
    respuesta = agente.consultar("¿dónde está mi caso?")
    assert respuesta.abstencion is True
    assert respuesta.respuesta == FRASE_ABSTENCION


def test_calcular_plazo_no_limpia_el_veredicto_de_abstencion(repo):
    agente, _ = construir(repo, motor_sin_evidencia(), [
        mensaje_tool(llamada("buscar_documentacion",
                             {"pregunta": "¿reparan tablets?"}, "call_1")),
        mensaje_tool(llamada("calcular_plazo", {
            "tipo_servicio": "diagnostico", "fecha_inicio": "2026-03-06"}, "call_2")),
        mensaje_final("El diagnóstico demora 2 días hábiles."),
    ])
    respuesta = agente.consultar("¿reparan tablets y cuánto demora?")
    # El razonamiento determinista no aporta evidencia documental.
    assert respuesta.abstencion is True
    assert respuesta.respuesta == FRASE_ABSTENCION


# --- T07: una llamada idéntica no se vuelve a ejecutar -------------------
ARGUMENTOS_REPETIDOS = {"pregunta": "¿reparan tablets?"}


def test_la_repeticion_identica_no_ejecuta_y_avisa_al_modelo(repo):
    motor = motor_con_evidencia()
    agente, modelo = construir(repo, motor, [
        mensaje_tool(llamada("buscar_documentacion",
                             dict(ARGUMENTOS_REPETIDOS), "call_1")),
        mensaje_tool(llamada("buscar_documentacion",
                             dict(ARGUMENTOS_REPETIDOS), "call_2")),
        mensaje_final("La documentación no menciona tablets."),
    ])
    respuesta = agente.consultar("¿reparan tablets?")

    # La primera llamada se ejecuta; la segunda solo deja constancia del error.
    assert len(respuesta.pasos) == 2
    primera, segunda = respuesta.pasos
    assert primera.ok is True
    assert segunda.ok is False
    assert segunda.error is not None
    assert "mismos argumentos" in segunda.error
    assert "buscar_documentacion" in segunda.error
    # El motor falso confirma que la herramienta corrió una sola vez.
    assert motor.llamadas == 1
    # El aviso llega al modelo como ToolMessage en el turno siguiente.
    tercer_turno = modelo.mensajes_recibidos[2]
    rechazos = [m for m in tercer_turno if isinstance(m, ToolMessage)
                and m.tool_call_id == "call_2"]
    assert len(rechazos) == 1
    payload = json.loads(str(rechazos[0].content))
    assert payload["ok"] is False
    assert "mismos argumentos" in payload["error"]
    # El turno sigue su curso: el modelo pudo responder con lo que tenía.
    assert respuesta.error is None
    assert respuesta.respuesta == "La documentación no menciona tablets."


def test_la_tercera_llamada_identica_corta_el_bucle(repo):
    motor = motor_con_evidencia()
    agente, modelo = construir(repo, motor, [
        mensaje_tool(llamada("buscar_documentacion",
                             dict(ARGUMENTOS_REPETIDOS), "call_1")),
        mensaje_tool(llamada("buscar_documentacion",
                             dict(ARGUMENTOS_REPETIDOS), "call_2")),
        mensaje_tool(llamada("buscar_documentacion",
                             dict(ARGUMENTOS_REPETIDOS), "call_3")),
        mensaje_final("Respuesta inventada que no debe usarse."),
    ])
    respuesta = agente.consultar("¿reparan tablets?")

    assert respuesta.error is not None
    assert "repetición" in respuesta.error.lower()
    assert "buscar_documentacion" in respuesta.error
    # Se corta sin inventar respuesta y sin convertirlo en abstención.
    assert respuesta.respuesta == ""
    assert respuesta.abstencion is False
    assert motor.llamadas == 1
    # La tercera llamada no se ejecutó: el modelo no volvió a ser invocado.
    assert len(respuesta.pasos) == 2
    assert len(modelo.mensajes_recibidos) == 3


def test_misma_herramienta_con_argumentos_distintos_no_es_repeticion(repo):
    motor = motor_con_evidencia()
    agente, _ = construir(repo, motor, [
        mensaje_tool(llamada("buscar_documentacion", {"pregunta": "tablets"}, "call_1")),
        mensaje_tool(llamada("buscar_documentacion", {"pregunta": "impresoras"}, "call_2")),
        mensaje_final("Cubro tablets e impresoras."),
    ])
    respuesta = agente.consultar("¿reparan tablets e impresoras?")

    assert [paso.ok for paso in respuesta.pasos] == [True, True]
    assert motor.llamadas == 2
    assert respuesta.error is None
    assert respuesta.respuesta == "Cubro tablets e impresoras."


def test_herramientas_distintas_con_los_mismos_argumentos_no_es_repeticion(repo):
    # Los mismos argumentos sobre otra herramienta no son la misma llamada: la
    # clave incluye el nombre. El repositorio no tiene dos tools que compartan
    # esquema de argumentos, así que la segunda se registra con su propio error
    # de validación en vez del aviso de repetición.
    motor = motor_con_evidencia()
    agente, _ = construir(repo, motor, [
        mensaje_tool(llamada("buscar_documentacion",
                             dict(ARGUMENTOS_REPETIDOS), "call_1")),
        mensaje_tool(llamada("consultar_caso",
                             dict(ARGUMENTOS_REPETIDOS), "call_2")),
        mensaje_final("Solo sé lo que dice la documentación."),
    ])
    respuesta = agente.consultar("¿reparan tablets?")

    assert len(respuesta.pasos) == 2
    primera, segunda = respuesta.pasos
    assert primera.ok is True
    assert segunda.herramienta == "consultar_caso"
    assert segunda.ok is False
    assert segunda.error is not None
    # Se intentó de verdad: el error es de validación, no el aviso de repetición.
    assert "mismos argumentos" not in segunda.error
    assert motor.llamadas == 1
    assert respuesta.respuesta == "Solo sé lo que dice la documentación."


def test_el_corte_por_repeticion_no_se_convierte_en_abstencion(repo):
    # Sin evidencia documental el guard impondría `FRASE_ABSTENCION` al cerrar el
    # turno; un corte por repetición es un error del bucle y sigue siendo error.
    motor = motor_sin_evidencia()
    agente, _ = construir(repo, motor, [
        mensaje_tool(llamada("buscar_documentacion",
                             dict(ARGUMENTOS_REPETIDOS), "call_1")),
        mensaje_tool(llamada("buscar_documentacion",
                             dict(ARGUMENTOS_REPETIDOS), "call_2")),
        mensaje_tool(llamada("buscar_documentacion",
                             dict(ARGUMENTOS_REPETIDOS), "call_3")),
        mensaje_final(FRASE_ABSTENCION),
    ])
    respuesta = agente.consultar("¿reparan tablets?")

    assert respuesta.respuesta == ""
    assert respuesta.abstencion is False
    assert respuesta.error is not None
    assert "repetición" in respuesta.error.lower()
    assert motor.llamadas == 1
