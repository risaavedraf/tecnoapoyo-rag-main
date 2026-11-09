"""Modo planificado del agente (RA2 / IL2.3): plan, síntesis y guard.

El modelo se reemplaza por un doble con guion: primero emite el JSON del plan,
después la síntesis. Sin red.
"""
import json

from langchain_core.messages import AIMessage

from src import config
from src.agente import FRASE_ABSTENCION, PREGUNTA_CLIENTE, AgenteTecnoApoyo
from src.herramientas import crear_herramientas
from src.memoria import Memoria, MemoriaLargoPlazo

from conftest import ModeloFalso, MotorFalso, construir


def plan_json(pasos, objetivo="resolver la consulta"):
    return AIMessage(content=json.dumps(
        {"objetivo": objetivo, "pasos": pasos, "version": 1}, ensure_ascii=False
    ))


def paso(id_paso, herramienta, argumentos, depende_de=None):
    return {"id": id_paso, "herramienta": herramienta,
            "argumentos": argumentos, "depende_de": depende_de or []}


def plan_tecnico():
    """Plan válido que falla al ejecutar por un dato inexistente (fallo técnico).

    `casos.9.id` no existe en la salida real de `listar_casos`: es un
    `dato_faltante` al resolver el argumento, no un `ok=False` de la tool.
    """
    return plan_json([
        paso("p1", "listar_casos", {"limite": 1}),
        paso("p2", "consultar_caso", {"id_caso": "{{p1.datos.casos.9.id}}"},
             depende_de=["p1"]),
    ])


FRAGMENTO = {
    "texto": "El diagnóstico demora hasta 48 horas hábiles.",
    "score": 0.88,
    "document_id": "PROC-004",
    "titulo": "Política de presupuestos y diagnósticos",
    "tema": "procedimientos",
    "version": "1.0",
    "fecha": "2026-01-01",
}


def test_el_modo_por_defecto_sigue_la_configuracion(repo, monkeypatch):
    monkeypatch.setattr(config, "AGENTE_MODO", "planificado")
    modelo = ModeloFalso([AIMessage(content="Hola.")])
    agente = AgenteTecnoApoyo(
        herramientas=crear_herramientas(repo, MotorFalso()), llm=modelo)
    assert agente.modo == config.AGENTE_MODO
    assert agente.modo == "planificado"


def test_un_modo_desconocido_cae_a_reactivo(repo):
    agente, _ = construir(repo, MotorFalso(), [AIMessage(content="Hola.")],
                          modo="telepático")
    assert agente.modo == "reactivo"


def test_modo_planificado_ejecuta_el_plan_y_sintetiza(repo):
    agente, _ = construir(repo, MotorFalso([dict(FRAGMENTO)]), [
        plan_json([paso("p1", "buscar_documentacion",
                        {"pregunta": "¿cuánto demora el diagnóstico?"})]),
        AIMessage(content="El diagnóstico demora hasta 48 horas hábiles."),
    ], modo="planificado")
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
    ], modo="planificado")
    respuesta = agente.consultar("¿en qué estado está mi caso?")
    assert respuesta.error is None
    assert [p.herramienta for p in respuesta.pasos] == ["listar_casos", "consultar_caso"]
    assert all(p.ok for p in respuesta.pasos)
    assert respuesta.respuesta == "Tu caso está cerrado."


def test_un_replan_invalido_responde_con_el_fallo_de_la_herramienta(repo):
    # Lo que se vio en consola: un paso con fallo técnico y un re-plan que no
    # valida. El turno responde con lo ejecutado en vez de quedar vacío.
    agente, _ = construir(repo, MotorFalso(), [
        plan_tecnico(),
        "no es un plan",
        "tampoco esto",
        "ni esto otro",
        AIMessage(content="No pude leer el caso, pero listé tus casos abiertos."),
    ], modo="planificado")
    respuesta = agente.consultar("¿en qué estado está mi caso?")

    assert respuesta.error is None
    assert respuesta.respuesta == (
        "No pude leer el caso, pero listé tus casos abiertos."
    )
    assert respuesta.replanes == 1
    assert [p.herramienta for p in respuesta.pasos] == [
        "listar_casos", "consultar_caso",
    ]
    assert [p.ok for p in respuesta.pasos] == [True, False]


def test_un_plan_vacio_responde_conversando_sin_herramientas(repo):
    # Un saludo no necesita herramientas: el modelo devuelve `pasos: []` y la
    # síntesis usa la rama conversacional, no la de resultados.
    agente, modelo = construir(repo, MotorFalso(), [
        plan_json([]),
        AIMessage(content="¡Hola! ¿En qué te ayudo?"),
    ], modo="planificado")
    respuesta = agente.consultar("hola")

    assert respuesta.error is None
    assert respuesta.respuesta == "¡Hola! ¿En qué te ayudo?"
    assert respuesta.pasos == []
    assert respuesta.abstencion is False
    sintesis = modelo.mensajes[-1][-1].content
    assert "No se ejecutó ninguna herramienta" in sintesis
    assert "SOLO los resultados" not in sintesis


def test_abstencion_forzada_en_modo_planificado(repo):
    agente, _ = construir(repo, MotorFalso([]), [
        plan_json([paso("p1", "buscar_documentacion", {"pregunta": "¿reparan tablets?"})]),
        AIMessage(content="Sí, reparamos tablets por 30.000 pesos."),
    ], modo="planificado")
    respuesta = agente.consultar("¿reparan tablets?")
    assert respuesta.abstencion is True
    assert respuesta.respuesta == FRASE_ABSTENCION


def test_un_plan_invalido_no_ejecuta_nada(repo):
    plan_malo = plan_json([paso("p1", "borrar_todo", {})])
    agente, _ = construir(repo, MotorFalso([dict(FRAGMENTO)]), [
        plan_malo, plan_malo,  # planificación + el único reintento
    ], modo="planificado")
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


def test_un_fallo_tecnico_dispara_un_replan_y_completa(repo):
    # S4: el fallo técnico (un índice de lista que no existe) sí consume el
    # re-plan, y el turno termina con la respuesta del plan nuevo.
    agente, modelo = construir(repo, MotorFalso([dict(FRAGMENTO)]), [
        plan_tecnico(),
        plan_json([paso("p1", "buscar_documentacion",
                        {"pregunta": "¿cuánto demora el diagnóstico?"})]),
        AIMessage(content="No encontré ese caso, pero la documentación dice 48 horas."),
    ], modo="planificado")
    respuesta = agente.consultar("¿dónde está mi caso?")
    assert respuesta.pasos[0].ok is True
    assert respuesta.pasos[1].ok is False
    assert respuesta.pasos[-1].ok is True
    assert respuesta.replanes == 1
    assert respuesta.error is None
    assert respuesta.respuesta == (
        "No encontré ese caso, pero la documentación dice 48 horas."
    )
    assert len(modelo.mensajes) == 3


def test_un_ok_false_de_negocio_responde_con_lo_ejecutado_sin_replan(repo):
    # S4: la herramienta respondió `ok=False` (el caso no existe). El turno
    # responde con eso y no gasta el re-plan disponible.
    agente, modelo = construir(repo, MotorFalso([dict(FRAGMENTO)]), [
        plan_json([paso("p1", "consultar_caso", {"id_caso": "CASO-9999"})]),
        AIMessage(content="No encontré el caso CASO-9999."),
    ], modo="planificado")
    respuesta = agente.consultar("¿en qué estado está CASO-9999?")

    assert respuesta.error is None
    assert respuesta.replanes == 0
    assert [p.herramienta for p in respuesta.pasos] == ["consultar_caso"]
    assert respuesta.pasos[0].ok is False
    assert respuesta.respuesta == "No encontré el caso CASO-9999."
    # Una sola invocación de plan y una de síntesis: no hay segundo plan.
    assert len(modelo.mensajes) == 2


def test_una_consulta_sin_cliente_termina_pidiendo_el_dato(repo):
    # Lo que el usuario reportó: la sesión no tiene cliente identificado y el
    # agente no preguntaba nada. Ahora el plan es un solo pedir_datos y la
    # respuesta es esa pregunta, sin una segunda llamada al modelo.
    pregunta = "¿Cuál es tu nombre y el número de caso que quieres revisar?"
    modelo = ModeloFalso([
        plan_json([paso("p1", "pedir_datos", {"pregunta": pregunta})]),
    ])
    agente = AgenteTecnoApoyo(
        herramientas=crear_herramientas(
            repo, MotorFalso(), alcance="sesion", cliente_actual=None,
            al_identificar=lambda nombre: None,
        ),
        llm=modelo,
        modo="planificado",
    )

    respuesta = agente.consultar("me gustaría saber el estado de mi consulta")

    assert respuesta.error is None
    assert respuesta.abstencion is False
    assert respuesta.respuesta == pregunta
    assert [p.herramienta for p in respuesta.pasos] == ["pedir_datos"]
    assert len(modelo.mensajes) == 1


def test_la_pregunta_pedida_se_respeta_caracter_por_caracter(repo):
    # La respuesta no puede recortar ni reformatear la pregunta: el usuario
    # recibe exactamente lo que el plan pidió mostrar.
    pregunta = "¿Cuál es el número de caso?  "
    modelo = ModeloFalso([
        plan_json([paso("p1", "pedir_datos", {"pregunta": pregunta})]),
    ])
    agente = AgenteTecnoApoyo(
        herramientas=crear_herramientas(
            repo, MotorFalso(), alcance="sesion", cliente_actual=None,
            al_identificar=lambda nombre: None,
        ),
        llm=modelo,
        modo="planificado",
    )

    respuesta = agente.consultar("¿en qué estado está mi caso?")

    assert respuesta.respuesta == pregunta


def test_la_sintesis_planificada_ve_el_historial_de_memoria(repo, tmp_path):
    # T11 real: el plan vacío hacía que la síntesis no viera el historial y
    # pidiera el nombre en vez de responder con el dato ya dicho.
    memoria = Memoria(largo=MemoriaLargoPlazo(ruta=tmp_path / "memoria.db"))
    memoria.corto.agregar("usuario", "Mi equipo es un Lenovo IdeaPad")
    memoria.corto.agregar("asistente", "Anotado: Lenovo IdeaPad.")
    modelo = ModeloFalso([
        plan_json([]),
        AIMessage(content="Tu equipo es un Lenovo IdeaPad."),
    ])
    agente = AgenteTecnoApoyo(
        herramientas=crear_herramientas(repo, MotorFalso()),
        llm=modelo,
        memoria=memoria,
        modo="planificado",
    )

    respuesta = agente.consultar("¿Qué equipo tengo registrado?")

    assert respuesta.error is None
    assert respuesta.respuesta == "Tu equipo es un Lenovo IdeaPad."
    assert respuesta.plan is not None and respuesta.plan.pasos == []
    # El historial llegó al prompt de la síntesis, no solo al planificador.
    assert any(
        "Lenovo IdeaPad" in str(mensaje.content)
        for mensaje in modelo.mensajes[-1]
    )
    # La instrucción conversacional también manda usar el historial antes de
    # pedir un identificador.
    instruccion = "".join(str(mensaje.content) for mensaje in modelo.mensajes[-1])
    assert "no pidas" in instruccion


def test_el_plan_se_acota_con_plan_max_pasos(repo, monkeypatch):
    """PLAN_MAX_PASOS gobierna el plan; AGENTE_MAX_PASOS gobierna el bucle."""
    from src import config

    monkeypatch.setattr(config, "PLAN_MAX_PASOS", 1)
    plan_largo = plan_json([
        paso("p1", "listar_casos", {"limite": 1}),
        paso("p2", "listar_casos", {"limite": 2}, depende_de=["p1"]),
    ])
    # Sin `max_pasos` propio: el bucle tiene que seguir `AGENTE_MAX_PASOS`.
    agente, modelo = construir(repo, MotorFalso([dict(FRAGMENTO)]), [
        plan_largo, plan_largo, plan_largo,  # los MAX_INTENTOS del planificador
    ], modo="planificado", max_pasos=None)
    assert agente.max_pasos == config.AGENTE_MAX_PASOS  # el bucle no cambia
    respuesta = agente.consultar("lista dos veces")
    assert respuesta.pasos == []
    assert respuesta.error is not None
    assert "tope" in respuesta.error.lower() or "pasos" in respuesta.error.lower()
    assert len(modelo.mensajes) == 3


# --- progreso por paso (consola) -----------------------------------------
def test_el_modo_planificado_avisa_cada_paso_ejecutado_en_orden(repo):
    # Un plan con fallo técnico y su re-plan producen dos pasos reales: la
    # consola recibe uno por paso, en el mismo orden que la traza final.
    agente, _ = construir(repo, MotorFalso([dict(FRAGMENTO)]), [
        plan_tecnico(),
        plan_json([paso("p1", "buscar_documentacion",
                        {"pregunta": "¿cuánto demora el diagnóstico?"})]),
        AIMessage(content="No encontré ese caso, pero la documentación dice 48 horas."),
    ], modo="planificado")
    vistos = []

    respuesta = agente.consultar("¿dónde está mi caso?", al_paso=vistos.append)

    assert [p.herramienta for p in vistos] == [
        "listar_casos", "consultar_caso", "buscar_documentacion",
    ]
    assert [p.ok for p in vistos] == [True, False, True]
    assert vistos == respuesta.pasos
    assert respuesta.replanes == 1


def test_el_modo_planificado_transmite_la_sintesis_cuando_hay_emisor(repo):
    """`consultar(al_emitir=...)` llega hasta la síntesis, no se queda en el borde."""
    from types import SimpleNamespace

    class ModeloConStream(ModeloFalso):
        def __init__(self, guion, trozos):
            super().__init__(guion)
            self.trozos = list(trozos)
            self.stream_invocado = 0

        def stream(self, mensajes, *args, **kwargs):
            self.stream_invocado += 1
            yield from self.trozos

    modelo = ModeloConStream([
        plan_json([paso("p1", "buscar_documentacion",
                        {"pregunta": "¿cuánto demora el diagnóstico?"})]),
    ], [
        SimpleNamespace(content="El diagnóstico "),
        SimpleNamespace(content="demora 48 horas.", usage_metadata={
            "input_tokens": 5, "output_tokens": 3, "total_tokens": 8,
        }),
    ])
    agente = AgenteTecnoApoyo(
        herramientas=crear_herramientas(repo, MotorFalso([dict(FRAGMENTO)])),
        llm=modelo, modo="planificado",
    )
    emitidos = []

    respuesta = agente.consultar(
        "¿cuánto demora el diagnóstico?", al_emitir=emitidos.append,
    )

    assert respuesta.error is None
    assert respuesta.respuesta == "El diagnóstico demora 48 horas."
    assert emitidos == ["El diagnóstico ", "demora 48 horas."]
    assert modelo.stream_invocado == 1
    # El uso del stream se suma a mano: los tokens del turno no quedan en cero
    # por haber evitado el callback de LangChain.
    assert respuesta.tokens_total == 8


# --- S1: el plan no identifica con datos de memoria ----------------------
# El modo planificado recibe la memoria como "Contexto disponible" y el modelo
# la usó para identificar a una persona que nunca dijo quién era. El contexto de
# memoria no es el mensaje del turno: la tool solo acepta un nombre que aparezca
# en lo que la persona escribió en este turno.
def memoria_con_correo(nombre="Ricardo"):
    """Memoria mínima con el estado que originó el defecto."""
    from types import SimpleNamespace

    return SimpleNamespace(
        corto=SimpleNamespace(texto=lambda: f"usuario: soy {nombre}"),
        bloque_para=lambda pregunta: (
            f"Memoria relevante del usuario:\n- correo: ricardo@testeo.com\n"
            f"- cliente: {nombre}"
        ),
        registrar=lambda rol, contenido: None,
    )


def agente_de_sesion(repo, guion, memoria=None):
    """Agente planificado con alcance de sesión y un holder del cliente."""
    actual: dict[str, str | None] = {"nombre": None}
    herramientas = crear_herramientas(
        repo, MotorFalso(), None, alcance="sesion",
        cliente_actual=lambda: actual["nombre"],
        al_identificar=lambda nombre: actual.__setitem__("nombre", nombre),
    )
    agente = AgenteTecnoApoyo(
        herramientas=herramientas, llm=ModeloFalso(guion),
        modo="planificado", memoria=memoria,
    )
    return agente, actual


def test_planificado_identifica_solo_con_el_nombre_del_turno(repo):
    repo.crear_caso(
        cliente="Ricardo", categoria="reparacion",
        descripcion="Notebook no enciende.",
    )
    agente, actual = agente_de_sesion(repo, [
        plan_json([paso("p1", "identificar_cliente", {"nombre": "Ricardo"})]),
        AIMessage(content="Listo, reviso tu caso."),
    ])

    respuesta = agente.consultar("hola, soy Ricardo, ¿en qué estado está mi caso?")

    assert respuesta.pasos[0].ok is True
    assert actual["nombre"] == "Ricardo"
    # El plan usó el nombre declarado: la sesión quedó atada a ese cliente y el
    # listado posterior devuelve solo sus casos.
    assert respuesta.plan is not None
    assert respuesta.plan.pasos[0].herramienta == "identificar_cliente"


def test_planificado_no_identifica_con_el_nombre_de_la_memoria(repo):
    repo.crear_caso(
        cliente="Ricardo", categoria="reparacion",
        descripcion="Notebook no enciende.",
    )
    plan_con_el_nombre_recordado = plan_json(
        [paso("p1", "identificar_cliente", {"nombre": "Ricardo"})]
    )
    # Guion de sobra: este test fija la guarda de identidad, no cuántos
    # re-planes consume el fallo.
    agente, actual = agente_de_sesion(repo, [
        plan_con_el_nombre_recordado, plan_con_el_nombre_recordado,
        plan_con_el_nombre_recordado,
    ], memoria=memoria_con_correo())

    respuesta = agente.consultar("¿en qué estado está mi reparación?")

    assert respuesta.pasos[0].ok is False
    assert actual["nombre"] is None


def test_planificado_no_identifica_con_el_correo_de_la_memoria(repo):
    plan_con_el_correo = plan_json(
        [paso("p1", "identificar_cliente", {"nombre": "ricardo@testeo.com"})]
    )
    agente, actual = agente_de_sesion(repo, [
        plan_con_el_correo, plan_con_el_correo, plan_con_el_correo,
    ], memoria=memoria_con_correo())

    respuesta = agente.consultar("¿en qué estado está mi caso?")

    assert respuesta.pasos[0].ok is False
    assert actual["nombre"] is None
    # El valor rechazado no vuelve al modelo en la salida del paso.
    assert "ricardo" not in (respuesta.pasos[0].salida or "")


# --- S3: el cliente de la sesión llega al prompt del planificador --------
def agente_planificado_con_lector(repo, modelo, lector):
    """Agente planificado de sesión con el mismo lector que pasa `src/app.py`."""
    herramientas = crear_herramientas(
        repo, MotorFalso(), None, alcance="sesion",
        cliente_actual=lector, al_identificar=lambda nombre: None,
    )
    return AgenteTecnoApoyo(
        herramientas=herramientas, llm=modelo, modo="planificado",
        cliente_actual=lector,
    )


def test_el_prompt_del_planificador_muestra_el_cliente_de_la_sesion(repo):
    # S5: el planificador planifica sabiendo a quién ya se está atendiendo.
    modelo = ModeloFalso([
        plan_json([paso("p1", "listar_casos", {"limite": 5})]),
        AIMessage(content="Tienes dos casos abiertos."),
    ])
    agente = agente_planificado_con_lector(
        repo, modelo, lambda: "Ricardo Saavedra"
    )

    agente.consultar("¿Qué casos hay abiertos?")

    sistema = str(modelo.mensajes[0][0].content)
    assert "Cliente de la sesión: Ricardo Saavedra" in sistema


def test_el_prompt_del_planificador_declara_la_falta_de_cliente(repo):
    modelo = ModeloFalso([plan_json([]), AIMessage(content="Hola.")])
    agente = agente_planificado_con_lector(repo, modelo, lambda: None)

    agente.consultar("¿qué equipo tengo registrado?")

    sistema = str(modelo.mensajes[0][0].content)
    assert "Cliente de la sesión: sin cliente identificado" in sistema


# --- S1/S2: la síntesis no contradice un listado exitoso -----------------
def caso_de_valentina(repo):
    """El caso sembrado que la consola devolvía mientras la síntesis lo negaba."""
    return repo.listar_casos(cliente="Valentina Soto", limite=1)[0]


def agente_de_consola(repo, guion, cliente="Valentina Soto"):
    """Agente como el de la consola: alcance de sesión y cliente identificado.

    El repositorio de los tests ya trae casos de otros clientes; el listado de
    una sesión solo cuenta los del cliente identificado, que es la situación
    real del defecto (un caso de Valentina Soto y una síntesis que lo negaba).
    """
    agente = AgenteTecnoApoyo(
        herramientas=crear_herramientas(
            repo, MotorFalso(), alcance="sesion",
            cliente_actual=lambda: cliente, al_identificar=lambda nombre: None,
        ),
        llm=ModeloFalso(guion),
        modo="planificado",
        cliente_actual=lambda: cliente,
    )
    return agente


def test_la_sintesis_que_niega_un_listado_exitoso_se_reemplaza(repo):
    # El defecto medido en consola: `listar_casos | ok=sí` y enseguida "No
    # tengo casos abiertos asociados a tu nombre".
    caso = caso_de_valentina(repo)
    agente = agente_de_consola(repo, [
        plan_json([paso("p1", "listar_casos", {"limite": 10})]),
        AIMessage(content="No tengo casos abiertos asociados a tu nombre."),
    ])

    respuesta = agente.consultar("¿qué casos abiertos tengo?")

    assert respuesta.error is None
    assert respuesta.abstencion is False
    assert respuesta.pasos[0].ok is True
    assert respuesta.respuesta == (
        "Encontré 1 caso:\n"
        f"- {caso['id']} · {caso['estado']} · {caso['categoria']}"
    )
    assert "No tengo casos" not in respuesta.respuesta


def test_el_guard_reconoce_la_negacion_sin_tildes_ni_mayusculas(repo):
    caso = caso_de_valentina(repo)
    agente = agente_de_consola(repo, [
        plan_json([paso("p1", "listar_casos", {"limite": 10})]),
        AIMessage(content="NO SE ENCONTRARON CASOS a tu nombre."),
    ])

    respuesta = agente.consultar("¿qué casos tengo?")

    assert caso["id"] in respuesta.respuesta
    assert respuesta.respuesta.startswith("Encontré 1 caso:")


def test_una_sintesis_de_ausencia_sin_casos_se_respeta(repo):
    # S2: con total == 0 el texto del modelo no se toca.
    agente = agente_de_consola(repo, [
        plan_json([paso("p1", "listar_casos", {"limite": 10})]),
        AIMessage(content="No tienes casos abiertos."),
    ], cliente="Pepito Nadie")

    respuesta = agente.consultar("¿qué casos abiertos tengo?")

    assert respuesta.pasos[0].ok is True
    assert respuesta.respuesta == "No tienes casos abiertos."


def test_un_consultar_caso_exitoso_no_dispara_el_guard(repo):
    # S1 habla del listado: un caso leído por id no habilita el reemplazo.
    caso = caso_de_valentina(repo)
    agente, _ = construir(repo, MotorFalso(), [
        plan_json([paso("p1", "consultar_caso", {"id_caso": caso["id"]})]),
        AIMessage(content="No tengo casos abiertos asociados a tu nombre."),
    ], modo="planificado")

    respuesta = agente.consultar("¿en qué estado está el caso?")

    assert respuesta.pasos[0].herramienta == "consultar_caso"
    assert respuesta.pasos[0].ok is True
    assert respuesta.respuesta == "No tengo casos abiertos asociados a tu nombre."


def test_el_guard_no_deja_pasar_el_texto_contradictorio_al_emisor(repo):
    """La consola no puede mostrar la negación y después corregirla."""
    from types import SimpleNamespace

    class ModeloConStream(ModeloFalso):
        def __init__(self, guion, trozos):
            super().__init__(guion)
            self.trozos = list(trozos)

        def stream(self, mensajes, *args, **kwargs):
            yield from self.trozos

    caso = caso_de_valentina(repo)
    modelo = ModeloConStream([
        plan_json([paso("p1", "listar_casos", {"limite": 10})]),
    ], [
        SimpleNamespace(content="No tengo casos "),
        SimpleNamespace(content="abiertos asociados a tu nombre."),
    ])
    agente = AgenteTecnoApoyo(
        herramientas=crear_herramientas(
            repo, MotorFalso(), alcance="sesion",
            cliente_actual=lambda: "Valentina Soto",
            al_identificar=lambda nombre: None,
        ),
        llm=modelo, modo="planificado", cliente_actual=lambda: "Valentina Soto",
    )
    emitidos = []

    respuesta = agente.consultar("¿qué casos abiertos tengo?",
                                al_emitir=emitidos.append)

    assert caso["id"] in respuesta.respuesta
    assert "".join(emitidos) == respuesta.respuesta
    assert "No tengo casos" not in "".join(emitidos)


# --- S13: la emisión no precede a la decisión del turno -------------------
class ModeloQueTransmite(ModeloFalso):
    """Modelo falso que además abre `stream`, para observar la emisión."""

    def __init__(self, guion, trozos):
        super().__init__(guion)
        self.trozos = list(trozos)
        self.stream_invocado = 0

    def stream(self, mensajes, *args, **kwargs):
        self.stream_invocado += 1
        yield from self.trozos


def test_el_turno_no_transmite_una_sintesis_que_luego_contradice(repo):
    """S13: con evidencia deficiente, lo transmitido es la abstención final.

    Un listado exitoso con casos y, después, una búsqueda documental bajo el
    umbral: el turno termina en la frase canónica. La síntesis del listado no
    puede llegar a la consola antes de esa decisión.
    """
    from types import SimpleNamespace

    caso = caso_de_valentina(repo)
    modelo = ModeloQueTransmite([
        plan_json([
            paso("p1", "listar_casos", {"limite": 10}),
            paso("p2", "buscar_documentacion", {"pregunta": "¿reparan tablets?"}),
        ]),
    ], [
        SimpleNamespace(content=f"Encontré 1 caso: {caso['id']}. "),
        SimpleNamespace(content="¿Quieres que revise el estado del equipo?"),
    ])
    agente = AgenteTecnoApoyo(
        herramientas=crear_herramientas(
            repo, MotorFalso(), alcance="sesion",
            cliente_actual=lambda: "Valentina Soto",
            al_identificar=lambda nombre: None,
        ),
        llm=modelo, modo="planificado", cliente_actual=lambda: "Valentina Soto",
    )
    emitidos = []

    respuesta = agente.consultar(
        "¿qué casos abiertos tengo? ¿reparan tablets?",
        al_emitir=emitidos.append,
    )

    assert [p.herramienta for p in respuesta.pasos] == [
        "listar_casos", "buscar_documentacion",
    ]
    assert respuesta.abstencion is True
    assert respuesta.respuesta == FRASE_ABSTENCION
    # Una sola emisión, y es exactamente el texto final.
    assert emitidos == [FRASE_ABSTENCION]
    assert "".join(emitidos) == respuesta.respuesta
    # Nada de la síntesis del listado llegó a la consola.
    assert "Encontré" not in "".join(emitidos)
    assert caso["id"] not in "".join(emitidos)


def test_un_turno_planificado_normal_sigue_transmitiendo(repo):
    """L8: diferir la emisión no puede tragarse la transmisión en vivo."""
    from types import SimpleNamespace

    modelo = ModeloQueTransmite([
        plan_json([paso("p1", "buscar_documentacion",
                        {"pregunta": "¿cuánto demora el diagnóstico?"})]),
    ], [
        SimpleNamespace(content="El diagnóstico "),
        SimpleNamespace(content="demora 48 horas."),
    ])
    agente = AgenteTecnoApoyo(
        herramientas=crear_herramientas(repo, MotorFalso([dict(FRAGMENTO)])),
        llm=modelo, modo="planificado",
    )
    emitidos = []

    respuesta = agente.consultar(
        "¿cuánto demora el diagnóstico?", al_emitir=emitidos.append,
    )

    assert respuesta.error is None
    assert respuesta.abstencion is False
    assert respuesta.respuesta == "El diagnóstico demora 48 horas."
    # La síntesis se transmitió trozo a trozo: el callback recibe texto y la
    # respuesta final no queda vacía.
    assert len(emitidos) > 1
    assert "".join(emitidos) == respuesta.respuesta
    assert modelo.stream_invocado == 1


# --- S3: el plan no identifica a un cliente que nadie nombró -------------
def test_el_agente_reemplaza_un_plan_con_un_nombre_inventado(repo):
    # Lo que se vio en consola: `identificar_cliente` con `nombre = Ricardo`
    # para una pregunta que nunca mencionó a Ricardo.
    agente, actual = agente_de_sesion(repo, [
        plan_json([paso("p1", "identificar_cliente", {"nombre": "Ricardo"})]),
    ])

    respuesta = agente.consultar("¿en qué estado está mi consulta?")

    assert respuesta.error is None
    assert respuesta.abstencion is False
    assert respuesta.respuesta == PREGUNTA_CLIENTE
    assert [p.herramienta for p in respuesta.pasos] == ["pedir_datos"]
    # El reemplazo no es un reintento del planificador.
    assert respuesta.replanes == 0
    assert actual["nombre"] is None
    assert respuesta.plan is not None
    assert respuesta.plan.pasos[0].herramienta == "pedir_datos"


def test_una_pregunta_pedida_no_la_reescribe_el_guard_del_listado(repo):
    # TRIANGULATE: la pregunta de `pedir_datos` es texto del plan, no una
    # síntesis. Tras un listado exitoso y un re-plan que pide un dato, un "No
    # hay problema, ¿me confirmas...?" no puede volverse el resumen del listado.
    caso_de_valentina(repo)
    pregunta = "No hay problema: ¿me confirmas tu nombre completo?"
    agente = agente_de_consola(repo, [
        plan_tecnico(),  # lista casos y falla al resolver el caso por índice
        plan_json([paso("p1", "pedir_datos", {"pregunta": pregunta})]),
    ])

    respuesta = agente.consultar("¿qué casos abiertos tengo?")

    assert respuesta.error is None
    assert respuesta.respuesta == pregunta
    assert [p.herramienta for p in respuesta.pasos] == [
        "listar_casos", "consultar_caso", "pedir_datos",
    ]


# --- S7: el re-plan no repite una escritura ya confirmada ----------------
def crear_caso_de_josefa():
    """El paso de escritura que el re-plan no puede volver a ejecutar."""
    return paso("p1", "crear_caso", {
        "cliente": "Josefa Vera", "categoria": "reparacion",
        "descripcion": "La notebook no enciende.",
    })


def plan_que_escribe_y_falla():
    """Escritura confirmada y, después, un fallo técnico que abre el re-plan.

    `casos.9.id` no existe en la salida real de `listar_casos(limite=1)`: el
    último paso falla al resolver el argumento, que es un fallo técnico y no un
    `ok=False` de negocio, así que deja margen a otro plan.
    """
    return plan_json([
        crear_caso_de_josefa(),
        paso("p2", "listar_casos", {"limite": 1}, depende_de=["p1"]),
        paso("p3", "consultar_caso",
             {"id_caso": "{{p2.datos.casos.9.id}}"}, depende_de=["p2"]),
    ])


def test_un_replan_no_repite_la_escritura_confirmada(repo):
    # S7: el re-plan podía volver a proponer el `crear_caso` que el primer plan
    # ya había confirmado y escribir un segundo caso. El orquestador no puede
    # garantizar que el plan nuevo no la repita, así que no consume el re-plan
    # después de una escritura exitosa: el turno responde con lo ejecutado.
    antes = repo.contar()
    acciones = []

    def confirmar(accion, resumen):
        acciones.append(accion)
        return True

    modelo = ModeloFalso([
        plan_que_escribe_y_falla(),
        AIMessage(content="Creé tu caso."),
    ])
    agente = AgenteTecnoApoyo(
        herramientas=crear_herramientas(repo, MotorFalso(), confirmar),
        llm=modelo, modo="planificado",
    )

    respuesta = agente.consultar("ábreme un caso: la notebook no enciende")

    assert respuesta.error is None
    assert repo.contar() - antes == 1
    # La traza muestra una sola ejecución de la escritura y el humano no tuvo
    # que aprobar una segunda: la escritura confirmada no se repitió.
    assert [p.herramienta for p in respuesta.pasos].count("crear_caso") == 1
    assert acciones == ["crear_caso"]
    assert respuesta.pasos[0].herramienta == "crear_caso"
    assert respuesta.pasos[0].ok is True
    # Sin re-plan no hay una segunda invocación del planificador: el turno
    # gasta solo el plan y la síntesis, y ninguna llamada lleva el aviso de
    # "el intento anterior falló" que abre un plan nuevo.
    assert respuesta.replanes == 0
    assert len(modelo.mensajes) == 2  # plan + síntesis
    assert not any(
        "El intento anterior falló" in str(mensaje.content)
        for mensajes in modelo.mensajes for mensaje in mensajes
    )
    assert respuesta.respuesta == "Creé tu caso."
