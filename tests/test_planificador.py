"""Planificación multi-etapa (RA2 / IL2.3): validación, ejecución y re-plan.

Ningún test toca la red. El `llm` se reemplaza por `ModeloFalso`, que emite
planes JSON como texto y cuenta sus invocaciones; las herramientas son las
reales (`crear_herramientas`) sobre un `RepositorioCasos` temporal y un motor
RAG falso. Así se ejercita el contrato completo —allowlist, placeholders,
orden topológico, presupuesto, re-plan y veredicto de evidencia— sin depender
del proveedor.
"""
import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import ValidationError

from src import config
from src.agente import PasoAgente
from src.casos import RepositorioCasos
from src.herramientas import PLAZOS_HABILES, crear_herramientas
from src.planificador import (
    EJEMPLO_PLAN_CANONICO,
    EjecutorPlan,
    ErrorPlan,
    OrquestadorPlanificado,
    PasoPlan,
    Plan,
    Planificador,
    ResultadoPlan,
    ResultadoPlanificacion,
    orden_topologico,
    resolver_argumentos,
    validar_plan,
)

FRAGMENTO_BUENO = {
    "texto": "El diagnóstico demora hasta 48 horas hábiles.",
    "score": 0.88,
    "document_id": "PROC-004",
    "titulo": "Política de presupuestos y diagnósticos",
    "tema": "procedimientos",
    "version": "1.0",
    "fecha": "2026-01-01",
}


# --- Dobles de test ------------------------------------------------------
class MotorFalso:
    """Motor RAG mínimo: solo lo que `buscar_documentacion` necesita."""

    def __init__(self, fragmentos):
        self.fragmentos = fragmentos

    def recuperar(self, pregunta, k=None):
        return self.fragmentos[:k] if k else self.fragmentos


class ModeloFalso:
    """Doble del LLM: devuelve textos predefinidos y cuenta invocaciones.

    Guarda además el `config` recibido en cada llamada, que es lo que permite
    verificar que los callbacks de observabilidad se reenvían.
    """

    def __init__(self, respuestas):
        self.respuestas = list(respuestas)
        self.invocaciones = 0
        self.mensajes_recibidos = []
        self.configs = []

    def invoke(self, mensajes, config=None):
        self.invocaciones += 1
        self.mensajes_recibidos.append(mensajes)
        self.configs.append(config)
        if not self.respuestas:
            raise AssertionError("El guion del modelo falso se agotó")
        return AIMessage(content=self.respuestas.pop(0))


def motor_con_evidencia():
    return MotorFalso([dict(FRAGMENTO_BUENO)])


def motor_sin_evidencia():
    return MotorFalso([])


@pytest.fixture
def repo(tmp_path):
    return RepositorioCasos(ruta=tmp_path / "casos.db")


def herramientas(repo, motor=None, confirmador=None):
    return crear_herramientas(repo, motor or motor_sin_evidencia(), confirmador)


def aprobado(accion, resumen):
    """Confirmador que autoriza toda escritura durante los tests."""
    return True


# --- Constructores de planes --------------------------------------------
def paso(id_paso, herramienta, argumentos=None, depende_de=None,
         criterio_exito=None):
    datos = {
        "id": id_paso,
        "herramienta": herramienta,
        "argumentos": argumentos if argumentos is not None else {},
        "depende_de": depende_de or [],
    }
    if criterio_exito is not None:
        datos["criterio_exito"] = criterio_exito
    return datos


def plan(*pasos, objetivo="Resolver una consulta de soporte"):
    return Plan.model_validate({"objetivo": objetivo, "pasos": list(pasos)})


def plan_json(*pasos, objetivo="Resolver una consulta de soporte", version=1):
    return json.dumps(
        {"objetivo": objetivo, "version": version, "pasos": list(pasos)},
        ensure_ascii=False,
    )


def orquestador(repo, modelo, motor=None, max_pasos=None, max_replanes=None,
                confirmador=aprobado):
    tools = herramientas(repo, motor, confirmador)
    planificador = Planificador(modelo, tools, max_pasos=max_pasos)
    ejecutor = EjecutorPlan(tools)
    return OrquestadorPlanificado(planificador, ejecutor, max_replanes=max_replanes), tools


# --- validar_plan: rechazos con código estable ---------------------------
def test_rechaza_herramienta_fuera_de_la_allowlist(repo):
    plan_malo = plan(paso("p1", "borrar_todo", {"confirmado": True}))
    with pytest.raises(ErrorPlan) as excinfo:
        validar_plan(plan_malo, herramientas(repo))
    assert excinfo.value.codigo == "herramienta_desconocida"


def test_rechaza_id_duplicado(repo):
    plan_malo = plan(paso("p1", "listar_casos"), paso("p1", "listar_casos"))
    with pytest.raises(ErrorPlan) as excinfo:
        validar_plan(plan_malo, herramientas(repo))
    assert excinfo.value.codigo == "id_duplicado"


def test_rechaza_id_con_formato_distinto_de_pn(repo):
    plan_malo = plan(paso("paso1", "listar_casos"))
    with pytest.raises(ErrorPlan) as excinfo:
        validar_plan(plan_malo, herramientas(repo))
    assert excinfo.value.codigo == "id_invalido"


def test_rechaza_dependencia_inexistente(repo):
    plan_malo = plan(paso("p1", "listar_casos", depende_de=["p9"]))
    with pytest.raises(ErrorPlan) as excinfo:
        validar_plan(plan_malo, herramientas(repo))
    assert excinfo.value.codigo == "dependencia_inexistente"


def test_rechaza_ciclo_de_dependencias(repo):
    plan_malo = plan(
        paso("p1", "listar_casos", depende_de=["p2"]),
        paso("p2", "listar_casos", depende_de=["p1"]),
    )
    with pytest.raises(ErrorPlan) as excinfo:
        validar_plan(plan_malo, herramientas(repo))
    assert excinfo.value.codigo == "ciclo"


def test_rechaza_exceso_de_pasos(repo):
    plan_largo = plan(
        paso("p1", "listar_casos"),
        paso("p2", "listar_casos"),
        paso("p3", "listar_casos"),
    )
    with pytest.raises(ErrorPlan) as excinfo:
        validar_plan(plan_largo, herramientas(repo), max_pasos=2)
    assert excinfo.value.codigo == "tope_de_pasos"


def test_el_tope_por_defecto_sale_de_config(repo, monkeypatch):
    monkeypatch.setattr(config, "PLAN_MAX_PASOS", 1)
    plan_dos = plan(paso("p1", "listar_casos"), paso("p2", "listar_casos"))
    with pytest.raises(ErrorPlan) as excinfo:
        validar_plan(plan_dos, herramientas(repo))
    assert excinfo.value.codigo == "tope_de_pasos"


def test_rechaza_plan_sin_pasos(repo):
    with pytest.raises(ErrorPlan) as excinfo:
        validar_plan(plan(), herramientas(repo))
    assert excinfo.value.codigo == "tope_de_pasos"


def test_rechaza_argumentos_que_no_validan_contra_el_esquema(repo):
    # `pregunta` tiene min_length=3: el plan propone una llamada inválida.
    plan_malo = plan(paso("p1", "buscar_documentacion", {"pregunta": "ab"}))
    with pytest.raises(ErrorPlan) as excinfo:
        validar_plan(plan_malo, herramientas(repo))
    assert excinfo.value.codigo == "argumentos_invalidos"


def test_rechaza_argumentos_con_campos_no_declarados(repo):
    # `extra="forbid"` del esquema: un campo inventado invalida la llamada.
    plan_malo = plan(paso("p1", "crear_caso", {
        "cliente": "Cliente Prueba", "categoria": "recepcion",
        "descripcion": "Notebook no enciende.", "confirmado": True,
    }))
    with pytest.raises(ErrorPlan) as excinfo:
        validar_plan(plan_malo, herramientas(repo))
    assert excinfo.value.codigo == "argumentos_invalidos"


def test_paso_plan_prohibe_campos_extra():
    with pytest.raises(ValidationError):
        PasoPlan.model_validate({"id": "p1", "herramienta": "listar_casos",
                                 "argumentos": {}, "confirmado": True})


def test_validar_plan_acepta_un_plan_valido(repo):
    plan_ok = plan(paso("p1", "listar_casos", {"limite": 5}))
    assert validar_plan(plan_ok, herramientas(repo)) is None


# --- placeholders --------------------------------------------------------
def salidas_de_ejemplo():
    return {"p1": {"ok": True, "datos": {"id": "CASO-0001", "total": 2,
                                        "estado": "abierto"}, "error": None}}


def test_resuelve_placeholder_simple_conservando_el_tipo():
    resuelto = resolver_argumentos(
        {"id_caso": "{{p1.datos.id}}", "limite": "{{p1.datos.total}}"},
        salidas_de_ejemplo(),
    )
    assert resuelto == {"id_caso": "CASO-0001", "limite": 2}
    assert isinstance(resuelto["limite"], int)


def test_resuelve_placeholder_anidado_en_dicts_y_listas():
    argumentos = {
        "filtro": {
            "primero": "{{p1.datos.id}}",
            "anidados": ["{{p1.datos.estado}}", {"profundo": "{{p1.datos.total}}"}],
        }
    }
    resuelto = resolver_argumentos(argumentos, salidas_de_ejemplo())
    assert resuelto == {
        "filtro": {
            "primero": "CASO-0001",
            "anidados": ["abierto", {"profundo": 2}],
        }
    }


def test_resuelve_placeholder_embebido_en_un_texto():
    resuelto = resolver_argumentos(
        {"motivo": "Escalado del caso {{p1.datos.id}} por pedido del cliente."},
        salidas_de_ejemplo(),
    )
    assert resuelto["motivo"] == "Escalado del caso CASO-0001 por pedido del cliente."


def test_un_camino_inexistente_es_un_dato_faltante():
    with pytest.raises(ErrorPlan) as excinfo:
        resolver_argumentos({"id_caso": "{{p1.datos.inexistente}}"},
                            salidas_de_ejemplo())
    assert excinfo.value.codigo == "dato_faltante"


def test_un_paso_sin_salida_es_un_dato_faltante():
    with pytest.raises(ErrorPlan) as excinfo:
        resolver_argumentos({"id_caso": "{{p7.datos.id}}"}, salidas_de_ejemplo())
    assert excinfo.value.codigo == "dato_faltante"


def test_rechaza_placeholder_que_referencia_un_paso_no_declarado(repo):
    plan_malo = plan(
        paso("p1", "listar_casos"),
        paso("p2", "consultar_caso", {"id_caso": "{{p1.datos.casos.0.id}}"}),
    )
    with pytest.raises(ErrorPlan) as excinfo:
        validar_plan(plan_malo, herramientas(repo))
    assert excinfo.value.codigo == "placeholder_no_declarado"


def test_acepta_placeholder_declarado_en_depende_de(repo):
    plan_ok = plan(
        paso("p1", "listar_casos"),
        paso("p2", "consultar_caso", {"id_caso": "{{p1.datos.casos.0.id}}"},
             depende_de=["p1"]),
    )
    assert validar_plan(plan_ok, herramientas(repo)) is None


# --- orden topológico ----------------------------------------------------
def test_orden_topologico_respeta_las_dependencias(repo):
    plan_desordenado = plan(
        paso("p3", "listar_casos", depende_de=["p1", "p2"]),
        paso("p2", "listar_casos", depende_de=["p1"]),
        paso("p1", "listar_casos"),
    )
    orden = orden_topologico(plan_desordenado)
    assert [item.id for item in orden] == ["p1", "p2", "p3"]


# --- happy path con las tools reales ------------------------------------
def test_happy_path_de_tres_pasos(repo):
    modelo = ModeloFalso([plan_json(
        paso("p1", "buscar_documentacion",
             {"pregunta": "¿cuánto demora el diagnóstico?"}),
        paso("p2", "consultar_caso", {"id_caso": "CASO-0001"}),
        paso("p3", "escalar_a_humano",
             {"id_caso": "{{p2.datos.id}}",
              "motivo": "El cliente pide hablar con un técnico."},
             depende_de=["p2"]),
    )])
    orquesta, _ = orquestador(repo, modelo, motor_con_evidencia())
    resultado = orquesta.resolver("Revisa el CASO-0001 y escálalo a un técnico")

    assert isinstance(resultado, ResultadoPlanificacion)
    assert resultado.error is None
    assert [item.herramienta for item in resultado.pasos] == [
        "buscar_documentacion", "consultar_caso", "escalar_a_humano",
    ]
    assert all(isinstance(item, PasoAgente) for item in resultado.pasos)
    assert all(item.ok for item in resultado.pasos)
    # El placeholder se resolvió con la salida real de p2, no con texto.
    assert resultado.pasos[2].argumentos["id_caso"] == "CASO-0001"
    assert set(resultado.salidas) == {"p1", "p2", "p3"}
    # La escritura ocurrió de verdad en el repositorio temporal.
    assert repo.obtener_caso("CASO-0001")["estado"] == "escalado"
    assert resultado.replanes == 0
    assert resultado.presupuesto_usado == 3
    assert resultado.evidencia_deficiente is False
    # Una sola consulta al modelo: el plan se ejecutó tal cual.
    assert modelo.invocaciones == 1


def test_el_planificador_envia_sistema_y_objetivo_al_modelo(repo):
    modelo = ModeloFalso([plan_json(paso("p1", "listar_casos"))])
    tools = herramientas(repo)
    Planificador(modelo, tools).planificar("lista los casos abiertos",
                                          contexto="Recepción, turno mañana")
    primer_turno = modelo.mensajes_recibidos[0]
    assert isinstance(primer_turno[0], SystemMessage)
    assert "listar_casos" in primer_turno[0].content
    assert isinstance(primer_turno[-1], HumanMessage)
    assert "lista los casos abiertos" in primer_turno[-1].content
    assert "Recepción, turno mañana" in primer_turno[-1].content


# --- ejecución: fallo contenido -----------------------------------------
def test_el_ejecutor_detiene_la_ejecucion_y_reporta_el_fallo(repo):
    plan_fallido = plan(
        paso("p1", "listar_casos"),
        paso("p2", "consultar_caso", {"id_caso": "CASO-9999"}),
        paso("p3", "listar_casos"),
    )
    resultado = EjecutorPlan(herramientas(repo)).ejecutar(plan_fallido)

    assert isinstance(resultado, ResultadoPlan)
    assert resultado.error is not None and "CASO-9999" in resultado.error
    assert [item.ok for item in resultado.pasos] == [True, False]
    assert resultado.presupuesto_usado == 2
    # p3 nunca se ejecutó: no hay salida registrada para él.
    assert set(resultado.salidas) == {"p1", "p2"}


def test_el_ejecutor_resuelve_placeholders_con_salidas_previas(repo):
    plan_ok = plan(paso("p1", "escalar_a_humano",
                        {"id_caso": "{{p0.datos.id}}",
                         "motivo": "El cliente pide hablar con un técnico."}))
    resultado = EjecutorPlan(herramientas(repo, confirmador=aprobado)).ejecutar(
        plan_ok, salidas_previas={"p0": {"ok": True, "datos": {"id": "CASO-0002"}}}
    )
    assert resultado.error is None
    assert resultado.pasos[0].ok is True
    assert repo.obtener_caso("CASO-0002")["estado"] == "escalado"


def test_un_placeholder_sin_dato_detiene_el_plan_sin_propagar(repo):
    plan_roto = plan(paso("p1", "consultar_caso",
                          {"id_caso": "{{p0.datos.id}}"}))
    resultado = EjecutorPlan(herramientas(repo)).ejecutar(plan_roto)
    assert resultado.pasos[0].ok is False
    assert resultado.error is not None and "p0" in resultado.error


# --- re-planificación ---------------------------------------------------
def test_un_paso_que_falla_dispara_el_replan_y_completa(repo):
    modelo = ModeloFalso([
        plan_json(paso("p1", "consultar_caso", {"id_caso": "CASO-9999"})),
        plan_json(paso("p1", "listar_casos", {"estado": "abierto"})),
    ])
    orquesta, _ = orquestador(repo, modelo)
    resultado = orquesta.resolver("¿en qué estado está el CASO-9999?")

    assert resultado.error is None
    assert resultado.replanes == 1
    assert modelo.invocaciones == 2
    assert [item.herramienta for item in resultado.pasos] == [
        "consultar_caso", "listar_casos",
    ]
    assert [item.ok for item in resultado.pasos] == [False, True]
    # El segundo plan se pidió con el fallo anterior en el prompt.
    segundo_turno = modelo.mensajes_recibidos[1][-1]
    assert "CASO-9999" in segundo_turno.content


def test_tope_de_replanes_agotado_corta_sin_bucle(repo):
    modelo = ModeloFalso([
        plan_json(paso("p1", "consultar_caso", {"id_caso": "CASO-9999"})),
    ])
    orquesta, _ = orquestador(repo, modelo, max_replanes=0)
    resultado = orquesta.resolver("¿en qué estado está el CASO-9999?")

    assert resultado.error is not None
    assert "replan" in resultado.error.lower()
    assert resultado.replanes == 0
    assert len(resultado.pasos) == 1
    assert modelo.invocaciones == 1
    assert resultado.salidas["p1"]["ok"] is False


def test_segundo_fallo_agota_el_ultimo_replan_sin_bucle(repo):
    plan_que_falla = plan_json(paso("p1", "consultar_caso",
                                    {"id_caso": "CASO-9999"}))
    modelo = ModeloFalso([plan_que_falla, plan_que_falla])
    orquesta, _ = orquestador(repo, modelo, max_replanes=1)
    resultado = orquesta.resolver("¿en qué estado está el CASO-9999?")

    assert resultado.error is not None
    assert "replan" in resultado.error.lower()
    assert resultado.replanes == 1
    assert modelo.invocaciones == 2
    assert len(resultado.pasos) == 2


# --- presupuesto global de pasos ---------------------------------------
def test_presupuesto_global_agotado_por_pasos_ejecutados(repo):
    modelo = ModeloFalso([
        plan_json(
            paso("p1", "listar_casos"),
            paso("p2", "consultar_caso", {"id_caso": "CASO-9999"}),
        ),
    ])
    orquesta, _ = orquestador(repo, modelo, max_pasos=2, max_replanes=3)
    resultado = orquesta.resolver("revisa el caso y resume los abiertos")

    assert resultado.error is not None
    assert "presupuesto" in resultado.error
    assert resultado.presupuesto_usado == 2
    assert len(resultado.pasos) == 2
    # Sin presupuesto no se pide otro plan: una sola invocación al modelo.
    assert modelo.invocaciones == 1


def test_presupuesto_impide_ejecutar_un_plan_demasiado_largo(repo):
    modelo = ModeloFalso([
        plan_json(paso("p1", "consultar_caso", {"id_caso": "CASO-9999"})),
        plan_json(paso("p1", "listar_casos"), paso("p2", "listar_casos")),
    ])
    orquesta, _ = orquestador(repo, modelo, max_pasos=2, max_replanes=1)
    resultado = orquesta.resolver("revisa el caso")

    assert resultado.error is not None
    assert "presupuesto" in resultado.error
    assert resultado.presupuesto_usado == 1
    assert len(resultado.pasos) == 1
    assert modelo.invocaciones == 2


# --- planificador: reintento y fallo ------------------------------------
def test_el_planificador_reintenta_con_los_errores_de_validacion(repo):
    modelo = ModeloFalso([
        plan_json(paso("p1", "borrar_todo", {"confirmado": True})),
        plan_json(paso("p1", "listar_casos")),
    ])
    planificado = Planificador(modelo, herramientas(repo)).planificar("lista casos")

    assert modelo.invocaciones == 2
    assert planificado.pasos[0].herramienta == "listar_casos"
    reintento = modelo.mensajes_recibidos[1][-1].content
    assert "borrar_todo" in reintento
    assert "validación" in reintento.lower()


def test_el_planificador_falla_tras_tres_intentos_sin_inventar_plan(repo):
    modelo = ModeloFalso(["no es un plan", "tampoco esto", "ni esto otro"])
    with pytest.raises(ErrorPlan) as excinfo:
        Planificador(modelo, herramientas(repo)).planificar("lista casos")
    assert excinfo.value.codigo == "plan_invalido"
    assert modelo.invocaciones == Planificador.MAX_INTENTOS


def test_el_planificador_no_acepta_un_plan_por_encima_del_tope(repo):
    plan_largo = plan_json(paso("p1", "listar_casos"), paso("p2", "listar_casos"))
    modelo = ModeloFalso([plan_largo] * Planificador.MAX_INTENTOS)
    with pytest.raises(ErrorPlan) as excinfo:
        Planificador(modelo, herramientas(repo), max_pasos=1).planificar("lista")
    assert excinfo.value.codigo == "plan_invalido"
    assert modelo.invocaciones == Planificador.MAX_INTENTOS


def test_el_planificador_tolera_cercas_de_markdown_y_prosa(repo):
    texto = ("Aquí está el plan solicitado:\n```json\n"
             + plan_json(paso("p1", "listar_casos"), objetivo="listar casos")
             + "\n```\nEspero que sirva.")
    modelo = ModeloFalso([texto])
    planificado = Planificador(modelo, herramientas(repo)).planificar("lista casos")
    assert planificado.objetivo == "listar casos"
    assert planificado.pasos[0].id == "p1"


# --- planificador: prompt, reparaciones y callbacks ---------------------
def test_el_prompt_del_planificador_trae_el_ejemplo_canonico_y_las_reglas(repo):
    modelo = ModeloFalso([plan_json(paso("p1", "listar_casos"))])
    Planificador(modelo, herramientas(repo)).planificar("lista los casos abiertos")
    sistema = modelo.mensajes_recibidos[0][0].content

    # Ejemplo canónico: consultar_caso y el paso que consume {{p1.datos.id}}.
    assert EJEMPLO_PLAN_CANONICO in sistema
    assert '"herramienta": "consultar_caso"' in sistema
    assert '"id_caso": "{{p1.datos.id}}"' in sistema
    assert '"depende_de": ["p1"]' in sistema
    # La regla de dependencia, en mayúsculas visibles.
    assert "REGLA OBLIGATORIA" in sistema
    assert "DEPENDE_DE" in sistema
    assert "OBLIGA" in sistema
    # Los dos formatos válidos de identificador.
    assert "CASO-NNNN" in sistema
    assert "PROC-NNN" in sistema
    # La lista real de tipos de servicio de PLAZOS_HABILES.
    for tipo in PLAZOS_HABILES:
        assert tipo in sistema
    # Un solo objeto JSON, sin markdown alrededor.
    assert "sin bloques de markdown" in sistema


def test_el_ejemplo_canonico_del_prompt_es_un_plan_valido(repo):
    ejemplo = Plan.model_validate(json.loads(EJEMPLO_PLAN_CANONICO))
    assert len(ejemplo.pasos) == 3
    assert validar_plan(ejemplo, herramientas(repo)) is None


def test_el_planificador_acepta_un_plan_valido_en_el_segundo_intento(repo):
    # Comillas simples: el defecto real más frecuente del modelo medido.
    modelo = ModeloFalso([
        "{'objetivo': 'lista casos', 'pasos': []}",
        plan_json(paso("p1", "listar_casos")),
    ])
    planificado = Planificador(modelo, herramientas(repo)).planificar("lista casos")

    assert modelo.invocaciones == 2
    assert planificado.pasos[0].herramienta == "listar_casos"
    # La reparación lleva el error del intento anterior.
    assert "JSON válido" in modelo.mensajes_recibidos[1][-1].content


def test_el_planificador_acepta_un_plan_valido_en_el_tercer_intento(repo):
    plan_con_tool_inexistente = plan_json(paso("p1", "borrar_todo",
                                               {"confirmado": True}))
    modelo = ModeloFalso([
        "{'objetivo': 'lista casos', 'pasos': []}",
        plan_con_tool_inexistente,
        plan_json(paso("p1", "listar_casos")),
    ])
    planificado = Planificador(modelo, herramientas(repo)).planificar("lista casos")

    assert modelo.invocaciones == 3
    assert planificado.pasos[0].herramienta == "listar_casos"
    # El tercer turno acumula los errores de los dos intentos previos.
    tercer_turno = modelo.mensajes_recibidos[2][-1].content
    assert "Intento 1" in tercer_turno
    assert "Intento 2" in tercer_turno
    assert "JSON válido" in tercer_turno
    assert "borrar_todo" in tercer_turno


def test_tres_intentos_invalidos_lanzan_plan_invalido(repo):
    modelo = ModeloFalso([
        "{'objetivo': 'lista casos', 'pasos': []}",
        "no es json",
        '{"objetivo": "lista casos"}',
    ])
    with pytest.raises(ErrorPlan) as excinfo:
        Planificador(modelo, herramientas(repo)).planificar("lista casos")

    assert excinfo.value.codigo == "plan_invalido"
    assert modelo.invocaciones == 3


def test_el_planificador_reenvia_los_callbacks_en_cada_invocacion(repo):
    recolector = object()
    modelo = ModeloFalso([
        "{'objetivo': 'lista casos', 'pasos': []}",
        plan_json(paso("p1", "listar_casos")),
    ])
    Planificador(modelo, herramientas(repo),
                 callbacks=[recolector]).planificar("lista casos")

    assert modelo.invocaciones == 2
    assert modelo.configs == [{"callbacks": [recolector]},
                              {"callbacks": [recolector]}]


def test_sin_callbacks_la_invocacion_no_lleva_config(repo):
    modelo = ModeloFalso([plan_json(paso("p1", "listar_casos"))])
    Planificador(modelo, herramientas(repo)).planificar("lista casos")

    assert modelo.invocaciones == 1
    config_recibida = modelo.configs[0]
    # Sin callbacks no se envía la clave; aceptar `None` también es válido.
    assert config_recibida is None or "callbacks" not in config_recibida


# --- veredicto de evidencia ---------------------------------------------
def test_evidencia_deficiente_cuando_la_ultima_busqueda_no_alcanza(repo):
    modelo = ModeloFalso([plan_json(
        paso("p1", "buscar_documentacion", {"pregunta": "¿reparan tablets?"}),
    )])
    orquesta, _ = orquestador(repo, modelo, motor_sin_evidencia())
    resultado = orquesta.resolver("¿reparan tablets?")

    assert resultado.evidencia_deficiente is True
    assert resultado.error is None
    assert resultado.pasos[0].ok is True


def test_evidencia_suficiente_no_marca_deficiencia(repo):
    modelo = ModeloFalso([plan_json(
        paso("p1", "buscar_documentacion", {"pregunta": "¿cuánto demora el diagnóstico?"}),
    )])
    orquesta, _ = orquestador(repo, modelo, motor_con_evidencia())
    resultado = orquesta.resolver("¿cuánto demora el diagnóstico?")

    assert resultado.evidencia_deficiente is False


def test_un_caso_valido_despues_de_la_busqueda_limpia_la_deficiencia(repo):
    modelo = ModeloFalso([plan_json(
        paso("p1", "buscar_documentacion", {"pregunta": "estado de mi orden"}),
        paso("p2", "consultar_caso", {"id_caso": "CASO-0001"}),
    )])
    orquesta, _ = orquestador(repo, modelo, motor_sin_evidencia())
    resultado = orquesta.resolver("¿en qué estado está mi caso?")

    assert resultado.evidencia_deficiente is False
    assert resultado.pasos[1].ok is True


def test_un_caso_inexistente_no_limpia_la_deficiencia(repo):
    modelo = ModeloFalso([plan_json(
        paso("p1", "buscar_documentacion", {"pregunta": "estado de mi orden"}),
        paso("p2", "consultar_caso", {"id_caso": "CASO-9999"}),
    )])
    orquesta, _ = orquestador(repo, modelo, motor_sin_evidencia(), max_replanes=0)
    resultado = orquesta.resolver("¿en qué estado está mi caso?")

    # p2 falló y detuvo el plan (sin re-plan): la última búsqueda sigue decidiendo.
    assert resultado.evidencia_deficiente is True
    assert resultado.error is not None
