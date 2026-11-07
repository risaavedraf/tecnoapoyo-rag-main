"""Contratos de las herramientas del agente (RA2 / IL2.1).

Sin red: el motor RAG se reemplaza por un doble que implementa `recuperar()`.

Hay dos capas y las pruebas las separan a propósito:

  - funciones puras: reciben repositorio/motor/confirmador y devuelven
    `ResultadoHerramienta`;
  - tools registradas: exponen solo los argumentos del negocio al modelo y
    devuelven JSON serializable para que el bucle del agente construya un
    `ToolMessage`.

La confirmación NO forma parte del esquema que ve el modelo. Un modelo que
envíe `confirmado=True` debe ser rechazado por el esquema, y la escritura no
debe ocurrir: la confirmación la inyecta el ejecutor con un `s/n` en consola.
"""
import json

import pytest
from pydantic import ValidationError

from src import config
from src.casos import RepositorioCasos
from src.herramientas import (
    PLAZOS_HABILES,
    ResultadoHerramienta,
    actualizar_estado,
    agregar_nota,
    buscar_documentacion,
    calcular_plazo,
    consultar_caso,
    crear_caso,
    crear_herramientas,
    escalar_a_humano,
    listar_casos,
    mensaje_del_turno,
    por_nombre,
    sumar_dias_habiles,
)

FRAGMENTO = {
    "texto": "El diagnóstico demora hasta 48 horas hábiles.",
    "score": 0.88,
    "document_id": "PROC-004",
    "titulo": "Política de presupuestos y diagnósticos",
    "tema": "procedimientos",
    "version": "1.0",
    "fecha": "2026-01-01",
}


class MotorFalso:
    """Doble de RAGEngine: solo expone `recuperar()`."""

    def __init__(self, fragmentos=None, excepcion=None):
        self.fragmentos = fragmentos if fragmentos is not None else []
        self.excepcion = excepcion
        self.llamadas = []

    def recuperar(self, pregunta, k=None):
        self.llamadas.append((pregunta, k))
        if self.excepcion is not None:
            raise self.excepcion
        return self.fragmentos[:k] if k else self.fragmentos


def confirmador_si(accion, resumen):
    return True


def confirmador_no(accion, resumen):
    return False


@pytest.fixture
def repo(tmp_path):
    return RepositorioCasos(ruta=tmp_path / "casos.db")


@pytest.fixture
def motor():
    return MotorFalso(fragmentos=[dict(FRAGMENTO)])


# --- funciones puras: consulta -------------------------------------------
def test_buscar_documentacion_delega_en_el_motor_y_respeta_k(motor):
    resultado = buscar_documentacion(motor, "¿cuánto demora el diagnóstico?", k=5)
    assert isinstance(resultado, ResultadoHerramienta)
    assert resultado.ok is True
    assert motor.llamadas == [("¿cuánto demora el diagnóstico?", 5)]
    assert resultado.datos["fragmentos"] == [FRAGMENTO]
    assert resultado.datos["mejor_score"] == 0.88
    assert resultado.datos["umbral"] == config.UMBRAL_ABSTENCION
    assert resultado.datos["evidencia_suficiente"] is True


def test_buscar_documentacion_propaga_la_abstencion_del_motor():
    motor_bajo = MotorFalso(fragmentos=[{**FRAGMENTO, "score": 0.10}])
    resultado = buscar_documentacion(motor_bajo, "¿reparan tablets?")
    assert resultado.ok is True
    assert resultado.datos["evidencia_suficiente"] is False
    assert resultado.datos["mejor_score"] == 0.10


def test_buscar_documentacion_sin_resultados_no_tiene_evidencia():
    resultado = buscar_documentacion(MotorFalso(), "¿hay soporte los domingos?")
    assert resultado.datos["mejor_score"] == 0.0
    assert resultado.datos["evidencia_suficiente"] is False


# --- frontera de dominio vista desde la tool -----------------------------
# La frontera vive en `src/rag.py`, no en la tool: estas pruebas usan el motor
# real (sin índice, con un vector store de score fijo) para comprobar que el
# veredicto llega a `evidencia_suficiente` sin tocar `CLAVES_SALIDA`.
def _motor_real_con_score(score):
    """`RAGEngine` sin índice ni red, con un vector store de score fijo."""
    from types import SimpleNamespace

    from src.rag import RAGEngine

    class VectorStoreFijo:
        def similarity_search_with_relevance_scores(self, pregunta, k):
            doc = SimpleNamespace(
                page_content="El diagnóstico demora hasta 48 horas hábiles.",
                metadata={
                    "document_id": "PROC-004",
                    "titulo": "Política de presupuestos y diagnósticos",
                    "tema": "procedimientos",
                    "version": "1.0",
                    "fecha": "2026-01-01",
                },
            )
            return [(doc, score)]

    motor_real = RAGEngine(carga_indice=False)
    motor_real.vector_store = VectorStoreFijo()
    return motor_real


@pytest.mark.parametrize("pregunta", [
    "¿Cuánto cuesta reparar un iPhone 15?",
    "¿Reparan tablets?",
])
def test_buscar_documentacion_niega_evidencia_fuera_de_alcance(monkeypatch, pregunta):
    # score 0.7499/0.7139 son los medidos en la evaluación real: superan el
    # umbral de 0.45, pero el corpus no cubre esos productos.
    monkeypatch.setattr(config, "LLM_API_KEY", "clave-de-prueba")
    monkeypatch.setattr(config, "UMBRAL_ABSTENCION", 0.45)

    resultado = buscar_documentacion(_motor_real_con_score(0.7499), pregunta, k=3)

    assert resultado.ok is True
    # Las claves son exactamente las declaradas: el motivo no viaja en `datos`.
    assert set(resultado.datos) == {
        "fragmentos", "mejor_score", "umbral", "evidencia_suficiente"
    }
    assert resultado.datos["fragmentos"] == []
    assert resultado.datos["mejor_score"] == 0.0
    assert resultado.datos["umbral"] == config.UMBRAL_ABSTENCION
    assert resultado.datos["evidencia_suficiente"] is False


def test_buscar_documentacion_conserva_la_evidencia_de_un_producto_cubierto(
    monkeypatch,
):
    monkeypatch.setattr(config, "LLM_API_KEY", "clave-de-prueba")
    monkeypatch.setattr(config, "UMBRAL_ABSTENCION", 0.45)

    resultado = buscar_documentacion(
        _motor_real_con_score(0.7499),
        "¿cuánto demora el diagnóstico de un notebook?",
        k=3,
    )

    assert resultado.datos["mejor_score"] == 0.7499
    assert resultado.datos["evidencia_suficiente"] is True
    assert len(resultado.datos["fragmentos"]) == 1


def test_buscar_documentacion_reporta_fallo_del_indice():
    motor_roto = MotorFalso(excepcion=FileNotFoundError("No existe el índice FAISS"))
    resultado = buscar_documentacion(motor_roto, "¿cuánto demora?")
    assert resultado.ok is False
    assert resultado.error is not None
    assert "No existe el índice FAISS" in resultado.error


def test_consultar_caso_existente_e_inexistente(repo):
    encontrado = consultar_caso(repo, "caso-0001")
    assert encontrado.ok is True
    assert encontrado.datos["id"] == "CASO-0001"
    faltante = consultar_caso(repo, "CASO-9999")
    assert faltante.ok is False
    assert faltante.error is not None
    assert "CASO-9999" in faltante.error


def test_listar_casos_filtra_y_reporta_estado_invalido(repo):
    abiertos = listar_casos(repo, estado="abierto")
    assert abiertos.ok is True
    assert abiertos.datos["total"] == 3
    # Más reciente primero: el listado no puede quedar anclado a los ids bajos.
    assert [caso["id"] for caso in abiertos.datos["casos"]] == [
        "CASO-0009", "CASO-0007", "CASO-0004",
    ]
    invalido = listar_casos(repo, estado="pendiente")
    assert invalido.ok is False
    assert invalido.error is not None
    assert "no permitido" in invalido.error


def test_listar_casos_reporta_el_total_real_y_no_el_tamano_de_pagina(repo):
    resultado = listar_casos(repo, limite=2)
    assert resultado.ok is True
    assert len(resultado.datos["casos"]) == 2
    # `total` es la cantidad de casos que cumplen el filtro, no la página.
    assert resultado.datos["total"] == 10


def test_el_total_respeta_el_filtro_de_estado(repo):
    abiertos = listar_casos(repo, estado="abierto", limite=1)
    assert len(abiertos.datos["casos"]) == 1
    assert abiertos.datos["total"] == 3


def test_listar_casos_incluye_el_caso_recien_creado(repo):
    repo.crear_caso(
        cliente="Cliente Once", categoria="recepcion",
        descripcion="Notebook sin imagen en pantalla.",
    )
    resultado = listar_casos(repo)
    assert resultado.datos["total"] == 11
    assert resultado.datos["casos"][0]["id"] == "CASO-0011"


# --- funciones puras: escritura y confirmación ---------------------------
def test_crear_caso_sin_confirmador_no_escribe(repo):
    antes = repo.contar()
    resultado = crear_caso(
        repo, None, cliente="Cliente Prueba", categoria="recepcion",
        descripcion="Notebook no enciende.", prioridad="alta",
    )
    assert resultado.ok is False
    assert resultado.error is not None and "confirmación" in resultado.error
    assert repo.contar() == antes


def test_crear_caso_con_confirmador_que_deniega_no_escribe(repo):
    antes = repo.contar()
    resultado = crear_caso(
        repo, confirmador_no, cliente="Cliente Prueba", categoria="recepcion",
        descripcion="Notebook no enciende.", prioridad="alta",
    )
    assert resultado.ok is False
    assert repo.contar() == antes


def test_crear_caso_confirmado_escribe_y_persiste(repo):
    resultado = crear_caso(
        repo, confirmador_si, cliente="Cliente Prueba", categoria="recepcion",
        descripcion="Notebook no enciende.", prioridad="alta",
        procedimiento="PROC-001",
    )
    assert resultado.ok is True
    assert resultado.datos["id"] == "CASO-0011"
    assert resultado.datos["procedimiento"] == "PROC-001"
    assert repo.contar() == 11


def test_confirmador_recibe_la_accion_y_el_resumen_de_lo_que_se_escribira(repo):
    recibido = {}

    def confirmador_espia(accion, resumen):
        recibido["accion"] = accion
        recibido["resumen"] = resumen
        return False

    agregar_nota(repo, confirmador_espia, "CASO-0009", "Se respondió por correo.")
    assert recibido["accion"] == "agregar_nota"
    assert recibido["resumen"]["id_caso"] == "CASO-0009"
    assert recibido["resumen"]["nota"] == "Se respondió por correo."
    assert repo.obtener_caso("CASO-0009")["notas"] == []


def test_agregar_nota_denegada_no_modifica_el_caso(repo):
    resultado = agregar_nota(repo, confirmador_no, "CASO-0009", "Nota sin confirmar.")
    assert resultado.ok is False
    assert repo.obtener_caso("CASO-0009")["notas"] == []


def test_agregar_nota_confirmada_escribe(repo):
    resultado = agregar_nota(repo, confirmador_si, "CASO-0009", "Se respondió por correo.")
    assert resultado.ok is True
    assert repo.obtener_caso("CASO-0009")["notas"] == ["Se respondió por correo."]


def test_actualizar_estado_denegado_no_cambia_el_estado(repo):
    resultado = actualizar_estado(repo, confirmador_no, "CASO-0004", "cerrado")
    assert resultado.ok is False
    assert repo.obtener_caso("CASO-0004")["estado"] == "abierto"


def test_actualizar_estado_confirmado_cambia_y_persiste(repo):
    resultado = actualizar_estado(repo, confirmador_si, "CASO-0004", "en_progreso")
    assert resultado.ok is True
    assert repo.obtener_caso("CASO-0004")["estado"] == "en_progreso"


def test_escalar_denegado_no_cambia_nada(repo):
    antes = repo.obtener_caso("CASO-0007")
    resultado = escalar_a_humano(repo, confirmador_no, "CASO-0007", "Urgente.")
    assert resultado.ok is False
    assert repo.obtener_caso("CASO-0007") == antes


def test_escalar_confirmado_deja_estado_y_motivo(repo):
    resultado = escalar_a_humano(repo, confirmador_si, "CASO-0007", "Urgente.")
    assert resultado.ok is True
    caso = repo.obtener_caso("CASO-0007")
    assert caso["estado"] == "escalado"
    assert "Escalado: Urgente." in caso["notas"]


# --- razonamiento determinista -------------------------------------------
def test_calcular_plazo_salta_el_fin_de_semana():
    # Viernes 2026-03-06 + 5 días hábiles = viernes 2026-03-13.
    resultado = calcular_plazo("reparacion", "2026-03-06")
    assert resultado.ok is True
    assert resultado.datos["dias_habiles_base"] == PLAZOS_HABILES["reparacion"]
    assert resultado.datos["fecha_estimada"] == "2026-03-13"


def test_calcular_plazo_suma_dias_extra():
    resultado = calcular_plazo("reparacion", "2026-03-06", dias_habiles_extra=2)
    assert resultado.datos["dias_habiles_totales"] == 7
    assert resultado.datos["fecha_estimada"] == "2026-03-17"


def test_sumar_dias_habiles_desde_sabado_mueve_al_lunes():
    from datetime import date
    assert sumar_dias_habiles(date(2026, 3, 7), 1) == date(2026, 3, 9)
    assert sumar_dias_habiles(date(2026, 3, 7), 0) == date(2026, 3, 7)


def test_calcular_plazo_rechaza_tipo_desconocido():
    resultado = calcular_plazo("telepatía", "2026-03-06")
    assert resultado.ok is False
    assert resultado.error is not None and "no reconocido" in resultado.error
    assert resultado.datos is None


def test_calcular_plazo_rechaza_fecha_invalida():
    resultado = calcular_plazo("diagnostico", "2026-13-01")
    assert resultado.ok is False
    assert resultado.error is not None and "inválida" in resultado.error


def test_calcular_plazo_sin_fecha_usa_hoy():
    from datetime import date
    resultado = calcular_plazo("diagnostico")
    assert resultado.datos["fecha_inicio"] == date.today().isoformat()


def test_calcular_plazo_es_determinista_para_la_misma_entrada():
    primera = calcular_plazo("mantenimiento_preventivo", "2026-05-04")
    segunda = calcular_plazo("mantenimiento_preventivo", "2026-05-04")
    assert primera == segunda


# --- tools registradas: esquema del modelo y salida serializable ---------
@pytest.fixture
def tools(repo, motor):
    return por_nombre(crear_herramientas(repo, motor, confirmador_si))


def test_expone_las_nueve_herramientas_con_su_nombre(tools):
    assert sorted(tools) == sorted([
        "buscar_documentacion", "consultar_caso", "listar_casos",
        "crear_caso", "agregar_nota", "actualizar_estado",
        "escalar_a_humano", "calcular_plazo", "pedir_datos",
    ])


def test_la_tool_devuelve_json_serializable(tools):
    salida = tools["consultar_caso"].invoke({"id_caso": "CASO-0001"})
    assert isinstance(salida, str)
    payload = json.loads(salida)
    assert set(payload) == {"ok", "datos", "error"}
    assert payload["ok"] is True
    assert payload["datos"]["id"] == "CASO-0001"


def test_la_tool_de_busqueda_devuelve_json_con_evidencia(tools):
    payload = json.loads(tools["buscar_documentacion"].invoke(
        {"pregunta": "¿cuánto demora el diagnóstico?"}
    ))
    assert payload["ok"] is True
    assert payload["datos"]["evidencia_suficiente"] is True


def test_el_esquema_del_modelo_no_expone_confirmado(tools):
    """La confirmación no puede ser un argumento que el modelo controle."""
    for nombre in ("crear_caso", "agregar_nota", "actualizar_estado",
                   "escalar_a_humano"):
        propiedades = tools[nombre].args_schema.model_json_schema()["properties"]
        assert "confirmado" not in propiedades, f"{nombre} expone confirmado"


@pytest.mark.parametrize("nombre,argumentos", [
    ("crear_caso", {
        "cliente": "Cliente Prueba", "categoria": "recepcion",
        "descripcion": "Notebook no enciende.", "confirmado": True,
    }),
    ("agregar_nota", {"id_caso": "CASO-0009", "nota": "Sin permiso.",
                      "confirmado": True}),
    ("actualizar_estado", {"id_caso": "CASO-0004", "estado": "cerrado",
                           "confirmado": True}),
    ("escalar_a_humano", {"id_caso": "CASO-0007", "motivo": "Urgente.",
                          "confirmado": True}),
])
def test_el_modelo_no_puede_autoconfirmarse_una_escritura(tools, repo,
                                                         nombre, argumentos):
    """El bypass: un modelo que envíe confirmado=True debe ser rechazado.

    No basta con que el valor por defecto sea False: si el campo existe en el
    esquema, el modelo puede rellenarlo. El esquema debe rechazar el campo y
    ninguna escritura debe ocurrir.
    """
    antes = {caso["id"]: caso for caso in repo.listar_casos(limite=100)}
    with pytest.raises(ValidationError):
        tools[nombre].invoke(argumentos)
    despues = {caso["id"]: caso for caso in repo.listar_casos(limite=100)}
    assert despues == antes
    assert repo.contar() == len(antes)


def test_sin_confirmador_las_tools_de_escritura_no_escriben(repo, motor):
    tools = por_nombre(crear_herramientas(repo, motor))
    antes = repo.contar()
    payload = json.loads(tools["crear_caso"].invoke({
        "cliente": "Cliente Prueba", "categoria": "recepcion",
        "descripcion": "Notebook no enciende.",
    }))
    assert payload["ok"] is False
    assert "confirmación" in payload["error"]
    assert repo.contar() == antes


def test_con_confirmador_real_la_tool_de_escritura_escribe(repo, motor):
    herramientas = crear_herramientas(repo, motor, lambda accion, resumen: True)
    payload = json.loads(por_nombre(herramientas)["crear_caso"].invoke({
        "cliente": "Cliente Prueba", "categoria": "recepcion",
        "descripcion": "Notebook no enciende.", "procedimiento": "proc-001",
    }))
    assert payload["ok"] is True
    assert payload["datos"]["id"] == "CASO-0011"
    assert repo.contar() == 11


def test_la_tool_rechaza_argumentos_invalidos_de_negocio(tools, repo):
    antes = repo.contar()
    with pytest.raises(ValidationError):
        tools["crear_caso"].invoke({
            "cliente": "Cliente Prueba", "categoria": "recepcion",
            "descripcion": "Notebook no enciende.", "prioridad": "urgentísima",
        })
    with pytest.raises(ValidationError):
        tools["actualizar_estado"].invoke({"id_caso": "CASO-0004", "estado": "en_limbo"})
    assert repo.contar() == antes


# --- alcance de las tools: global vs sesión ------------------------------
def test_alcance_de_sesion_sin_cliente_no_lista_y_pide_identificarlo(repo, motor):
    tools = por_nombre(crear_herramientas(repo, motor, None, alcance="sesion"))
    payload = json.loads(tools["listar_casos"].invoke({}))
    assert payload["ok"] is False
    assert payload["datos"] is None
    # El mensaje es accionable: nombra el comando que resuelve el bloqueo.
    assert "/cliente" in payload["error"]
    # Y no filtra nada: ni ids ni nombres de la semilla.
    assert "CASO-" not in payload["error"]
    assert "María" not in payload["error"]


def test_alcance_de_sesion_acota_el_listado_al_cliente_actual(repo, motor):
    repo.crear_caso(
        cliente="Valentina Soto", categoria="reparacion",
        descripcion="Teclado con teclas pegadas.",
    )
    tools = por_nombre(crear_herramientas(
        repo, motor, None, alcance="sesion", cliente_actual="valentina soto",
    ))
    payload = json.loads(tools["listar_casos"].invoke({"limite": 1}))
    assert payload["ok"] is True
    assert [caso["id"] for caso in payload["datos"]["casos"]] == ["CASO-0011"]
    assert payload["datos"]["total"] == 2

    # Un estado que existe sobre todo en casos ajenos no abre la puerta: el
    # cliente lo fija la sesión, no los argumentos del modelo.
    abiertos = json.loads(tools["listar_casos"].invoke(
        {"estado": "abierto", "limite": 50}
    ))
    ids = [caso["id"] for caso in abiertos["datos"]["casos"]]
    assert ids == ["CASO-0011", "CASO-0007"]
    assert all(
        caso["cliente"].lower() == "valentina soto"
        for caso in abiertos["datos"]["casos"]
    )
    assert abiertos["datos"]["total"] == 2


def test_el_alcance_ignora_mayusculas_tambien_en_nombres_con_tilde(repo, motor):
    tools = por_nombre(crear_herramientas(
        repo, motor, None, alcance="sesion", cliente_actual="MARÍA FERNANDA ROJAS",
    ))

    listado = json.loads(tools["listar_casos"].invoke({}))
    consulta = json.loads(tools["consultar_caso"].invoke({"id_caso": "CASO-0001"}))

    assert listado["ok"] is True
    assert listado["datos"]["total"] == 1
    assert [caso["id"] for caso in listado["datos"]["casos"]] == ["CASO-0001"]
    assert consulta["ok"] is True
    assert consulta["datos"]["id"] == "CASO-0001"


def test_alcance_de_sesion_con_cliente_sin_casos_no_inventa_resultados(repo, motor):
    tools = por_nombre(crear_herramientas(
        repo, motor, None, alcance="sesion", cliente_actual="Cliente Sin Casos",
    ))
    payload = json.loads(tools["listar_casos"].invoke({}))
    assert payload["ok"] is True
    assert payload["datos"] == {"total": 0, "casos": []}


def test_alcance_global_sigue_listando_todos_los_casos(repo, motor):
    """Regresión: `scripts/evaluar_agente.py` construye las tools sin alcance."""
    tools = por_nombre(crear_herramientas(repo, motor, None))
    payload = json.loads(tools["listar_casos"].invoke({}))
    assert payload["ok"] is True
    assert payload["datos"]["total"] == 10
    assert len(payload["datos"]["casos"]) == 10
    assert {caso["cliente"] for caso in payload["datos"]["casos"]} == {
        "María Fernanda Rojas", "Constructora Andes Ltda.", "Pedro Salinas",
        "Colegio San Ramón", "Ana Lucía Vera", "Jorge Peña",
        "Valentina Soto", "Estudio Contable Bravo", "Rosa Medina",
        "Municipalidad de La Reina",
    }


# --- alcance de sesión: acceso por id directo (lectura y escritura) ------
# `listar_casos` no es la única puerta a los datos: si el modelo ya conoce un
# id, `consultar_caso` lo devuelve y las tools de escritura lo modifican. El
# alcance de sesión tiene que cerrar también esa vía.
ESCRITURAS = (
    ("agregar_nota", {"id_caso": "CASO-0004", "nota": "Nota no autorizada."}),
    ("actualizar_estado", {"id_caso": "CASO-0004", "estado": "cerrado"}),
    ("escalar_a_humano", {"id_caso": "CASO-0004", "motivo": "Sin autorización."}),
)


def _instantanea(repo):
    """Estado completo de los casos, para probar que nada se modificó."""
    return {caso["id"]: caso for caso in repo.listar_casos(limite=100)}


def confirmador_espia(registro):
    def confirmar(accion, resumen):
        registro.append((accion, dict(resumen)))
        return True

    return confirmar


def test_sesion_sin_cliente_no_permite_leer_un_caso_por_id(repo, motor):
    tools = por_nombre(crear_herramientas(repo, motor, None, alcance="sesion"))
    payload = json.loads(tools["consultar_caso"].invoke({"id_caso": "CASO-0004"}))
    assert payload["ok"] is False
    assert payload["datos"] is None
    assert "/cliente" in payload["error"]
    # Ni un dato del caso ajeno en el mensaje.
    assert "Colegio San Ramón" not in payload["error"]
    assert "mantenimiento preventivo" not in payload["error"]


def test_sesion_con_otro_cliente_no_permite_leer_un_caso_ajeno(repo, motor):
    tools = por_nombre(crear_herramientas(
        repo, motor, None, alcance="sesion", cliente_actual="Valentina Soto",
    ))
    ajeno = json.loads(tools["consultar_caso"].invoke({"id_caso": "CASO-0004"}))
    assert ajeno["ok"] is False
    assert ajeno["datos"] is None
    assert "Colegio San Ramón" not in ajeno["error"]
    assert "mantenimiento preventivo" not in ajeno["error"]
    assert "PROC-003" not in ajeno["error"]

    # Un id inexistente y un id ajeno responden lo mismo (salvo el id que el
    # propio modelo envió): la diferencia no es observable y no sirve de sondeo.
    inexistente = json.loads(tools["consultar_caso"].invoke({"id_caso": "CASO-9999"}))
    assert inexistente["ok"] is False
    assert inexistente["datos"] is None
    assert ajeno["error"].replace("CASO-0004", "CASO-X") == \
        inexistente["error"].replace("CASO-9999", "CASO-X")


def test_sesion_con_el_cliente_dueno_si_lee_su_caso(repo, motor):
    tools = por_nombre(crear_herramientas(
        repo, motor, None, alcance="sesion", cliente_actual="valentina soto",
    ))
    payload = json.loads(tools["consultar_caso"].invoke({"id_caso": "CASO-0007"}))
    assert payload["ok"] is True
    assert payload["datos"]["id"] == "CASO-0007"
    assert payload["datos"]["cliente"] == "Valentina Soto"


def test_sesion_sin_cliente_no_escribe_ni_pide_confirmacion(repo, motor):
    registro: list = []
    tools = por_nombre(crear_herramientas(
        repo, motor, confirmador_espia(registro), alcance="sesion",
    ))
    antes = _instantanea(repo)
    for nombre, argumentos in ESCRITURAS:
        payload = json.loads(tools[nombre].invoke(argumentos))
        assert payload["ok"] is False, nombre
        assert payload["datos"] is None, nombre
        assert "/cliente" in payload["error"], nombre
    # Sin cliente no se llega al humano ni se toca la base.
    assert registro == []
    assert _instantanea(repo) == antes


def test_sesion_con_otro_cliente_no_escribe_ni_pide_confirmacion(repo, motor):
    registro: list = []
    tools = por_nombre(crear_herramientas(
        repo, motor, confirmador_espia(registro), alcance="sesion",
        cliente_actual="Valentina Soto",
    ))
    antes = _instantanea(repo)
    for nombre, argumentos in ESCRITURAS:
        payload = json.loads(tools[nombre].invoke(argumentos))
        assert payload["ok"] is False, nombre
        assert payload["datos"] is None, nombre
        assert "Colegio San Ramón" not in payload["error"], nombre
    # El caso ajeno no se modificó y el humano nunca fue consultado.
    assert registro == []
    assert _instantanea(repo) == antes
    assert repo.contar() == 10


def test_sesion_con_el_cliente_dueno_si_permite_escribir(repo, motor):
    registro: list = []
    tools = por_nombre(crear_herramientas(
        repo, motor, confirmador_espia(registro), alcance="sesion",
        cliente_actual="Valentina Soto",
    ))
    nota = json.loads(tools["agregar_nota"].invoke(
        {"id_caso": "CASO-0007", "nota": "Se respondió por teléfono."}
    ))
    assert nota["ok"] is True
    assert "Se respondió por teléfono." in nota["datos"]["notas"]

    estado = json.loads(tools["actualizar_estado"].invoke(
        {"id_caso": "CASO-0007", "estado": "en_progreso"}
    ))
    assert estado["ok"] is True
    assert estado["datos"]["estado"] == "en_progreso"

    escalado = json.loads(tools["escalar_a_humano"].invoke(
        {"id_caso": "CASO-0007", "motivo": "Requiere autorización."}
    ))
    assert escalado["ok"] is True
    assert escalado["datos"]["estado"] == "escalado"
    assert [accion for accion, _ in registro] == [
        "agregar_nota", "actualizar_estado", "escalar_a_humano",
    ]


def test_alcance_global_conserva_la_lectura_y_escritura_por_id(repo, motor):
    """Regresión: la evaluación y la demo leen y escriben casos de cualquier cliente."""
    registro: list = []
    tools = por_nombre(crear_herramientas(repo, motor, confirmador_espia(registro)))
    ajeno = json.loads(tools["consultar_caso"].invoke({"id_caso": "CASO-0004"}))
    assert ajeno["ok"] is True
    assert ajeno["datos"]["cliente"] == "Colegio San Ramón"
    # El error específico del alcance global no cambia.
    faltante = json.loads(tools["consultar_caso"].invoke({"id_caso": "CASO-9999"}))
    assert faltante["ok"] is False
    assert "No existe el caso" in faltante["error"]
    # Y una escritura sobre un caso cualquiera sigue pidiendo confirmación y se
    # aplica cuando el humano aprueba.
    nota = json.loads(tools["agregar_nota"].invoke(
        {"id_caso": "CASO-0009", "nota": "Se respondió por correo."}
    ))
    assert nota["ok"] is True
    assert [accion for accion, _ in registro] == ["agregar_nota"]


# --- identificación del cliente de la sesión -----------------------------
def _con_identificacion(repo, motor):
    """Tools de sesión con el cliente leído en vivo desde un holder mutable."""
    actual: dict[str, str | None] = {"nombre": None}
    tools = por_nombre(crear_herramientas(
        repo, motor, None, alcance="sesion",
        cliente_actual=lambda: actual["nombre"],
        al_identificar=lambda nombre: actual.__setitem__("nombre", nombre),
    ))
    return tools, actual


def test_identificar_cliente_ata_la_sesion_y_permite_listar(repo, motor):
    repo.crear_caso(
        cliente="Ricardo", categoria="reparacion",
        descripcion="Notebook no enciende.",
    )
    tools, actual = _con_identificacion(repo, motor)
    assert json.loads(tools["listar_casos"].invoke({}))["ok"] is False

    # S1: el nombre viaja en el mensaje del turno, que es la única fuente de
    # identidad válida. Antes bastaba con que existiera en la base.
    with mensaje_del_turno("hola, soy ricardo, ¿cómo va mi reparación?"):
        identificado = json.loads(
            tools["identificar_cliente"].invoke({"nombre": "ricardo"})
        )
    assert identificado["ok"] is True
    assert identificado["datos"] == {"cliente": "Ricardo", "casos": 1}
    assert actual["nombre"] == "Ricardo"

    # El alcance cambió para las tools ya construidas: el mismo turno puede
    # listar los casos del cliente que acaba de identificarse.
    despues = json.loads(tools["listar_casos"].invoke({"limite": 5}))
    assert despues["ok"] is True
    assert [caso["id"] for caso in despues["datos"]["casos"]] == ["CASO-0011"]


def test_identificar_cliente_sin_casos_no_cambia_el_alcance(repo, motor):
    tools, actual = _con_identificacion(repo, motor)

    with mensaje_del_turno("me llamo ricardo"):
        resultado = json.loads(tools["identificar_cliente"].invoke({"nombre": "ricardo"}))
    assert resultado["ok"] is False
    # El nombre no se repite en el error, y la guía sigue siendo accionable.
    assert "ricardo" not in resultado["error"].lower()
    assert "CASO-" in resultado["error"]
    assert actual["nombre"] is None
    assert json.loads(tools["listar_casos"].invoke({}))["ok"] is False


def test_identificar_cliente_solo_existe_en_alcance_de_sesion(repo, motor):
    # La evaluación y la demo trabajan con alcance global: no identifican
    # clientes y su comportamiento no cambia.
    tools = por_nombre(crear_herramientas(repo, motor, None))
    assert "identificar_cliente" not in tools


# --- S1: identificación solo con el nombre declarado en el turno ---------
# La sesión real trae `correo = ricardo@testeo.com` en el contexto de memoria y
# el modelo lo usó para identificar a una persona que nunca dijo quién era. La
# herramienta no puede decidir identidad con datos de memoria: solo con el
# nombre que aparece en el mensaje del turno en curso.
def test_identificar_cliente_rechaza_un_correo_sin_repetirlo(repo, motor):
    repo.crear_caso(
        cliente="Ricardo", categoria="reparacion",
        descripcion="Notebook no enciende.",
    )
    tools, actual = _con_identificacion(repo, motor)

    with mensaje_del_turno("¿cómo va mi reparación? escríbeme a ricardo@testeo.com"):
        resultado = json.loads(
            tools["identificar_cliente"].invoke({"nombre": "ricardo@testeo.com"})
        )

    assert resultado["ok"] is False
    assert resultado["datos"] is None
    # El valor rechazado no se repite: repetirlo lo devuelve al modelo y al
    # historial, que es justo lo que la regla evita.
    assert "ricardo" not in resultado["error"].lower()
    assert "@" not in resultado["error"]
    # Ni el alcance ni el listado cambian: la sesión sigue sin identificar.
    assert actual["nombre"] is None
    assert json.loads(tools["listar_casos"].invoke({}))["ok"] is False


def test_identificar_cliente_rechaza_un_nombre_que_no_esta_en_el_turno(repo, motor):
    # El nombre existe en la base y en la memoria, pero la persona no lo dijo:
    # el turno no lo contiene, así que no identifica a nadie.
    repo.crear_caso(
        cliente="Ricardo", categoria="reparacion",
        descripcion="Notebook no enciende.",
    )
    tools, actual = _con_identificacion(repo, motor)

    with mensaje_del_turno("¿en qué estado está mi reparación?"):
        resultado = json.loads(
            tools["identificar_cliente"].invoke({"nombre": "Ricardo"})
        )

    assert resultado["ok"] is False
    assert resultado["datos"] is None
    assert "ricardo" not in resultado["error"].lower()
    assert actual["nombre"] is None
    assert json.loads(tools["listar_casos"].invoke({}))["ok"] is False


def test_identificar_cliente_acepta_el_nombre_declarado_sin_tildes_ni_mayusculas(repo, motor):
    tools, actual = _con_identificacion(repo, motor)

    # La persona lo escribe sin tilde y el modelo lo repite con tilde y en
    # mayúsculas: es el mismo nombre, y la comparación no puede depender de eso.
    with mensaje_del_turno("Hola, soy Maria Fernanda Rojas y consulto por mi equipo."):
        resultado = json.loads(
            tools["identificar_cliente"].invoke({"nombre": "MARÍA FERNANDA ROJAS"})
        )

    assert resultado["ok"] is True
    assert resultado["datos"] == {"cliente": "María Fernanda Rojas", "casos": 1}
    assert actual["nombre"] == "María Fernanda Rojas"
    listado = json.loads(tools["listar_casos"].invoke({}))
    assert listado["ok"] is True
    assert listado["datos"]["total"] == 1


def test_identificar_cliente_acepta_el_nombre_sin_tildes(repo, motor):
    """S2: la persona escribe `Maria` y el modelo declara `MARIA FERNANDA ROJAS`.

    El defecto observado: el filtro solo ignoraba mayúsculas, así que el nombre
    sin tilde no encontraba la fila canónica `María Fernanda Rojas`.
    """
    tools, actual = _con_identificacion(repo, motor)

    with mensaje_del_turno("Hola, soy Maria Fernanda Rojas y consulto por mi equipo."):
        resultado = json.loads(
            tools["identificar_cliente"].invoke({"nombre": "MARIA FERNANDA ROJAS"})
        )

    assert resultado["ok"] is True
    assert resultado["datos"] == {"cliente": "María Fernanda Rojas", "casos": 1}
    # El alcance queda atado al nombre canónico, y el listado del mismo turno
    # devuelve el caso con la grafía de la base.
    assert actual["nombre"] == "María Fernanda Rojas"
    listado = json.loads(tools["listar_casos"].invoke({}))
    assert listado["ok"] is True
    assert [caso["id"] for caso in listado["datos"]["casos"]] == ["CASO-0001"]
    assert listado["datos"]["casos"][0]["cliente"] == "María Fernanda Rojas"


def test_el_guard_de_sesion_compara_sin_tildes_y_sigue_cerrando_lo_ajeno(repo, motor):
    """S3: el guard de propiedad usa la misma comparación que el filtro SQL."""
    tools = por_nombre(crear_herramientas(
        repo, motor, None, alcance="sesion",
        cliente_actual="MARIA FERNANDA ROJAS",
    ))

    propio = json.loads(tools["consultar_caso"].invoke({"id_caso": "CASO-0001"}))
    assert propio["ok"] is True
    assert propio["datos"]["cliente"] == "María Fernanda Rojas"

    # Un caso de otro cliente sigue bloqueado, y el bloqueo no filtra datos.
    ajeno = json.loads(tools["consultar_caso"].invoke({"id_caso": "CASO-0004"}))
    assert ajeno["ok"] is False
    assert ajeno["datos"] is None
    assert "Colegio San Ramón" not in ajeno["error"]
    assert "mantenimiento preventivo" not in ajeno["error"]

    # La escritura ajena tampoco se ejecuta ni pide confirmación.
    registro: list = []
    con_confirmacion = por_nombre(crear_herramientas(
        repo, motor, confirmador_espia(registro), alcance="sesion",
        cliente_actual="MARIA FERNANDA ROJAS",
    ))
    negada = json.loads(con_confirmacion["agregar_nota"].invoke(
        {"id_caso": "CASO-0004", "nota": "Nota no autorizada."}
    ))
    assert negada["ok"] is False
    assert registro == []


def test_dos_clientes_parecidos_siguen_separados_en_el_alcance(repo, motor):
    """S4: la comparación sin tildes no une a dos clientes distintos."""
    tools = por_nombre(crear_herramientas(
        repo, motor, None, alcance="sesion",
        cliente_actual="Maria Fernanda Rojas Silva",
    ))

    listado = json.loads(tools["listar_casos"].invoke({}))
    assert listado["ok"] is True
    assert listado["datos"] == {"total": 0, "casos": []}

    ajeno = json.loads(tools["consultar_caso"].invoke({"id_caso": "CASO-0001"}))
    assert ajeno["ok"] is False
    assert ajeno["datos"] is None


def test_identificar_cliente_no_acepta_un_nombre_embebido_en_un_correo(repo, motor):
    # «ricardo» está dentro de un correo que aparece en el mensaje, pero la
    # persona no se identificó: un correo no identifica en este dominio y el
    # nombre suelto no es una declaración de identidad.
    repo.crear_caso(
        cliente="Ricardo", categoria="reparacion",
        descripcion="Notebook no enciende.",
    )
    tools, actual = _con_identificacion(repo, motor)

    with mensaje_del_turno("escríbeme a ricardo@testeo.com cuando lo revises"):
        resultado = json.loads(
            tools["identificar_cliente"].invoke({"nombre": "ricardo"})
        )

    assert resultado["ok"] is False
    assert resultado["datos"] is None
    assert "ricardo" not in resultado["error"].lower()
    assert actual["nombre"] is None


def test_identificar_cliente_no_repite_el_nombre_sin_casos(repo, motor):
    tools, actual = _con_identificacion(repo, motor)

    with mensaje_del_turno("soy pepito"):
        resultado = json.loads(
            tools["identificar_cliente"].invoke({"nombre": "pepito"})
        )

    assert resultado["ok"] is False
    assert resultado["datos"] is None
    assert "pepito" not in resultado["error"].lower()
    assert actual["nombre"] is None


def test_identificar_cliente_sin_mensaje_de_turno_falla_cerrado(repo, motor):
    # Sin mensaje publicado (una tool invocada fuera del turno) no hay fuente
    # de identidad: la respuesta segura es no identificar.
    repo.crear_caso(
        cliente="Ricardo", categoria="reparacion",
        descripcion="Notebook no enciende.",
    )
    tools, actual = _con_identificacion(repo, motor)

    resultado = json.loads(tools["identificar_cliente"].invoke({"nombre": "Ricardo"}))

    assert resultado["ok"] is False
    assert resultado["datos"] is None
    assert actual["nombre"] is None


def test_el_mensaje_del_turno_no_sobrevive_al_turno(repo, motor):
    repo.crear_caso(
        cliente="Ricardo", categoria="reparacion",
        descripcion="Notebook no enciende.",
    )
    tools, actual = _con_identificacion(repo, motor)

    with mensaje_del_turno("soy ricardo"):
        assert json.loads(
            tools["identificar_cliente"].invoke({"nombre": "Ricardo"})
        )["ok"] is True
    assert actual["nombre"] == "Ricardo"

    # El turno siguiente no hereda el mensaje anterior: la mención no se
    # reutiliza para volver a identificar sin que la persona lo diga.
    with mensaje_del_turno("¿y el plazo?"):
        otro = json.loads(tools["identificar_cliente"].invoke({"nombre": "Ricardo"}))
    assert otro["ok"] is False
    assert actual["nombre"] == "Ricardo"


# --- S3: cada tool declara en un solo lugar qué devuelve en `datos` ------
CAMPOS_DE_CASO = ("id", "cliente", "categoria", "descripcion", "prioridad",
                  "estado", "procedimiento", "fecha_creacion", "notas")


def test_la_declaracion_de_salidas_cubre_cada_tool_publicada(repo, motor):
    from src.herramientas import CLAVES_SALIDA

    publicadas = {herramienta.name for herramienta in crear_herramientas(
        repo, motor, confirmador_si, alcance="sesion",
        al_identificar=lambda nombre: None,
    )}
    publicadas.add("recordar")  # la agrega el agente cuando tiene memoria
    assert publicadas <= set(CLAVES_SALIDA)
    # Una declaración vacía no sirve para validar un camino.
    assert all(CLAVES_SALIDA[nombre] for nombre in CLAVES_SALIDA)


def test_la_descripcion_de_cada_tool_documenta_las_claves_que_devuelve(repo, motor):
    from src.herramientas import CLAVES_SALIDA

    for herramienta in crear_herramientas(
        repo, motor, confirmador_si, alcance="sesion",
        al_identificar=lambda nombre: None,
    ):
        descripcion = herramienta.description or ""
        for clave in CLAVES_SALIDA[herramienta.name]:
            assert clave in descripcion, f"{herramienta.name} no documenta {clave}"


class MemoriaFalsa:
    """Doble de `Memoria` para la tool `recordar`."""

    def recordar(self, clave, valor):
        return {"usuario": "local", "clave": clave, "valor": valor,
                "actualizado_en": "2026-01-01T00:00:00"}


def test_lo_declarado_coincide_con_lo_que_la_tool_devuelve(repo, motor):
    """La declaración no puede derivar de la salida real de la tool."""
    from src.herramientas import CLAVES_SALIDA, crear_herramienta_recordar

    # Con alcance de sesión y un cliente con casos: las tools que operan sobre
    # un caso necesitan un dueño, y el cliente es el de la semilla.
    dueno = "María Fernanda Rojas"
    tools = por_nombre(crear_herramientas(
        repo, motor, confirmador_si, alcance="sesion",
        cliente_actual=dueno, al_identificar=lambda nombre: None,
    ))
    tools["recordar"] = crear_herramienta_recordar(MemoriaFalsa())
    invocaciones = {
        "buscar_documentacion": {"pregunta": "¿cuánto demora el diagnóstico?"},
        "consultar_caso": {"id_caso": "CASO-0001"},
        "listar_casos": {"limite": 2},
        "crear_caso": {"cliente": dueno, "categoria": "recepcion",
                       "descripcion": "Notebook no enciende."},
        "agregar_nota": {"id_caso": "CASO-0001", "nota": "Se respondió."},
        "actualizar_estado": {"id_caso": "CASO-0001", "estado": "cerrado"},
        "escalar_a_humano": {"id_caso": "CASO-0001", "motivo": "Urgente."},
        "calcular_plazo": {"tipo_servicio": "reparacion"},
        "identificar_cliente": {"nombre": dueno},
        "recordar": {"clave": "correo", "valor": "ricardo@testeo.com"},
        "pedir_datos": {"pregunta": "¿Cuál es el número de caso?"},
    }
    assert set(invocaciones) == set(CLAVES_SALIDA)

    for nombre, argumentos in invocaciones.items():
        with mensaje_del_turno(f"hola, soy {dueno}"):
            payload = json.loads(tools[nombre].invoke(argumentos))
        assert payload["ok"] is True, nombre
        assert set(payload["datos"]) == set(CLAVES_SALIDA[nombre]), nombre

    # Los campos del caso, en la declaración y en la base, son los mismos.
    assert set(CLAVES_SALIDA["consultar_caso"]) == set(CAMPOS_DE_CASO)


def test_pedir_datos_rechaza_una_pregunta_en_blanco(repo, motor):
    # `min_length` cuenta espacios; una pregunta sin contenido no puede llegar
    # al usuario como si fuera una pregunta válida.
    tools = por_nombre(crear_herramientas(repo, motor))
    with pytest.raises(ValidationError):
        tools["pedir_datos"].invoke({"pregunta": "   "})


def test_pedir_datos_conserva_la_pregunta_exacta(repo, motor):
    tools = por_nombre(crear_herramientas(repo, motor))
    pregunta = "  ¿Cuál es el número de caso?  "
    payload = json.loads(tools["pedir_datos"].invoke({"pregunta": pregunta}))

    assert payload["datos"]["pregunta"] == pregunta


def test_pedir_datos_devuelve_la_pregunta_tal_cual(repo, motor):
    tools = por_nombre(crear_herramientas(repo, motor))
    pregunta = "¿Cuál es el número de caso o el nombre del cliente?"
    payload = json.loads(tools["pedir_datos"].invoke({"pregunta": pregunta}))

    assert payload["ok"] is True
    assert payload["datos"] == {"pregunta": pregunta}
    assert payload["error"] is None
