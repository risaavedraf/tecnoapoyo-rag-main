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
    assert [caso["id"] for caso in abiertos.datos["casos"]] == [
        "CASO-0004", "CASO-0007", "CASO-0009",
    ]
    invalido = listar_casos(repo, estado="pendiente")
    assert invalido.ok is False
    assert invalido.error is not None
    assert "no permitido" in invalido.error


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


def test_expone_las_ocho_herramientas_con_su_nombre(tools):
    assert sorted(tools) == sorted([
        "buscar_documentacion", "consultar_caso", "listar_casos",
        "crear_caso", "agregar_nota", "actualizar_estado",
        "escalar_a_humano", "calcular_plazo",
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
