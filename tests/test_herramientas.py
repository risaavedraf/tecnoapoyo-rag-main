"""Contratos de las herramientas del agente (RA2 / IL2.1).

Sin red: el motor RAG se reemplaza por un doble que implementa `recuperar()`,
igual que hizo la suite del propio motor. Lo que se verifica aquí es el
contrato que el agente consume: validación de argumentos, abstención
propagada, y que ninguna escritura ocurra sin confirmación.
"""
import pytest
from pydantic import ValidationError

from src import config
from src.casos import RepositorioCasos
from src.herramientas import (
    PLAZOS_HABILES,
    ResultadoHerramienta,
    calcular_plazo,
    crear_herramientas,
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


@pytest.fixture
def repo(tmp_path):
    return RepositorioCasos(ruta=tmp_path / "casos.db")


@pytest.fixture
def motor():
    return MotorFalso(fragmentos=[dict(FRAGMENTO)])


@pytest.fixture
def tools(repo, motor):
    return por_nombre(crear_herramientas(repo, motor))


def test_expone_las_ocho_herramientas_con_su_nombre(tools):
    assert sorted(tools) == sorted([
        "buscar_documentacion", "consultar_caso", "listar_casos",
        "crear_caso", "agregar_nota", "actualizar_estado",
        "escalar_a_humano", "calcular_plazo",
    ])


# --- consulta -------------------------------------------------------------
def test_buscar_documentacion_delega_en_el_motor_y_respeta_k(tools, motor):
    resultado = tools["buscar_documentacion"].invoke(
        {"pregunta": "¿cuánto demora el diagnóstico?", "k": 5}
    )
    assert isinstance(resultado, ResultadoHerramienta)
    assert resultado.ok is True
    assert motor.llamadas == [("¿cuánto demora el diagnóstico?", 5)]
    assert resultado.datos["fragmentos"] == [FRAGMENTO]
    assert resultado.datos["mejor_score"] == 0.88
    assert resultado.datos["umbral"] == config.UMBRAL_ABSTENCION
    assert resultado.datos["evidencia_suficiente"] is True


def test_buscar_documentacion_propaga_la_abstencion_del_motor(repo):
    motor_bajo = MotorFalso(fragmentos=[{**FRAGMENTO, "score": 0.10}])
    tools = por_nombre(crear_herramientas(repo, motor_bajo))
    resultado = tools["buscar_documentacion"].invoke({"pregunta": "¿reparan tablets?"})
    assert resultado.ok is True
    assert resultado.datos["evidencia_suficiente"] is False
    assert resultado.datos["mejor_score"] == 0.10


def test_buscar_documentacion_sin_resultados_no_tiene_evidencia(repo):
    tools = por_nombre(crear_herramientas(repo, MotorFalso()))
    resultado = tools["buscar_documentacion"].invoke({"pregunta": "¿hay soporte los domingos?"})
    assert resultado.datos["mejor_score"] == 0.0
    assert resultado.datos["evidencia_suficiente"] is False


def test_buscar_documentacion_reporta_fallo_del_indice(repo):
    motor_roto = MotorFalso(excepcion=FileNotFoundError("No existe el índice FAISS"))
    tools = por_nombre(crear_herramientas(repo, motor_roto))
    resultado = tools["buscar_documentacion"].invoke({"pregunta": "¿cuánto demora?"})
    assert resultado.ok is False
    assert "No existe el índice FAISS" in resultado.error


@pytest.mark.parametrize("argumentos", [
    {"pregunta": "ab"},                      # pregunta demasiado corta
    {"pregunta": "¿cuánto demora?", "k": 0},  # k fuera de rango
    {"pregunta": "¿cuánto demora?", "k": 11},
])
def test_buscar_documentacion_rechaza_argumentos_invalidos(tools, argumentos):
    with pytest.raises(ValidationError):
        tools["buscar_documentacion"].invoke(argumentos)


def test_consultar_caso_existente_e_inexistente(tools):
    encontrado = tools["consultar_caso"].invoke({"id_caso": "caso-0001"})
    assert encontrado.ok is True
    assert encontrado.datos["id"] == "CASO-0001"
    faltante = tools["consultar_caso"].invoke({"id_caso": "CASO-9999"})
    assert faltante.ok is False
    assert "CASO-9999" in faltante.error


def test_consultar_caso_rechaza_id_con_formato_invalido(tools):
    with pytest.raises(ValidationError):
        tools["consultar_caso"].invoke({"id_caso": "0001"})


def test_listar_casos_filtra_y_reporta_estado_invalido(tools):
    abiertos = tools["listar_casos"].invoke({"estado": "abierto"})
    assert abiertos.ok is True
    assert abiertos.datos["total"] == 3
    assert [caso["id"] for caso in abiertos.datos["casos"]] == [
        "CASO-0004", "CASO-0007", "CASO-0009",
    ]
    invalido = tools["listar_casos"].invoke({"estado": "pendiente"})
    assert invalido.ok is False
    assert "no permitido" in invalido.error


# --- escritura: la confirmación es obligatoria ----------------------------
def test_crear_caso_sin_confirmacion_no_escribe(tools, repo):
    antes = repo.contar()
    resultado = tools["crear_caso"].invoke({
        "cliente": "Cliente Prueba",
        "categoria": "recepcion",
        "descripcion": "Notebook no enciende.",
        "prioridad": "alta",
    })
    assert resultado.ok is False
    assert "crear_caso" in resultado.error
    assert "confirmación" in resultado.error
    assert repo.contar() == antes


def test_crear_caso_confirmado_escribe_y_persiste(tools, repo):
    resultado = tools["crear_caso"].invoke({
        "cliente": "Cliente Prueba",
        "categoria": "recepcion",
        "descripcion": "Notebook no enciende.",
        "prioridad": "alta",
        "procedimiento": "proc-001",
        "confirmado": True,
    })
    assert resultado.ok is True
    assert resultado.datos["id"] == "CASO-0011"
    assert resultado.datos["procedimiento"] == "PROC-001"
    assert repo.contar() == 11


def test_crear_caso_rechaza_prioridad_invalida_antes_de_escribir(tools, repo):
    antes = repo.contar()
    with pytest.raises(ValidationError):
        tools["crear_caso"].invoke({
            "cliente": "Cliente Prueba",
            "categoria": "recepcion",
            "descripcion": "Notebook no enciende.",
            "prioridad": "urgentísima",
            "confirmado": True,
        })
    assert repo.contar() == antes


def test_agregar_nota_sin_confirmacion_no_modifica_el_caso(tools, repo):
    resultado = tools["agregar_nota"].invoke({
        "id_caso": "CASO-0009", "nota": "Se respondió por correo.",
    })
    assert resultado.ok is False
    assert repo.obtener_caso("CASO-0009")["notas"] == []


def test_agregar_nota_confirmada_escribe(tools, repo):
    resultado = tools["agregar_nota"].invoke({
        "id_caso": "CASO-0009", "nota": "Se respondió por correo.",
        "confirmado": True,
    })
    assert resultado.ok is True
    assert repo.obtener_caso("CASO-0009")["notas"] == ["Se respondió por correo."]


def test_actualizar_estado_sin_confirmacion_no_cambia_el_estado(tools, repo):
    resultado = tools["actualizar_estado"].invoke({
        "id_caso": "CASO-0004", "estado": "cerrado",
    })
    assert resultado.ok is False
    assert repo.obtener_caso("CASO-0004")["estado"] == "abierto"


def test_actualizar_estado_confirmado_cambia_y_persiste(tools, repo):
    resultado = tools["actualizar_estado"].invoke({
        "id_caso": "CASO-0004", "estado": "en_progreso", "confirmado": True,
    })
    assert resultado.ok is True
    assert repo.obtener_caso("CASO-0004")["estado"] == "en_progreso"


def test_actualizar_estado_rechaza_estado_fuera_de_allowlist(tools, repo):
    with pytest.raises(ValidationError):
        tools["actualizar_estado"].invoke({
            "id_caso": "CASO-0004", "estado": "en_limbo", "confirmado": True,
        })
    assert repo.obtener_caso("CASO-0004")["estado"] == "abierto"


def test_escalar_sin_confirmacion_no_cambia_nada(tools, repo):
    antes = repo.obtener_caso("CASO-0007")
    resultado = tools["escalar_a_humano"].invoke({
        "id_caso": "CASO-0007", "motivo": "Cliente solicita atención inmediata.",
    })
    assert resultado.ok is False
    assert repo.obtener_caso("CASO-0007") == antes


def test_escalar_confirmado_deja_estado_y_motivo(tools, repo):
    resultado = tools["escalar_a_humano"].invoke({
        "id_caso": "CASO-0007", "motivo": "Cliente solicita atención inmediata.",
        "confirmado": True,
    })
    assert resultado.ok is True
    caso = repo.obtener_caso("CASO-0007")
    assert caso["estado"] == "escalado"
    assert "Escalado: Cliente solicita atención inmediata." in caso["notas"]


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
    # Sábado 2026-03-07 + 1 día hábil = lunes 2026-03-09.
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
