"""Evaluación del agente (RA2 / IL2.1-IL2.4): puntuación y agregación.

Ningún test toca la red ni construye el agente: las funciones de puntuación son
puras sobre la traza (`RespuestaAgente`/`PasoAgente`), y el camino offline del
CLI (`--verificar-dataset`) solo lee el JSON del contrato.
"""
import json
from typing import Any

import pytest

from scripts import evaluar_agente
from scripts.evaluar_agente import (
    RUTA_DATASET,
    agregar_metricas,
    cargar_dataset,
    main,
    puntuar_tarea,
    validar_dataset,
)
from src.agente import PasoAgente, RespuestaAgente


# --------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------
def paso(numero, herramienta, argumentos, ok=True, error=None, salida=None):
    return PasoAgente(
        numero=numero,
        herramienta=herramienta,
        argumentos=argumentos,
        salida=salida,
        error=error,
        ok=ok,
    )


def respuesta(*pasos, abstencion=False, texto="Listo.", plan=None,
              replanes=0, tokens=0, latencia=0.0, error=None):
    return RespuestaAgente(
        pregunta="?",
        respuesta=texto,
        abstencion=abstencion,
        pasos=list(pasos),
        iteraciones=len(pasos),
        tokens_total=tokens,
        latencia_seg=latencia,
        error=error,
        replanes=replanes,
        plan=plan,
    )


def tarea(tipo="seleccion_herramienta", herramienta=None, argumentos=None,
          abstencion=False, confirmacion=False, min_pasos=None,
          turnos_previos=None, id_tarea="TX", pregunta="¿consulta?"):
    return {
        "id": id_tarea,
        "tipo": tipo,
        "pregunta": pregunta,
        "turnos_previos": turnos_previos or [],
        "esperado": {
            "herramienta": herramienta,
            "argumentos_clave": argumentos or {},
            "abstencion": abstencion,
            "requiere_confirmacion": confirmacion,
            "min_pasos_plan": min_pasos,
        },
    }


@pytest.fixture
def dataset_real():
    return cargar_dataset()


# --------------------------------------------------------------------------
# Contrato del dataset
# --------------------------------------------------------------------------
def test_ruta_dataset_es_la_del_contrato():
    assert RUTA_DATASET == evaluar_agente.config.RAIZ / "evaluation" / "dataset_agente.json"
    assert RUTA_DATASET.is_file()


def test_dataset_real_es_valido(dataset_real):
    assert validar_dataset(dataset_real) == []
    assert len(dataset_real["tareas"]) == 14


def test_cargar_dataset_desde_otra_ruta(tmp_path):
    ruta = tmp_path / "mini.json"
    ruta.write_text(
        json.dumps({"tareas": [tarea()]}, ensure_ascii=False), encoding="utf-8"
    )
    assert len(cargar_dataset(ruta)["tareas"]) == 1


def test_validar_dataset_detecta_tipo_desconocido(dataset_real):
    datos = json.loads(json.dumps(dataset_real))
    datos["tareas"][0]["tipo"] = "conversacion"
    errores = validar_dataset(datos)
    assert any("tipo desconocido" in error for error in errores)


@pytest.mark.parametrize(
    "campo", ["herramienta", "argumentos_clave", "abstencion",
              "requiere_confirmacion", "min_pasos_plan"]
)
def test_validar_dataset_detecta_campo_faltante_en_esperado(dataset_real, campo):
    datos = json.loads(json.dumps(dataset_real))
    del datos["tareas"][2]["esperado"][campo]
    errores = validar_dataset(datos)
    assert any(campo in error and "T03" in error for error in errores)


def test_validar_dataset_detecta_ids_duplicados(dataset_real):
    datos = json.loads(json.dumps(dataset_real))
    datos["tareas"][1]["id"] = datos["tareas"][0]["id"]
    errores = validar_dataset(datos)
    assert any("duplicado" in error.lower() for error in errores)


def test_validar_dataset_detecta_estructura_ausente():
    lista_en_lugar_de_objeto: Any = [{"id": "T01"}]
    assert validar_dataset({}) != []
    assert validar_dataset({"tareas": []}) != []
    assert validar_dataset(lista_en_lugar_de_objeto) != []


def test_validar_dataset_detecta_min_pasos_invalido(dataset_real):
    datos = json.loads(json.dumps(dataset_real))
    datos["tareas"][12]["esperado"]["min_pasos_plan"] = 0
    assert any("min_pasos_plan" in error for error in validar_dataset(datos))


# --------------------------------------------------------------------------
# Selección de herramienta
# --------------------------------------------------------------------------
def test_herramienta_correcta_acierta():
    puntaje = puntuar_tarea(
        tarea(herramienta="consultar_caso"),
        respuesta(paso(1, "consultar_caso", {"id_caso": "CASO-0004"})),
    )
    assert puntaje["herramienta_ok"] is True
    assert puntaje["motivos"] == []


def test_herramienta_equivocada_falla():
    puntaje = puntuar_tarea(
        tarea(herramienta="consultar_caso"),
        respuesta(paso(1, "listar_casos", {"estado": "abierto"})),
    )
    assert puntaje["herramienta_ok"] is False
    assert "herramienta" in puntaje["motivos"]


def test_sin_herramienta_esperada_no_exige_ninguna():
    puntaje = puntuar_tarea(
        tarea(tipo="abstencion", herramienta=None, abstencion=True),
        respuesta(abstencion=True, texto="No tengo información suficiente"),
    )
    assert puntaje["herramienta_ok"] is True


def test_escritura_denegada_cuenta_como_seleccion_correcta():
    """La tool responde ok=False al denegar, pero el agente sí la eligió."""
    puntaje = puntuar_tarea(
        tarea(tipo="escritura", herramienta="crear_caso",
              argumentos={"cliente": "Juan Pérez"}, confirmacion=True),
        respuesta(paso(1, "crear_caso", {"cliente": "Juan Perez"},
                       ok=False, error="requiere confirmación")),
        escrituras_antes=10,
        escrituras_despues=10,
    )
    assert puntaje["herramienta_ok"] is True
    assert puntaje["argumentos_ok"] is True
    assert puntaje["escritura_intentada"] is True
    assert puntaje["confirmacion_respetada"] is True


# --------------------------------------------------------------------------
# Argumentos
# --------------------------------------------------------------------------
def test_argumento_acierta_sin_acentos_ni_mayusculas():
    puntaje = puntuar_tarea(
        tarea(herramienta="buscar_documentacion",
              argumentos={"pregunta": "garantía"}),
        respuesta(paso(1, "buscar_documentacion",
                       {"pregunta": "Qué cubre la GARANTIA de fabrica"})),
    )
    assert puntaje["argumentos_ok"] is True


def test_argumento_incorrecto_falla():
    puntaje = puntuar_tarea(
        tarea(herramienta="buscar_documentacion",
              argumentos={"pregunta": "garantía"}),
        respuesta(paso(1, "buscar_documentacion", {"pregunta": "horarios"})),
    )
    assert puntaje["argumentos_ok"] is False
    assert "argumentos" in puntaje["motivos"]


def test_argumento_ausente_falla():
    puntaje = puntuar_tarea(
        tarea(herramienta="listar_casos", argumentos={"estado": "abierto"}),
        respuesta(paso(1, "listar_casos", {"limite": 5})),
    )
    assert puntaje["argumentos_ok"] is False


def test_id_caso_se_compara_en_mayusculas_exactas():
    esperada = tarea(herramienta="consultar_caso",
                     argumentos={"id_caso": "CASO-0004"})
    assert puntuar_tarea(
        esperada, respuesta(paso(1, "consultar_caso", {"id_caso": "caso-0004"}))
    )["argumentos_ok"] is True
    assert puntuar_tarea(
        esperada, respuesta(paso(1, "consultar_caso", {"id_caso": "CASO-00040"}))
    )["argumentos_ok"] is False


def test_argumentos_correctos_en_un_paso_posterior_cuentan():
    puntaje = puntuar_tarea(
        tarea(herramienta="calcular_plazo",
              argumentos={"tipo_servicio": "reparación"}),
        respuesta(
            paso(1, "calcular_plazo", {"tipo_servicio": "diagnostico"}),
            paso(2, "calcular_plazo", {"tipo_servicio": "Reparacion"}),
        ),
    )
    assert puntaje["argumentos_ok"] is True


# --------------------------------------------------------------------------
# Abstención
# --------------------------------------------------------------------------
def test_abstencion_correcta_acierta():
    puntaje = puntuar_tarea(
        tarea(tipo="abstencion", abstencion=True),
        respuesta(abstencion=True),
    )
    assert puntaje["abstencion_ok"] is True


def test_abstencion_falsa_falla():
    """Se abstuvo cuando debía responder."""
    puntaje = puntuar_tarea(
        tarea(herramienta="buscar_documentacion", abstencion=False),
        respuesta(abstencion=True),
    )
    assert puntaje["abstencion_ok"] is False
    assert puntaje["abstencion_esperada"] is False
    assert puntaje["abstencion_obtenida"] is True


def test_abstencion_omitida_falla():
    """No se abstuvo cuando debía abstenerse."""
    puntaje = puntuar_tarea(
        tarea(tipo="abstencion", abstencion=True),
        respuesta(abstencion=False, texto="Sí, reparamos tablets."),
    )
    assert puntaje["abstencion_ok"] is False
    assert "abstencion" in puntaje["motivos"]


# --------------------------------------------------------------------------
# Memoria
# --------------------------------------------------------------------------
TURNOS_CASO = [
    {"rol": "usuario", "contenido": "Mi caso es el CASO-0004"},
    {"rol": "asistente", "contenido": "Anotado: CASO-0004."},
]
TURNOS_EQUIPO = [
    {"rol": "usuario", "contenido": "Mi equipo es un Lenovo IdeaPad"},
    {"rol": "asistente", "contenido": "Anotado: Lenovo IdeaPad."},
]


def test_memoria_usada_desde_los_argumentos_acierta():
    puntaje = puntuar_tarea(
        tarea(tipo="memoria", herramienta="consultar_caso",
              argumentos={"id_caso": "CASO-0004"},
              turnos_previos=TURNOS_CASO, pregunta="¿Y en qué estado está?"),
        respuesta(paso(1, "consultar_caso", {"id_caso": "CASO-0004"}),
                  texto="El caso CASO-0004 está abierto."),
    )
    assert puntaje["memoria_ok"] is True


def test_memoria_usada_solo_en_la_respuesta_acierta():
    puntaje = puntuar_tarea(
        tarea(tipo="memoria", turnos_previos=TURNOS_EQUIPO,
              pregunta="¿Qué equipo tengo registrado?"),
        respuesta(texto="Tu equipo registrado es un Lenovo IdeaPad."),
    )
    assert puntaje["memoria_ok"] is True


def test_memoria_no_usada_falla():
    puntaje = puntuar_tarea(
        tarea(tipo="memoria", turnos_previos=TURNOS_EQUIPO,
              pregunta="¿Qué equipo tengo registrado?"),
        respuesta(texto="No tengo esa información disponible."),
    )
    assert puntaje["memoria_ok"] is False
    assert "memoria" in puntaje["motivos"]


def test_memoria_con_abstencion_falla():
    puntaje = puntuar_tarea(
        tarea(tipo="memoria", turnos_previos=TURNOS_CASO,
              pregunta="¿Y en qué estado está?"),
        respuesta(abstencion=True, texto="Lenovo CASO-0004"),
    )
    assert puntaje["memoria_ok"] is False


def test_memoria_no_aplica_a_otras_tareas():
    puntaje = puntuar_tarea(
        tarea(tipo="seleccion_herramienta"),
        respuesta(texto="Nada que ver con el contexto."),
    )
    assert puntaje["memoria_ok"] is True


# --------------------------------------------------------------------------
# Confirmación de escritura
# --------------------------------------------------------------------------
def test_escritura_ejecutada_sin_confirmacion_falla():
    """El conteo cambió: la herramienta escribió aunque el humano no aprobó."""
    puntaje = puntuar_tarea(
        tarea(tipo="escritura", herramienta="crear_caso", confirmacion=True),
        respuesta(paso(1, "crear_caso", {"cliente": "Juan Pérez"}, ok=True)),
        escrituras_antes=10,
        escrituras_despues=11,
    )
    assert puntaje["escritura_intentada"] is True
    assert puntaje["confirmacion_respetada"] is False
    assert "confirmacion" in puntaje["motivos"]


def test_escritura_denegada_sin_conteos_queda_sin_verificar():
    """Con intento pero sin conteos el resultado es desconocido, no un fallo.

    Contrato corregido: `confirmacion_respetada` solo es True/False cuando se
    pudo verificar el conteo. Antes se fallaba cerrado y se mezclaba "no
    verificado" con "no respetada".
    """
    puntaje = puntuar_tarea(
        tarea(tipo="escritura", herramienta="crear_caso", confirmacion=True),
        respuesta(paso(1, "crear_caso", {"cliente": "Juan Pérez"}, ok=False)),
    )
    assert puntaje["escritura_intentada"] is True
    assert puntaje["confirmacion_respetada"] is None
    assert "confirmacion" not in puntaje["motivos"]


def test_escritura_denegada_y_conteo_estable_acierta():
    puntaje = puntuar_tarea(
        tarea(tipo="escritura", herramienta="crear_caso", confirmacion=True),
        respuesta(paso(1, "crear_caso", {"cliente": "Juan Pérez"}, ok=False,
                       error="requiere confirmación")),
        escrituras_antes=7,
        escrituras_despues=7,
    )
    assert puntaje["confirmacion_respetada"] is True


def test_sin_requerir_confirmacion_no_pide_escritura():
    puntaje = puntuar_tarea(
        tarea(herramienta="consultar_caso"),
        respuesta(paso(1, "consultar_caso", {"id_caso": "CASO-0004"})),
        escrituras_antes=7,
        escrituras_despues=7,
    )
    # Sin intento de escritura no hay confirmación que evaluar: `n/d`, no True.
    assert puntaje["escritura_intentada"] is False
    assert puntaje["confirmacion_respetada"] is None


def test_escritura_que_nunca_se_intento_no_falla_la_confirmacion():
    """La tarea pedía escritura pero el agente nunca llegó a escribir.

    No escribir es correcto; la confirmación simplemente no aplica. Antes esta
    tarea contaba como fallo de confirmación.
    """
    puntaje = puntuar_tarea(
        tarea(tipo="escritura", herramienta="crear_caso", confirmacion=True),
        respuesta(abstencion=True, texto="No puedo crear el caso sin el equipo."),
        escrituras_antes=7,
        escrituras_despues=7,
    )
    assert puntaje["requiere_confirmacion"] is True
    assert puntaje["escritura_intentada"] is False
    assert puntaje["confirmacion_respetada"] is None
    assert "confirmacion" not in puntaje["motivos"]


def test_escritura_intentada_con_conteo_igual_confirma():
    puntaje = puntuar_tarea(
        tarea(tipo="escritura", herramienta="crear_caso", confirmacion=True),
        respuesta(paso(1, "crear_caso", {"cliente": "Juan Pérez"}, ok=False,
                       error="requiere confirmación")),
        escrituras_antes=7,
        escrituras_despues=7,
    )
    assert puntaje["escritura_intentada"] is True
    assert puntaje["confirmacion_respetada"] is True


def test_escritura_intentada_con_conteo_distinto_no_confirma():
    puntaje = puntuar_tarea(
        tarea(tipo="escritura", herramienta="crear_caso", confirmacion=True),
        respuesta(paso(1, "crear_caso", {"cliente": "Juan Pérez"}, ok=False)),
        escrituras_antes=7,
        escrituras_despues=8,
    )
    assert puntaje["escritura_intentada"] is True
    assert puntaje["confirmacion_respetada"] is False
    assert "confirmacion" in puntaje["motivos"]


# --------------------------------------------------------------------------
# Plan
# --------------------------------------------------------------------------
def test_plan_valido_acierta():
    puntaje = puntuar_tarea(
        tarea(tipo="planificacion", min_pasos=2),
        respuesta(
            paso(1, "consultar_caso", {"id_caso": "CASO-0004"}),
            paso(2, "calcular_plazo", {"tipo_servicio": "reparacion"}),
            plan=object(),
        ),
    )
    assert puntaje["plan_ok"] is True


def test_plan_invalido_por_menos_pasos_falla():
    puntaje = puntuar_tarea(
        tarea(tipo="planificacion", min_pasos=2),
        respuesta(paso(1, "consultar_caso", {"id_caso": "CASO-0004"}),
                  plan=object()),
    )
    assert puntaje["plan_ok"] is False
    assert "plan" in puntaje["motivos"]


def test_plan_ausente_falla():
    puntaje = puntuar_tarea(
        tarea(tipo="planificacion", min_pasos=2),
        respuesta(
            paso(1, "consultar_caso", {"id_caso": "CASO-0004"}),
            paso(2, "calcular_plazo", {"tipo_servicio": "reparacion"}),
            plan=None,
        ),
    )
    assert puntaje["plan_ok"] is False


def test_plan_no_exigido_no_se_evalua():
    puntaje = puntuar_tarea(tarea(min_pasos=None), respuesta(plan=None))
    assert puntaje["plan_ok"] is True


def test_plan_no_aplica_en_modo_reactivo():
    """Reactivo no tiene planificador: la métrica no aplica, no es un fallo."""
    puntaje = puntuar_tarea(
        tarea(tipo="planificacion", min_pasos=2),
        respuesta(paso(1, "consultar_caso", {"id_caso": "CASO-0004"}), plan=None),
        modo="reactivo",
    )
    assert puntaje["plan_aplica"] is False
    assert puntaje["plan_ok"] is None
    assert "plan" not in puntaje["motivos"]


def test_plan_en_planificado_se_calcula_como_antes():
    puntaje = puntuar_tarea(
        tarea(tipo="planificacion", min_pasos=2),
        respuesta(paso(1, "consultar_caso", {"id_caso": "CASO-0004"}), plan=None),
        modo="planificado",
    )
    assert puntaje["plan_aplica"] is True
    assert puntaje["plan_ok"] is False
    assert "plan" in puntaje["motivos"]


# --------------------------------------------------------------------------
# Forma del puntaje
# --------------------------------------------------------------------------
def test_puntaje_trae_los_campos_del_contrato():
    puntaje = puntuar_tarea(
        tarea(herramienta="consultar_caso"),
        respuesta(paso(1, "consultar_caso", {"id_caso": "CASO-0004"}),
                  replanes=2, tokens=120, latencia=1.5),
    )
    for campo in (
        "id", "tipo", "herramienta_ok", "argumentos_ok", "abstencion_ok",
        "memoria_ok", "escritura_intentada", "confirmacion_respetada",
        "plan_aplica", "plan_ok", "pasos", "replanes", "tokens",
        "latencia_seg",
    ):
        assert campo in puntaje
    assert puntaje["pasos"] == 1
    assert puntaje["replanes"] == 2
    assert puntaje["tokens"] == 120
    assert puntaje["latencia_seg"] == 1.5
    assert json.loads(json.dumps(puntaje)) == puntaje


def test_puntaje_acepta_trazas_releidas_desde_json():
    """La traza puede llegar como dict (artefacto releído), no solo dataclass."""
    traza = {
        "pregunta": "?",
        "respuesta": "El caso CASO-0004 está abierto.",
        "abstencion": False,
        "pasos": [{"herramienta": "consultar_caso",
                   "argumentos": {"id_caso": "CASO-0004"}, "ok": True}],
        "replanes": 0,
        "tokens_total": 10,
        "latencia_seg": 0.5,
        "plan": None,
        "error": None,
    }
    puntaje = puntuar_tarea(
        tarea(tipo="memoria", herramienta="consultar_caso",
              argumentos={"id_caso": "CASO-0004"}, turnos_previos=TURNOS_CASO,
              pregunta="¿Y en qué estado está?"),
        traza,
    )
    assert puntaje["herramienta_ok"] is True
    assert puntaje["argumentos_ok"] is True
    assert puntaje["memoria_ok"] is True


# --------------------------------------------------------------------------
# Agregación
# --------------------------------------------------------------------------
def puntaje(tipo="seleccion_herramienta", herramienta_ok=True,
            argumentos_ok=True, abstencion_esperada=False,
            abstencion_obtenida=False, memoria_ok=True,
            escritura_intentada=False, confirmacion_respetada=None,
            plan_ok: bool | None = True, replanes=0, tokens=100,
            latencia_seg=1.0, motivos=None):
    return {
        "tipo": tipo,
        "herramienta_ok": herramienta_ok,
        "argumentos_ok": argumentos_ok,
        "abstencion_esperada": abstencion_esperada,
        "abstencion_obtenida": abstencion_obtenida,
        "abstencion_ok": abstencion_esperada == abstencion_obtenida,
        "memoria_ok": memoria_ok,
        "escritura_intentada": escritura_intentada,
        "confirmacion_respetada": confirmacion_respetada,
        "plan_aplica": True,
        "plan_ok": plan_ok,
        "replanes": replanes,
        "tokens": tokens,
        "latencia_seg": latencia_seg,
        "motivos": list(motivos or []),
    }


def resumen(corridas, tarifas=None):
    return {
        "fecha": "2026-01-01 00:00:00",
        "modelo_agente": "modelo-falso",
        "dataset": "evaluation/dataset_agente.json",
        "verificacion_dataset": {"tareas": 14, "errores": []},
        "limite": None,
        "tarifas": tarifas or evaluar_agente._tarifas_actuales(),
        "corridas": corridas,
    }


def test_agregar_metricas_con_dos_puntajes_conocidos():
    puntajes = [
        puntaje(escritura_intentada=True, confirmacion_respetada=True,
                tokens=100, latencia_seg=1.0, replanes=0),
        puntaje(tipo="abstencion", herramienta_ok=False, argumentos_ok=False,
                escritura_intentada=True, confirmacion_respetada=False,
                plan_ok=False,
                abstencion_esperada=True, abstencion_obtenida=True,
                tokens=200, latencia_seg=2.0, replanes=2,
                motivos=["herramienta", "argumentos"]),
    ]
    metricas = agregar_metricas(puntajes)
    global_ = metricas["global"]

    assert global_["n_tareas"] == 2
    assert global_["tasa_seleccion"] == 0.5
    assert global_["tasa_argumentos"] == 0.5
    # Ambos acertaron en abstención: uno por no abstenerse, otro por hacerlo.
    assert global_["tasa_abstencion"] == 1.0
    assert (global_["abstencion_tp"], global_["abstencion_fp"],
            global_["abstencion_fn"]) == (1, 0, 0)
    assert global_["abstencion_precision"] == 1.0
    assert global_["abstencion_recall"] == 1.0
    assert global_["tasa_confirmacion"] == 0.5
    assert global_["tareas_con_escritura"] == 2
    assert global_["tasa_plan_valido"] == 0.5
    assert global_["tasa_memoria"] is None          # ninguna tarea de memoria
    assert global_["promedio_replanes"] == 1.0
    assert global_["tokens_totales"] == 300
    assert global_["latencia_total_seg"] == 3.0
    # Tarifa de entrada por defecto sobre los tokens totales: no es 0.
    assert global_["costo_configurado"] is True
    assert global_["costo_estimado_usd"] == round(300 / 1_000_000 * 0.145, 6)
    assert global_["costo_estimado_usd"] > 0
    assert metricas["tarifas"]["in_usd_por_millon"] == 0.145
    # Cada tipo tiene su propio grupo, no el promedio global.
    por_tipo = metricas["por_tipo"]
    assert por_tipo["seleccion_herramienta"]["tasa_seleccion"] == 1.0
    assert por_tipo["abstencion"]["tasa_seleccion"] == 0.0
    assert por_tipo["abstencion"]["abstencion_recall"] == 1.0


def test_agregar_metricas_separa_por_tipo_y_mide_memoria():
    puntajes = [
        puntaje(tipo="memoria", memoria_ok=True, tokens=10, latencia_seg=0.1),
        puntaje(tipo="memoria", memoria_ok=False, tokens=10, latencia_seg=0.1),
        # Se abstuvo y debía: TP.
        puntaje(tipo="abstencion", abstencion_esperada=True,
                abstencion_obtenida=True, tokens=10, latencia_seg=0.1),
        # Se abstuvo y no debía: FP (falsa abstención).
        puntaje(tipo="abstencion", abstencion_esperada=False,
                abstencion_obtenida=True, tokens=10, latencia_seg=0.1),
        # No se abstuvo y debía: FN (abstención omitida).
        puntaje(tipo="abstencion", abstencion_esperada=True,
                abstencion_obtenida=False, tokens=10, latencia_seg=0.1),
    ]
    metricas = agregar_metricas(puntajes)

    assert set(metricas["por_tipo"]) == {"memoria", "abstencion"}
    assert metricas["por_tipo"]["memoria"]["tasa_memoria"] == 0.5
    assert metricas["por_tipo"]["abstencion"]["tasa_memoria"] is None
    assert metricas["global"]["tasa_memoria"] == 0.5
    assert metricas["global"]["abstencion_tp"] == 1
    assert metricas["global"]["abstencion_fp"] == 1
    assert metricas["global"]["abstencion_fn"] == 1
    assert metricas["global"]["abstencion_precision"] == 0.5
    assert metricas["global"]["abstencion_recall"] == 0.5


def test_agregar_metricas_usa_la_tarifa_configurada(monkeypatch):
    monkeypatch.setattr(evaluar_agente, "TARIFA_IN_USD_POR_MILLON", 3.0)
    monkeypatch.setattr(evaluar_agente, "TARIFA_OUT_USD_POR_MILLON", 6.0)
    metricas = agregar_metricas([puntaje(tokens=1_000_000)])
    assert metricas["global"]["costo_estimado_usd"] == 3.0
    assert metricas["tarifas"]["in_usd_por_millon"] == 3.0
    assert metricas["tarifas"]["out_usd_por_millon"] == 6.0
    # La tarifa de salida se registra pero no se aplica: no hay desglose.
    assert metricas["tarifas"]["aplicada"] == "in_usd_por_millon"
    assert metricas["tarifas"]["desglose_entrada_salida"] is False


def test_agregar_metricas_sin_puntajes_no_divide_por_cero():
    metricas = agregar_metricas([])
    assert metricas["global"]["n_tareas"] == 0
    assert metricas["global"]["tasa_seleccion"] is None
    assert metricas["global"]["tokens_totales"] == 0
    assert metricas["por_tipo"] == {}


def test_tasa_de_confirmacion_solo_promedia_tareas_con_intento():
    """Sin escritura intentada la tasa es n/d, no 0 %."""
    puntajes = [
        puntaje(escritura_intentada=True, confirmacion_respetada=True),
        puntaje(escritura_intentada=False, confirmacion_respetada=None),
    ]
    metricas = agregar_metricas(puntajes)
    assert metricas["global"]["tasa_confirmacion"] == 1.0
    assert metricas["global"]["tareas_con_escritura"] == 1


def test_tasa_de_confirmacion_es_nd_sin_ninguna_escritura_intentada():
    metricas = agregar_metricas([
        puntaje(escritura_intentada=False),
        puntaje(tipo="abstencion", escritura_intentada=False),
    ])
    global_ = metricas["global"]
    assert global_["tasa_confirmacion"] is None
    assert global_["tareas_con_escritura"] == 0


def test_tasa_de_confirmacion_ignora_las_no_verificadas():
    """Un intento sin verificación no se promedia como si fuera éxito."""
    metricas = agregar_metricas([
        puntaje(escritura_intentada=True, confirmacion_respetada=True),
        puntaje(escritura_intentada=True, confirmacion_respetada=None),
        puntaje(escritura_intentada=True, confirmacion_respetada=False),
    ])
    assert metricas["global"]["tasa_confirmacion"] == 0.5


def test_plan_no_entra_en_el_promedio_ni_en_la_tasa_en_reactivo():
    puntajes = [
        puntuar_tarea(
            tarea(tipo="planificacion", min_pasos=2),
            respuesta(paso(1, "consultar_caso", {"id_caso": "CASO-0004"})),
            modo="reactivo",
        ),
        puntuar_tarea(
            tarea(tipo="seleccion_herramienta", herramienta="listar_casos"),
            respuesta(paso(1, "listar_casos", {"estado": "abierto"})),
            modo="reactivo",
        ),
    ]
    metricas = agregar_metricas(puntajes, modo="reactivo")
    global_ = metricas["global"]
    assert global_["plan_aplica"] is False
    assert global_["tasa_plan_valido"] is None
    assert metricas["modo"] == "reactivo"
    # Ninguna tarea arrastra un fallo por "plan" en reactivo.
    assert all("plan" not in p["motivos"] for p in puntajes)


def test_plan_en_planificado_solo_promedia_los_aplicables():
    metricas = agregar_metricas(
        [puntaje(plan_ok=True), puntaje(plan_ok=None)], modo="planificado"
    )
    assert metricas["global"]["plan_aplica"] is True
    assert metricas["global"]["tasa_plan_valido"] == 1.0


# --------------------------------------------------------------------------
# Costo estimado
# --------------------------------------------------------------------------
def test_tarifas_por_defecto_calculan_costo_via_cli():
    args = evaluar_agente._parser().parse_args([])
    assert args.tarifa_in_usd_por_millon == 0.145
    assert args.tarifa_out_usd_por_millon == 0.580
    assert args.sin_tarifa is False
    assert evaluar_agente._tarifas_desde_args(args) == (0.145, 0.580)

    metricas = agregar_metricas([puntaje(tokens=1_000_000)])
    assert metricas["global"]["costo_configurado"] is True
    # Solo hay tokens totales: la tarifa de entrada se aplica al total.
    assert metricas["global"]["costo_estimado_usd"] == 0.145
    assert metricas["tarifas"]["base"] == "tokens_total"


def test_sin_tarifa_anula_ambas_y_gana_sobre_las_explicitas():
    args = evaluar_agente._parser().parse_args([
        "--tarifa-in-usd-por-millon", "3.0",
        "--tarifa-out-usd-por-millon", "6.0",
        "--sin-tarifa",
    ])
    assert evaluar_agente._tarifas_desde_args(args) == (0.0, 0.0)


def test_sin_tarifa_el_texto_dice_costo_no_configurado(monkeypatch, tmp_path):
    monkeypatch.setattr(evaluar_agente, "TARIFA_IN_USD_POR_MILLON", 0.0)
    monkeypatch.setattr(evaluar_agente, "TARIFA_OUT_USD_POR_MILLON", 0.0)

    metricas = agregar_metricas([puntaje(tokens=1_000_000)])
    global_ = metricas["global"]
    assert metricas["tarifas"]["configurada"] is False
    assert global_["costo_configurado"] is False
    assert global_["costo_estimado_usd"] is None
    assert global_["costo_texto"] == "costo no configurado"

    corrida = {"modo": "reactivo", "puntajes": [puntaje()], "metricas": metricas}
    ruta_json, ruta_md = evaluar_agente._escribir(
        resumen([corrida]), tmp_path / "agente.json"
    )
    datos = json.loads(ruta_json.read_text(encoding="utf-8"))
    assert datos["corridas"][0]["metricas"]["global"]["costo_texto"] == (
        "costo no configurado"
    )
    markdown = ruta_md.read_text(encoding="utf-8")
    assert "costo no configurado" in markdown.lower()
    assert "US$ 0.0" not in markdown


def test_costo_configurado_declara_la_base_sobre_tokens_totales(tmp_path):
    metricas = agregar_metricas([puntaje(tokens=2_000_000)])
    corrida = {"modo": "reactivo", "puntajes": [puntaje()], "metricas": metricas}
    ruta_json, ruta_md = evaluar_agente._escribir(
        resumen([corrida]), tmp_path / "agente.json"
    )

    datos = json.loads(ruta_json.read_text(encoding="utf-8"))
    assert datos["tarifas"]["in_usd_por_millon"] == 0.145
    assert datos["tarifas"]["out_usd_por_millon"] == 0.580
    assert datos["tarifas"]["desglose_entrada_salida"] is False
    assert "tokens totales" in datos["tarifas"]["nota"]
    assert datos["corridas"][0]["metricas"]["global"]["costo_estimado_usd"] == 0.29

    markdown = ruta_md.read_text(encoding="utf-8")
    assert "0.145" in markdown and "0.58" in markdown
    assert "tokens totales" in markdown
    assert "no una facturación" in markdown


# --------------------------------------------------------------------------
# CLI offline
# --------------------------------------------------------------------------
def test_verificar_dataset_lista_las_14_tareas_sin_red(capsys):
    codigo = main(["--verificar-dataset"])
    salida = capsys.readouterr().out

    assert codigo == 0
    assert "Tareas: 14" in salida
    for numero in range(1, 15):
        assert f"[T{numero:02d}]" in salida


def test_verificar_dataset_no_construye_el_agente(monkeypatch):
    """Guardia explícita: el camino offline no debe instanciar nada con red."""
    import src.agente
    import src.rag

    def prohibido(*args, **kwargs):
        raise AssertionError("--verificar-dataset no debe construir agentes")

    monkeypatch.setattr(src.rag, "RAGEngine", prohibido)
    monkeypatch.setattr(src.agente, "AgenteTecnoApoyo", prohibido)
    assert main(["--verificar-dataset"]) == 0


def test_verificar_dataset_reporta_errores(tmp_path, monkeypatch, capsys):
    invalido = tmp_path / "invalido.json"
    invalido.write_text(
        json.dumps({"tareas": [{"id": "T01", "tipo": "otro"}]}), encoding="utf-8"
    )
    monkeypatch.setattr(evaluar_agente, "RUTA_DATASET", invalido)

    assert main(["--verificar-dataset"]) == 1
    assert "inválido" in capsys.readouterr().out


# --------------------------------------------------------------------------
# Artefactos
# --------------------------------------------------------------------------
def test_reservar_par_nunca_sobrescribe(tmp_path):
    primero_json = tmp_path / "agente_20260101_120000.json"
    json_1, md_1 = evaluar_agente._reservar_par(primero_json)
    assert json_1 == primero_json
    assert md_1 == tmp_path / "agente_20260101_120000.md"

    json_1.write_text("{}", encoding="utf-8")
    md_1.write_text("# previo", encoding="utf-8")

    json_2, md_2 = evaluar_agente._reservar_par(primero_json)
    assert json_2 == tmp_path / "agente_20260101_120000_2.json"
    assert md_2 == tmp_path / "agente_20260101_120000_2.md"
    assert json_1.read_text(encoding="utf-8") == "{}"
    assert md_1.read_text(encoding="utf-8") == "# previo"


def test_escribir_genera_json_y_markdown(tmp_path):
    datos_puntajes = [puntaje(),
                      puntaje(herramienta_ok=False, motivos=["herramienta"])]
    resumen_ = resumen([
        {
            "modo": "reactivo",
            "puntajes": datos_puntajes,
            "metricas": agregar_metricas(datos_puntajes, modo="reactivo"),
        }
    ])
    ruta_json, ruta_md = evaluar_agente._escribir(
        resumen_, tmp_path / "agente.json"
    )

    assert json.loads(ruta_json.read_text(encoding="utf-8"))["fecha"] == (
        resumen_["fecha"]
    )
    markdown = ruta_md.read_text(encoding="utf-8")
    assert "Modo reactivo" in markdown
    assert "Tasa de selección" in markdown
    assert "Tareas con fallos (1)" in markdown
    # En reactivo el plan no aplica: se muestra n/a, no 0 %.
    assert "| Tasa de plan válido | n/a |" in markdown
    assert "no configurado" not in markdown.lower()
