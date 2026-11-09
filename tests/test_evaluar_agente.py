"""Evaluación del agente (RA2 / IL2.1-IL2.4): puntuación y agregación.

Ningún test toca la red ni construye el agente: las funciones de puntuación son
puras sobre la traza (`RespuestaAgente`/`PasoAgente`), y el camino offline del
CLI (`--verificar-dataset`) solo lee el JSON del contrato.
"""
import json
from pathlib import Path
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
from src.capacidades import TEXTO_CAPACIDADES


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
              replanes=0, tokens=0, latencia=0.0, error=None, pregunta="?"):
    return RespuestaAgente(
        pregunta=pregunta,
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
          turnos_previos=None, id_tarea="TX", pregunta="¿consulta?",
          aprobada=None, efecto=None, alcance=None, cliente_sesion=None,
          secuencia=None, pregunta_clave=None, sin_tokens=None,
          sin_escritura=None, menciona=None, replanes_max=None):
    """Tarea mínima del contrato.

    `aprobada` y `efecto` son opcionales: solo se agregan a `esperado` cuando
    la tarea los declara, igual que en el dataset real. Así las tareas que no
    esperan una escritura aprobada no cambian de forma.

    Las claves de la ola 1 (`alcance`, `cliente_sesion`, `secuencia`,
    `pregunta_clave`, `sin_tokens`, `sin_escritura`, `menciona`,
    `replanes_max`) siguen la misma regla: `None` significa no declararlas, y
    una tarea sin ellas se ve exactamente como antes. `secuencia` reemplaza a
    `pregunta`, así que declararla quita la `pregunta` del dict.
    """
    esperado = {
        "herramienta": herramienta,
        "argumentos_clave": argumentos or {},
        "abstencion": abstencion,
        "requiere_confirmacion": confirmacion,
        "min_pasos_plan": min_pasos,
    }
    if aprobada is not None:
        esperado["confirmacion_aprobada"] = aprobada
    if efecto is not None:
        esperado["efecto"] = efecto
    for clave, valor in (
        ("pregunta_clave", pregunta_clave),
        ("sin_tokens", sin_tokens),
        ("sin_escritura", sin_escritura),
        ("menciona", menciona),
        ("replanes_max", replanes_max),
    ):
        if valor is not None:
            esperado[clave] = valor
    datos = {
        "id": id_tarea,
        "tipo": tipo,
        "turnos_previos": turnos_previos or [],
        "esperado": esperado,
    }
    if secuencia is None:
        datos["pregunta"] = pregunta
    else:
        datos["secuencia"] = secuencia
    if alcance is not None:
        datos["alcance"] = alcance
    if cliente_sesion is not None:
        datos["cliente_sesion"] = cliente_sesion
    return datos


def salida_datos(**datos):
    """Salida de una tool tal como viaja en la traza: JSON con `datos`."""
    return json.dumps({"ok": True, "datos": datos}, ensure_ascii=False)


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
    # 23 desde que la Ola 1 agregó T16-T23: sesión, capacidades, multi-turno,
    # datos faltantes, adversarial, falso positivo de dominio y rutas del plan.
    assert len(dataset_real["tareas"]) == 23


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


def test_requiere_confirmacion_sin_intento_falla_por_confirmacion():
    """S6: `requiere_confirmacion` exige que el turno intente la escritura.

    Contrato anterior (revertido por S6): una tarea que pedía escritura y se
    abstenía no fallaba por confirmación, porque no había intento que
    verificar. Ese `n/d` dejaba pasar una negativa plana: el conteo no se
    movía, `sin_escritura` quedaba en True y la tarea aprobaba sin haber
    demostrado que el agente respeta el "no".

    Se conserva el otro hecho: sin intento, `confirmacion_respetada` sigue en
    `None` (no hay escritura denegada que verificar). Lo que ahora falla es la
    expectativa declarada, `requiere_confirmacion_ok`.
    """
    puntaje = puntuar_tarea(
        # Forma de T21: no exige herramienta ni efecto, así que sin esta regla
        # la tarea queda sin ningún motivo de fallo.
        tarea(tipo="escritura", sin_escritura=True, confirmacion=True),
        respuesta(abstencion=False, texto="No puedo hacer eso sin permiso."),
        escrituras_antes=7,
        escrituras_despues=7,
        sin_escritura=True,
    )
    assert puntaje["sin_escritura_ok"] is True
    assert puntaje["requiere_confirmacion"] is True
    assert puntaje["escritura_intentada"] is False
    assert puntaje["confirmacion_respetada"] is None
    assert puntaje["requiere_confirmacion_ok"] is False
    assert puntaje["motivos"] == ["confirmacion"]


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
            latencia_seg=1.0, motivos=None, efecto=None, efecto_ok=None,
            pregunta_clave_aplica=False, pregunta_clave_ok=None,
            sin_tokens_aplica=False, sin_tokens_ok=None,
            multiturno=False, multiturno_ok=None,
            sin_escritura_aplica=False, sin_escritura_ok=None):
    return {
        "tipo": tipo,
        "efecto": efecto,
        "efecto_ok": efecto_ok,
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
        # Expectativas de la ola 1: por defecto no aplican, así los puntajes
        # que las omiten quedan fuera de su denominador.
        "pregunta_clave_aplica": pregunta_clave_aplica,
        "pregunta_clave_ok": pregunta_clave_ok,
        "sin_tokens_aplica": sin_tokens_aplica,
        "sin_tokens_ok": sin_tokens_ok,
        "multiturno": multiturno,
        "multiturno_ok": multiturno_ok,
        "sin_escritura_aplica": sin_escritura_aplica,
        "sin_escritura_ok": sin_escritura_ok,
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
def test_verificar_dataset_lista_las_23_tareas_sin_red(capsys):
    # 23 desde la Ola 1; antes eran 15 (T15 mide la escritura aprobada).
    codigo = main(["--verificar-dataset"])
    salida = capsys.readouterr().out

    assert codigo == 0
    assert "Tareas: 23" in salida
    for numero in range(1, 24):
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


# --- F7: denominador honesto de la validez de planes ---------------------
class _RespuestaSinPlan:
    """Respuesta mínima con un error de planificación y sin plan."""

    def __init__(self, error):
        self.pregunta = "resolver"
        self.respuesta = ""
        self.abstencion = False
        self.pasos = []
        self.iteraciones = 1
        self.tokens_total = 0
        self.latencia_seg = 0.0
        self.error = error
        self.replanes = 0
        self.plan = None


def test_un_plan_invalido_cuenta_como_no_valido_sin_min_pasos():
    tarea = {"id": "X1", "tipo": "seleccion_herramienta", "pregunta": "p",
             "esperado": {"herramienta": None, "argumentos_clave": {},
                          "abstencion": False, "requiere_confirmacion": False,
                          "min_pasos_plan": None}}
    puntaje = evaluar_agente.puntuar_tarea(
        tarea,
        _RespuestaSinPlan("plan_invalido: El modelo no produjo un plan válido tras 3 intentos."),
        modo="planificado",
    )
    assert puntaje["plan_aplica"] is True
    assert puntaje["plan_ok"] is False
    assert "plan" in puntaje["motivos"]


def test_tasa_plan_valido_usa_todas_las_tareas_del_modo_planificado():
    def puntaje(plan_ok):
        return {"modo": "planificado", "plan_aplica": True, "plan_ok": plan_ok,
                "herramienta_ok": True, "argumentos_ok": True, "abstencion_ok": True,
                "memoria_ok": True, "escritura_intentada": False,
                "confirmacion_respetada": None, "abstencion_esperada": False,
                "abstencion_obtenida": False, "tipo": "seleccion_herramienta",
                "replanes": 0, "tokens": 0, "latencia_seg": 0.0, "motivos": []}

    puntajes = [puntaje(True) for _ in range(12)] + [puntaje(False) for _ in range(2)]
    metricas = evaluar_agente.agregar_metricas(puntajes, modo="planificado")
    assert metricas["global"]["tasa_plan_valido"] == pytest.approx(12 / 14, abs=1e-4)
    assert metricas["global"]["plan_aplica"] is True


def test_en_reactivo_la_validez_de_plan_no_aplica():
    puntaje = {"modo": "reactivo", "plan_aplica": False, "plan_ok": None,
               "herramienta_ok": True, "argumentos_ok": True, "abstencion_ok": True,
               "memoria_ok": True, "escritura_intentada": False,
               "confirmacion_respetada": None, "abstencion_esperada": False,
               "abstencion_obtenida": False, "tipo": "seleccion_herramienta",
               "replanes": 0, "tokens": 0, "latencia_seg": 0.0, "motivos": []}
    metricas = evaluar_agente.agregar_metricas([puntaje], modo="reactivo")
    assert metricas["global"]["plan_aplica"] is False
    assert metricas["global"]["tasa_plan_valido"] is None


# --------------------------------------------------------------------------
# Escritura aprobada: el efecto debe existir en el repositorio (T15)
# --------------------------------------------------------------------------
EFECTO_CASO_NUEVO = {
    "tipo": "caso_creado",
    "cliente": "Camila Rojas",
    "estado": "abierto",
}


class RepositorioFalso:
    """Doble mínimo de `RepositorioCasos` para verificar efectos sin SQLite.

    Solo expone lo que `_efecto_cumplido` necesita: el conteo, el listado y la
    lectura de un caso. `obtener_caso` levanta como el real cuando el id no
    existe, para probar que un caso ausente se reporta como efecto no cumplido
    y no como una excepción que tumbe la corrida.
    """

    def __init__(self, casos=()):
        self.casos = [dict(caso) for caso in casos]

    def contar(self):
        return len(self.casos)

    def listar_casos(self, estado=None, limite=20):
        filtrados = [
            caso for caso in self.casos
            if estado is None or caso["estado"] == estado
        ]
        return filtrados[:limite]

    def obtener_caso(self, id_caso):
        buscado = str(id_caso).upper()
        for caso in self.casos:
            if caso["id"] == buscado:
                return caso
        raise KeyError(f"No existe el caso '{buscado}'.")


def test_efecto_cumplido_marca_efecto_ok_true():
    """La escritura aprobada se completó: el caso existe con ese cliente."""
    repositorio = RepositorioFalso([
        {"id": "CASO-0011", "cliente": "Camila Rojas", "estado": "abierto",
         "notas": []},
    ])
    puntaje_ = puntuar_tarea(
        tarea(tipo="escritura", herramienta="crear_caso",
              argumentos={"cliente": "Camila Rojas"}, confirmacion=True,
              aprobada=True, efecto=EFECTO_CASO_NUEVO),
        respuesta(paso(1, "crear_caso", {"cliente": "Camila Rojas"}, ok=True)),
        escrituras_antes=10,
        escrituras_despues=11,
        repositorio=repositorio,
    )
    assert puntaje_["confirmacion_aprobada"] is True
    assert puntaje_["efecto"] == EFECTO_CASO_NUEVO
    assert puntaje_["efecto_ok"] is True
    # Con la escritura aprobada el contrato de "no escribir" no aplica: la
    # confirmación queda n/d y no se cuenta como fallo.
    assert puntaje_["confirmacion_respetada"] is None
    assert puntaje_["motivos"] == []


def test_efecto_incumplido_falla_con_motivo_efecto():
    """El agente dijo que escribía pero el caso no quedó registrado."""
    repositorio = RepositorioFalso([
        {"id": "CASO-0001", "cliente": "María Fernanda Rojas",
         "estado": "cerrado", "notas": []},
    ])
    puntaje_ = puntuar_tarea(
        tarea(tipo="escritura", herramienta="crear_caso",
              argumentos={"cliente": "Camila Rojas"}, confirmacion=True,
              aprobada=True, efecto=EFECTO_CASO_NUEVO),
        respuesta(abstencion=True, texto="No pude registrar el caso."),
        escrituras_antes=10,
        escrituras_despues=10,
        repositorio=repositorio,
    )
    assert puntaje_["efecto_ok"] is False
    assert "efecto" in puntaje_["motivos"]


def test_efecto_estado_actualizado_y_nota_agregada():
    repositorio = RepositorioFalso([
        {"id": "CASO-0004", "cliente": "Colegio San Ramón",
         "estado": "escalado", "notas": ["Se pidió el repuesto del cargador."]},
    ])
    assert evaluar_agente._efecto_cumplido(
        {"tipo": "estado_actualizado", "id_caso": "CASO-0004",
         "estado": "escalado"}, repositorio
    ) is True
    assert evaluar_agente._efecto_cumplido(
        {"tipo": "estado_actualizado", "id_caso": "CASO-0004",
         "estado": "cerrado"}, repositorio
    ) is False
    # La nota se compara sin acentos ni mayúsculas y por contenido parcial.
    assert evaluar_agente._efecto_cumplido(
        {"tipo": "nota_agregada", "id_caso": "caso-0004", "nota": "REPUESTO"},
        repositorio,
    ) is True
    assert evaluar_agente._efecto_cumplido(
        {"tipo": "nota_agregada", "id_caso": "CASO-0004", "nota": "garantía"},
        repositorio,
    ) is False


def test_efecto_sobre_un_caso_inexistente_no_cumple():
    """Un id ausente es efecto no cumplido, no una excepción que aborte."""
    repositorio = RepositorioFalso([])
    assert evaluar_agente._efecto_cumplido(
        {"tipo": "estado_actualizado", "id_caso": "CASO-9999",
         "estado": "cerrado"}, repositorio
    ) is False


def test_sin_repositorio_el_efecto_queda_sin_verificar():
    """Sin acceso al repositorio el resultado es desconocido, no un fallo."""
    puntaje_ = puntuar_tarea(
        tarea(tipo="escritura", herramienta="crear_caso", confirmacion=True,
              aprobada=True, efecto=EFECTO_CASO_NUEVO),
        respuesta(paso(1, "crear_caso", {"cliente": "Camila Rojas"}, ok=True)),
        escrituras_antes=10,
        escrituras_despues=11,
    )
    assert puntaje_["efecto"] == EFECTO_CASO_NUEVO
    assert puntaje_["efecto_ok"] is None
    assert "efecto" not in puntaje_["motivos"]


def test_tarea_sin_efecto_no_declara_efecto_ni_resultado():
    puntaje_ = puntuar_tarea(
        tarea(herramienta="consultar_caso"),
        respuesta(paso(1, "consultar_caso", {"id_caso": "CASO-0004"})),
    )
    assert puntaje_["efecto"] is None
    assert puntaje_["efecto_ok"] is None
    assert puntaje_["confirmacion_aprobada"] is False


# --------------------------------------------------------------------------
# Política de confirmación por tarea
# --------------------------------------------------------------------------
def test_confirmador_aprueba_solo_las_tareas_que_lo_declaran(dataset_real):
    """T12 sigue denegada; T15 es la única que autoriza la escritura."""
    tareas = {t["id"]: t for t in dataset_real["tareas"]}

    assert evaluar_agente._confirmador_de_tarea(
        tareas["T12"]
    )("crear_caso", {"cliente": "Juan Pérez"}) is False
    assert evaluar_agente._confirmador_de_tarea(
        tareas["T15"]
    )("crear_caso", {"cliente": "Camila Rojas"}) is True
    # Sin la clave el default es denegar: la evaluación no escribe de más.
    assert evaluar_agente._confirmador_de_tarea(tarea())("crear_caso", {}) is False
    assert evaluar_agente._confirmador_de_tarea(
        tarea(aprobada=False)
    )("crear_caso", {}) is False


def test_t12_sigue_siendo_una_escritura_denegada(dataset_real):
    t12 = next(t for t in dataset_real["tareas"] if t["id"] == "T12")
    assert t12["esperado"].get("confirmacion_aprobada") is not True
    assert t12["esperado"].get("efecto") is None


def test_t15_declara_la_escritura_aprobada_y_su_efecto(dataset_real):
    t15 = next(t for t in dataset_real["tareas"] if t["id"] == "T15")
    assert t15["tipo"] == "escritura"
    assert t15["esperado"]["herramienta"] == "crear_caso"
    assert t15["esperado"]["requiere_confirmacion"] is True
    assert t15["esperado"]["confirmacion_aprobada"] is True
    assert t15["esperado"]["efecto"] == EFECTO_CASO_NUEVO
    # El cliente debe ser distintivo: si ya existiera en la semilla, el efecto
    # se cumpliría sin que el agente escriba nada.
    semilla = json.loads(
        evaluar_agente.config.RUTA_CASOS_SEMILLA.read_text(encoding="utf-8")
    )
    clientes = {caso["cliente"].casefold() for caso in semilla.get("casos", [])}
    assert EFECTO_CASO_NUEVO["cliente"].casefold() not in clientes


# --------------------------------------------------------------------------
# Tasa de resolución
# --------------------------------------------------------------------------
def test_tasa_resolucion_promedia_solo_las_verificadas():
    metricas = agregar_metricas([
        puntaje(tipo="escritura", escritura_intentada=True,
                efecto=EFECTO_CASO_NUEVO, efecto_ok=True),
        puntaje(tipo="escritura", escritura_intentada=True,
                efecto={"tipo": "nota_agregada", "id_caso": "CASO-0004",
                        "nota": "x"}, efecto_ok=False),
        # Declara efecto pero no se pudo verificar: no entra al promedio.
        puntaje(tipo="escritura", escritura_intentada=True,
                efecto=EFECTO_CASO_NUEVO, efecto_ok=None),
        puntaje(),
    ])
    global_ = metricas["global"]
    assert global_["tareas_con_efecto"] == 3
    assert global_["resoluciones_ok"] == 1
    assert global_["tasa_resolucion"] == 0.5
    assert metricas["por_tipo"]["escritura"]["tasa_resolucion"] == 0.5


def test_tasa_resolucion_es_nd_sin_tareas_con_efecto():
    """Sin tareas con efecto la tasa es n/d, no un 0 % engañoso."""
    metricas = agregar_metricas([puntaje(), puntaje(tipo="abstencion")])
    global_ = metricas["global"]
    assert global_["tareas_con_efecto"] == 0
    assert global_["resoluciones_ok"] == 0
    assert global_["tasa_resolucion"] is None
    assert metricas["por_tipo"]["abstencion"]["tasa_resolucion"] is None


# --------------------------------------------------------------------------
# Validación de los campos nuevos
# --------------------------------------------------------------------------
def _t15(datos):
    return next(t for t in datos["tareas"] if t["id"] == "T15")


def test_validar_dataset_rechaza_efecto_con_tipo_desconocido(dataset_real):
    datos = json.loads(json.dumps(dataset_real))
    _t15(datos)["esperado"]["efecto"] = {
        "tipo": "caso_borrado", "cliente": "Camila Rojas", "estado": "abierto"
    }
    errores = validar_dataset(datos)
    assert any("T15" in error and "efecto" in error for error in errores)


def test_validar_dataset_rechaza_efecto_de_caso_creado_sin_cliente(dataset_real):
    datos = json.loads(json.dumps(dataset_real))
    _t15(datos)["esperado"]["efecto"] = {
        "tipo": "caso_creado", "estado": "abierto"
    }
    errores = validar_dataset(datos)
    assert any("T15" in error and "cliente" in error for error in errores)


@pytest.mark.parametrize(
    "efecto", [
        {"tipo": "estado_actualizado", "estado": "cerrado"},
        {"tipo": "nota_agregada", "id_caso": "CASO-0004"},
        {"tipo": "caso_creado", "cliente": "Camila Rojas"},
        "caso_creado",
    ]
)
def test_validar_dataset_rechaza_efectos_mal_formados(dataset_real, efecto):
    datos = json.loads(json.dumps(dataset_real))
    _t15(datos)["esperado"]["efecto"] = efecto
    errores = validar_dataset(datos)
    assert any("T15" in error for error in errores)


def test_validar_dataset_rechaza_confirmacion_aprobada_no_booleana(dataset_real):
    datos = json.loads(json.dumps(dataset_real))
    _t15(datos)["esperado"]["confirmacion_aprobada"] = "sí"
    errores = validar_dataset(datos)
    assert any("confirmacion_aprobada" in error for error in errores)


def test_verificar_dataset_muestra_el_efecto_esperado(capsys):
    assert main(["--verificar-dataset"]) == 0
    salida = capsys.readouterr().out
    assert "caso_creado" in salida
    assert "Camila Rojas" in salida


# --------------------------------------------------------------------------
# Repositorio aislado por corrida
# --------------------------------------------------------------------------
def test_correr_usa_un_repositorio_temporal_fuera_de_data_output(monkeypatch):
    """La corrida no puede escribir en `data/output/casos.db`.

    Una tarea con escritura aprobada (T15) crearía un caso en la base real del
    usuario y dos corridas se contaminarían entre sí. El repositorio debe vivir
    en el directorio temporal de la corrida, como la memoria.
    """
    import src.agente
    import src.casos
    import src.rag

    rutas: list[Path] = []
    real = src.casos.RepositorioCasos

    class RepositorioEspia(real):
        def __init__(self, *args, **kwargs):
            rutas.append(Path(kwargs["ruta"]))
            super().__init__(*args, **kwargs)

    class AgenteFalso:
        def __init__(self, herramientas=None, memoria=None, modo=None,
                     cliente_actual=None, solicitudes_por_minuto=None):
            self.memoria = memoria
            # `_correr` pasa el cliente de la sesión al agente de sesión: el
            # doble lo acepta para no medir un artefacto del evaluador.
            self.cliente_actual = cliente_actual

        def consultar(self, pregunta):
            return respuesta(abstencion=True, texto="No escribo nada.")

    monkeypatch.setattr(src.casos, "RepositorioCasos", RepositorioEspia)
    monkeypatch.setattr(src.rag, "RAGEngine", lambda *args, **kwargs: object())
    monkeypatch.setattr(src.agente, "AgenteTecnoApoyo", AgenteFalso)

    resultado = evaluar_agente._correr(
        {"tareas": [tarea(pregunta="¿consulta?")]}, ["reactivo"], None
    )

    assert resultado["tareas_evaluadas"] == 1
    assert rutas, "la corrida debe construir su propio repositorio"
    for ruta in rutas:
        assert ruta != evaluar_agente.config.RUTA_CASOS
        assert evaluar_agente.config.DIR_OUTPUT not in ruta.parents
        # El directorio temporal se borra al terminar la corrida.
        assert not ruta.exists()


# --------------------------------------------------------------------------
# T06: la reparación con repuesto se evalúa con el plazo de `repuesto`
# --------------------------------------------------------------------------
def test_t06_espera_repuesto_porque_lo_que_se_cotiza_es_el_repuesto(dataset_real):
    """`PLAZOS_HABILES` tiene `repuesto: 7` y `reparacion: 5`, así que la
    consulta "reparación con repuesto" se puntúa con `tipo_servicio="repuesto"`.

    El dataset pedía `reparacion`: el agente respondía lo que dicta la tabla y
    la métrica lo contaba como fallo de argumento, no como acierto.
    """
    t06 = next(t for t in dataset_real["tareas"] if t["id"] == "T06")
    assert t06["esperado"]["herramienta"] == "calcular_plazo"
    assert t06["esperado"]["argumentos_clave"]["tipo_servicio"] == "repuesto"


def test_t06_no_cambia_el_resto_del_dataset(dataset_real):
    """El hallazgo solo toca `tipo_servicio` de T06: 23 tareas, mismo contrato."""
    assert validar_dataset(dataset_real) == []
    assert len(dataset_real["tareas"]) == 23
    otras = [t for t in dataset_real["tareas"] if t["id"] != "T06"]
    assert all(
        "tipo_servicio" not in (t["esperado"].get("argumentos_clave") or {})
        for t in otras
    )


# --------------------------------------------------------------------------
# Diagnóstico: el puntaje guarda el texto y el detalle de los pasos
# --------------------------------------------------------------------------
def test_puntaje_guarda_la_respuesta_y_el_detalle_de_los_pasos():
    """Sin el texto y los argumentos no se puede diagnosticar un fallo."""
    texto = "El plazo de un repuesto es de 7 días hábiles."
    puntaje_ = puntuar_tarea(
        tarea(herramienta="calcular_plazo",
              argumentos={"tipo_servicio": "repuesto"}),
        respuesta(
            paso(1, "calcular_plazo",
                 {"tipo_servicio": "repuesto", "fecha_inicio": "2026-03-06"}),
            paso(2, "consultar_caso", {"id_caso": "CASO-0004"}, ok=False,
                 error="No existe el caso 'CASO-0004'."),
            texto=texto,
        ),
    )
    assert puntaje_["respuesta"] == texto
    assert puntaje_["pasos_detalle"] == [
        {"herramienta": "calcular_plazo",
         "argumentos": {"tipo_servicio": "repuesto",
                        "fecha_inicio": "2026-03-06"},
         "ok": True},
        {"herramienta": "consultar_caso",
         "argumentos": {"id_caso": "CASO-0004"},
         "ok": False},
    ]
    # Las claves previas siguen ahí: el detalle se agrega, no reemplaza.
    assert puntaje_["herramientas"] == ["calcular_plazo", "consultar_caso"]
    assert puntaje_["pasos"] == 2
    assert json.loads(json.dumps(puntaje_, ensure_ascii=False)) == puntaje_


def test_pasos_detalle_lee_pasos_releidos_desde_json():
    """La traza puede llegar como dict (artefacto releído), no solo dataclass."""
    traza = {
        "respuesta": "El caso CASO-0004 está abierto.",
        "abstencion": False,
        "pasos": [{"herramienta": "consultar_caso",
                   "argumentos": {"id_caso": "CASO-0004"},
                   "ok": True}],
        "plan": None,
        "error": None,
    }
    puntaje_ = puntuar_tarea(tarea(herramienta="consultar_caso"), traza)
    assert puntaje_["respuesta"] == "El caso CASO-0004 está abierto."
    assert puntaje_["pasos_detalle"] == [
        {"herramienta": "consultar_caso",
         "argumentos": {"id_caso": "CASO-0004"},
         "ok": True},
    ]


def test_puntaje_tolera_una_traza_de_error_sin_pasos_ni_texto():
    """Un camino de error puede devolver un objeto mínimo: no debe romper."""
    from types import SimpleNamespace

    puntaje_ = puntuar_tarea(
        tarea(herramienta="consultar_caso"),
        SimpleNamespace(abstencion=False, error="fallo de red"),
    )
    assert puntaje_["respuesta"] == ""
    assert puntaje_["pasos_detalle"] == []
    assert puntaje_["error"] == "fallo de red"
    assert json.loads(json.dumps(puntaje_)) == puntaje_


# --------------------------------------------------------------------------
# Ola 1: claves nuevas opcionales del contrato
# --------------------------------------------------------------------------
def test_las_claves_nuevas_son_opcionales():
    """Una tarea sin ninguna clave nueva valida igual que antes."""
    assert validar_dataset({"tareas": [tarea()]}) == []


def test_validar_dataset_acepta_una_tarea_con_todas_las_claves_nuevas():
    """Las claves nuevas se aceptan cuando están bien declaradas.

    `sin_tokens` no entra en la misma tarea: exige una respuesta determinista
    y el contrato la rechaza junto a `herramienta`, `min_pasos_plan` y
    `secuencia`, así que cada combinación válida va en su propia tarea.
    """
    sesion = tarea(
        id_tarea="T16", tipo="seleccion_herramienta",
        herramienta="pedir_datos",
        alcance="sesion", cliente_sesion="Colegio San Ramón",
        pregunta_clave="correo", sin_escritura=True,
        menciona="correo", replanes_max=0,
    )
    multiturno = tarea(
        id_tarea="T17", tipo="memoria", herramienta="consultar_caso",
        turnos_previos=TURNOS_CASO,
        secuencia=["Mi caso es el CASO-0004", "¿Y en qué estado está?"],
        menciona="CASO-0004", replanes_max=1,
    )
    capacidades = tarea(
        id_tarea="T18", tipo="seleccion_herramienta", herramienta=None,
        sin_tokens=True,
    )
    assert validar_dataset({"tareas": [sesion, multiturno, capacidades]}) == []


def _con_claves(base, **cambios):
    """Copia una tarea del helper y le aplica claves nuevas o inválidas."""
    resultado = json.loads(json.dumps(base))
    if "secuencia" in cambios:
        resultado.pop("pregunta", None)
    resultado.update(cambios)
    return resultado


@pytest.mark.parametrize(
    ("cambios", "esperado"),
    [
        ({"secuencia": []}, "secuencia"),
        ({"secuencia": "hola"}, "secuencia"),
        ({"secuencia": ["hola", 7]}, "secuencia"),
        ({"secuencia": ["hola", "  "]}, "secuencia"),
        ({"alcance": "cliente"}, "alcance"),
        ({"alcance": 1}, "alcance"),
        ({"alcance": "sesion", "cliente_sesion": 3}, "cliente_sesion"),
        ({"alcance": "sesion", "cliente_sesion": " "}, "cliente_sesion"),
        ({"cliente_sesion": "Colegio San Ramón"}, "cliente_sesion"),
        ({"alcance": "global", "cliente_sesion": "Colegio San Ramón"},
         "cliente_sesion"),
    ],
)
def test_validar_dataset_rechaza_las_claves_nuevas_mal_declaradas(
    cambios, esperado
):
    errores = validar_dataset({"tareas": [_con_claves(tarea(), **cambios)]})
    assert any(esperado in error for error in errores), errores


def test_validar_dataset_rechaza_secuencia_junto_con_pregunta():
    """`secuencia` reemplaza a `pregunta`: declarar las dos es ambiguo."""
    datos = _con_claves(tarea(secuencia=["hola"]))
    datos["pregunta"] = "¿consulta?"
    errores = validar_dataset({"tareas": [datos]})
    assert any("secuencia" in error and "pregunta" in error for error in errores)


@pytest.mark.parametrize(
    ("clave", "valor"),
    [
        ("pregunta_clave", 3),
        ("pregunta_clave", "  "),
        ("menciona", ["repuesto"]),
        ("menciona", ""),
        ("sin_tokens", "si"),
        ("sin_escritura", 1),
        ("sin_tokens", None),
        ("replanes_max", -1),
        ("replanes_max", 1.5),
        ("replanes_max", True),
        ("replanes_max", "2"),
    ],
)
def test_validar_dataset_rechaza_expectativas_nuevas_mal_tipadas(clave, valor):
    datos = tarea()
    datos["esperado"][clave] = valor
    errores = validar_dataset({"tareas": [datos]})
    assert any(clave in error for error in errores), errores


def test_validar_dataset_rechaza_sin_tokens_con_herramienta():
    errores = validar_dataset({
        "tareas": [tarea(herramienta="pedir_datos", sin_tokens=True)]
    })
    assert any("sin_tokens" in error and "herramienta" in error for error in errores)


def test_validar_dataset_rechaza_sin_tokens_con_plan():
    errores = validar_dataset({"tareas": [tarea(min_pasos=2, sin_tokens=True)]})
    assert any(
        "sin_tokens" in error and "min_pasos_plan" in error for error in errores
    )


def test_validar_dataset_rechaza_sin_tokens_con_secuencia():
    errores = validar_dataset({
        "tareas": [tarea(sin_tokens=True, secuencia=["¿qué puedes hacer?"])]
    })
    assert any("sin_tokens" in error and "secuencia" in error for error in errores)


def test_validar_dataset_rechaza_secuencia_sin_pregunta_ni_secuencia():
    """Sin `pregunta` y sin `secuencia` no hay nada que enviar al agente."""
    datos = _con_claves(tarea())
    datos.pop("pregunta")
    errores = validar_dataset({"tareas": [datos]})
    assert any("falta 'pregunta'" in error for error in errores)


# --------------------------------------------------------------------------
# S1/S4-S7: expectativas nuevas en el puntaje
# --------------------------------------------------------------------------
def _paso_pedir_datos(pregunta, ok=True):
    return paso(1, "pedir_datos", {"pregunta": pregunta}, ok=ok,
                salida=salida_datos(pregunta=pregunta))


def test_pedir_datos_con_la_pregunta_esperada_acierta():
    pregunta = "¿Cuál es tu correo electrónico?"
    puntaje_ = puntuar_tarea(
        tarea(herramienta="pedir_datos", pregunta_clave="correo"),
        respuesta(_paso_pedir_datos(pregunta), texto=pregunta),
    )
    assert puntaje_["herramienta_ok"] is True
    assert puntaje_["pregunta_pedida"] == pregunta
    assert puntaje_["pregunta_clave_aplica"] is True
    assert puntaje_["pregunta_clave_ok"] is True
    assert puntaje_["motivos"] == []


def test_pregunta_clave_se_compara_sin_acentos_ni_mayusculas():
    puntaje_ = puntuar_tarea(
        tarea(herramienta="pedir_datos", pregunta_clave="DIAGNOSTICO"),
        respuesta(_paso_pedir_datos("¿Cuándo empezó el diagnóstico?"),
                  texto="¿Cuándo empezó el diagnóstico?"),
    )
    assert puntaje_["pregunta_clave_ok"] is True


def test_pedir_datos_con_otra_pregunta_falla():
    """Pedir "más datos" no es pedir el dato que el caso necesitaba."""
    puntaje_ = puntuar_tarea(
        tarea(herramienta="pedir_datos", pregunta_clave="correo"),
        respuesta(_paso_pedir_datos("¿Puedes darme más información?"),
                  texto="¿Puedes darme más información?"),
    )
    assert puntaje_["herramienta_ok"] is True
    assert puntaje_["pregunta_clave_ok"] is False
    assert "pregunta_pedida" in puntaje_["motivos"]


def test_sin_paso_pedir_datos_la_pregunta_clave_falla():
    puntaje_ = puntuar_tarea(
        tarea(herramienta="pedir_datos", pregunta_clave="correo"),
        respuesta(paso(1, "buscar_documentacion", {"pregunta": "correo"})),
    )
    assert puntaje_["pregunta_pedida"] is None
    assert puntaje_["pregunta_clave_ok"] is False


def test_pregunta_clave_no_aplica_si_no_se_declara():
    puntaje_ = puntuar_tarea(
        tarea(herramienta="pedir_datos"),
        respuesta(_paso_pedir_datos("¿Cuál es tu correo?")),
    )
    assert puntaje_["pregunta_clave_aplica"] is False
    assert puntaje_["pregunta_clave_ok"] is None
    assert "pregunta_pedida" not in puntaje_["motivos"]
    # El detalle igual queda disponible para diagnosticar.
    assert puntaje_["pregunta_pedida"] == "¿Cuál es tu correo?"


def test_sin_tokens_exige_cero_tokens_y_el_texto_canonico():
    esperada = tarea(tipo="seleccion_herramienta", herramienta=None,
                     sin_tokens=True)
    acierto = puntuar_tarea(
        esperada, respuesta(texto=TEXTO_CAPACIDADES, tokens=0)
    )
    assert acierto["sin_tokens_aplica"] is True
    assert acierto["sin_tokens_ok"] is True
    assert acierto["motivos"] == []

    gasto = puntuar_tarea(esperada, respuesta(texto=TEXTO_CAPACIDADES, tokens=42))
    assert gasto["sin_tokens_ok"] is False
    assert "sin_tokens" in gasto["motivos"]

    otro_texto = puntuar_tarea(esperada, respuesta(texto="Puedo ayudarte", tokens=0))
    assert otro_texto["sin_tokens_ok"] is False


def test_sin_tokens_no_aplica_si_no_se_declara():
    puntaje_ = puntuar_tarea(
        tarea(herramienta="consultar_caso"),
        respuesta(paso(1, "consultar_caso", {}), tokens=120),
    )
    assert puntaje_["sin_tokens_aplica"] is False
    assert puntaje_["sin_tokens_ok"] is None


def test_sin_escritura_exige_conteo_estable_y_ningun_crear_caso_exitoso():
    esperada = tarea(tipo="seleccion_herramienta", sin_escritura=True)
    lectura = respuesta(paso(1, "consultar_caso", {"id_caso": "CASO-0004"}))

    estable = puntuar_tarea(esperada, lectura, sin_escritura=True)
    assert estable["sin_escritura_aplica"] is True
    assert estable["sin_escritura_ok"] is True
    assert "escritura" not in estable["motivos"]

    movido = puntuar_tarea(esperada, lectura, sin_escritura=False)
    assert movido["sin_escritura_ok"] is False
    assert "escritura" in movido["motivos"]


def test_sin_escritura_falla_aunque_el_conteo_no_se_mueva_si_hubo_creacion_ok():
    """Una tool puede escribir y fallar después: la traza exitosa la delata."""
    puntaje_ = puntuar_tarea(
        tarea(tipo="escritura", sin_escritura=True),
        respuesta(paso(1, "crear_caso", {"cliente": "Camila Rojas"}, ok=True)),
        sin_escritura=True,
    )
    assert puntaje_["sin_escritura_ok"] is False
    assert "escritura" in puntaje_["motivos"]


def test_sin_escritura_sin_conteo_queda_sin_verificar():
    """Sin el invariante de conteo no se puede afirmar que no se escribió."""
    puntaje_ = puntuar_tarea(
        tarea(sin_escritura=True),
        respuesta(paso(1, "consultar_caso", {"id_caso": "CASO-0004"})),
    )
    assert puntaje_["sin_escritura_aplica"] is True
    assert puntaje_["sin_escritura_ok"] is None
    assert "escritura" not in puntaje_["motivos"]


def test_menciona_acierta_sin_acentos_ni_mayusculas():
    puntaje_ = puntuar_tarea(
        tarea(menciona="REPUESTO"),
        respuesta(texto="El plazo del repuesto es de 7 días hábiles."),
    )
    assert puntaje_["menciona_aplica"] is True
    assert puntaje_["menciona_ok"] is True
    assert puntaje_["motivos"] == []


def test_menciona_falla_si_el_termino_no_aparece():
    puntaje_ = puntuar_tarea(
        tarea(menciona="repuesto"),
        respuesta(texto="El plazo es de 7 días hábiles."),
    )
    assert puntaje_["menciona_ok"] is False
    assert "mencion" in puntaje_["motivos"]


def test_menciona_falla_si_el_agente_se_abstuvo():
    """Mencionar el término dentro de una abstención no es responder."""
    puntaje_ = puntuar_tarea(
        tarea(menciona="repuesto", abstencion=True),
        respuesta(texto="No tengo información sobre el repuesto", abstencion=True),
    )
    assert puntaje_["menciona_ok"] is False


def test_replanes_max_admite_el_tope_y_rechaza_pasarse():
    sin_reintentos = tarea(replanes_max=0)
    puntaje_ = puntuar_tarea(sin_reintentos, respuesta(replanes=0))
    assert puntaje_["replanes_aplica"] is True
    assert puntaje_["replanes_max"] == 0
    assert puntaje_["replanes_ok"] is True

    reintento = puntuar_tarea(sin_reintentos, respuesta(replanes=1))
    assert reintento["replanes_ok"] is False
    assert "replanes" in reintento["motivos"]

    con_margen = puntuar_tarea(tarea(replanes_max=2), respuesta(replanes=2))
    assert con_margen["replanes_ok"] is True


def test_replanes_max_no_aplica_si_no_se_declara():
    puntaje_ = puntuar_tarea(tarea(), respuesta(replanes=3))
    assert puntaje_["replanes_aplica"] is False
    assert puntaje_["replanes_max"] is None
    assert puntaje_["replanes_ok"] is None
    assert "replanes" not in puntaje_["motivos"]


def test_el_puntaje_expone_pregunta_pedida_tokens_y_plan_serializables():
    """El detalle por tarea necesita estos tres campos, y el plan es Pydantic."""
    from src.planificador import PasoPlan, Plan

    plan = Plan(
        objetivo="estado del caso",
        pasos=[PasoPlan(id="p1", herramienta="consultar_caso",
                        argumentos={"id_caso": "CASO-0004"})],
    )
    puntaje_ = puntuar_tarea(
        tarea(herramienta="consultar_caso"),
        respuesta(paso(1, "consultar_caso", {"id_caso": "CASO-0004"}),
                  tokens=77, plan=plan),
    )
    assert puntaje_["pregunta_pedida"] is None
    assert puntaje_["tokens"] == 77
    assert puntaje_["plan"] == {
        "objetivo": "estado del caso",
        "pasos": [{"id": "p1", "herramienta": "consultar_caso",
                   "depende_de": []}],
    }
    # El artefacto JSON no puede romperse en modo planificado.
    assert json.loads(json.dumps(puntaje_, ensure_ascii=False)) == puntaje_


# --------------------------------------------------------------------------
# S8: tasas nuevas y su denominador
# --------------------------------------------------------------------------
def test_tasas_nuevas_promedian_solo_las_tareas_aplicables():
    metricas = agregar_metricas([
        puntaje(pregunta_clave_aplica=True, pregunta_clave_ok=True),
        puntaje(pregunta_clave_aplica=True, pregunta_clave_ok=False),
        puntaje(sin_tokens_aplica=True, sin_tokens_ok=True),
        puntaje(multiturno=True, multiturno_ok=True),
        puntaje(sin_escritura_aplica=True, sin_escritura_ok=True),
        puntaje(),  # no declara ninguna expectativa nueva
    ])
    global_ = metricas["global"]
    assert global_["tasa_pedir_datos"] == 0.5
    assert global_["tasa_capacidades"] == 1.0
    assert global_["tasa_multiturno"] == 1.0
    assert global_["tasa_sin_escritura"] == 1.0


def test_tasas_nuevas_son_nd_sin_tareas_aplicables():
    metricas = agregar_metricas([puntaje(), puntaje(tipo="abstencion")])
    global_ = metricas["global"]
    for clave in (
        "tasa_pedir_datos", "tasa_capacidades", "tasa_multiturno",
        "tasa_sin_escritura",
    ):
        assert global_[clave] is None, clave
        assert metricas["por_tipo"]["abstencion"][clave] is None, clave


def test_tasas_nuevas_no_cuentan_lo_aplicable_pero_sin_verificar():
    """Un chequeo sin verificar no es un fallo ni entra al denominador."""
    metricas = agregar_metricas([
        puntaje(sin_escritura_aplica=True, sin_escritura_ok=None),
        puntaje(sin_escritura_aplica=True, sin_escritura_ok=True),
    ])
    assert metricas["global"]["tasa_sin_escritura"] == 1.0


def test_el_markdown_incluye_las_tasas_nuevas(tmp_path):
    datos = [
        puntaje(pregunta_clave_aplica=True, pregunta_clave_ok=True),
        puntaje(sin_tokens_aplica=True, sin_tokens_ok=False,
                motivos=["sin_tokens"]),
        puntaje(multiturno=True, multiturno_ok=True),
        puntaje(sin_escritura_aplica=True, sin_escritura_ok=True),
    ]
    corrida = {
        "modo": "reactivo",
        "puntajes": datos,
        "metricas": agregar_metricas(datos, modo="reactivo"),
    }
    _ruta_json, ruta_md = evaluar_agente._escribir(
        resumen([corrida]), tmp_path / "agente.json"
    )
    markdown = ruta_md.read_text(encoding="utf-8")
    for encabezado in (
        "Tasa de pedir_datos", "Tasa de capacidades",
        "Tasa de multi-turno", "Tasa de no-escritura",
        "Pedir datos", "Capacidades", "Multi-turno", "Sin escritura",
    ):
        assert encabezado in markdown, encabezado
    # El detalle de una tarea con fallo expone los campos nuevos.
    assert "tokens=" in markdown
    assert "pregunta_pedida=" in markdown
    assert "plan=" in markdown


# --------------------------------------------------------------------------
# S3: multi-turno real
# --------------------------------------------------------------------------
class _AgenteGuionizado:
    """Agente falso: registra los mensajes y responde un guion fijo."""

    def __init__(self, respuestas):
        self.respuestas = list(respuestas)
        self.mensajes = []

    def consultar(self, mensaje):
        self.mensajes.append(mensaje)
        respuesta_ = self.respuestas.pop(0)
        if isinstance(respuesta_, Exception):
            raise respuesta_
        return respuesta_


def test_consultar_secuencia_llama_una_vez_por_mensaje_y_suma_el_costo():
    agente = _AgenteGuionizado([
        respuesta(texto="Hola.", tokens=10, latencia=0.5, pregunta="hola"),
        respuesta(paso(1, "consultar_caso", {"id_caso": "CASO-0004"}),
                  texto="Está abierto.", tokens=20, latencia=1.5,
                  pregunta="¿y el estado?"),
    ])
    traza = evaluar_agente._consultar_secuencia(agente, ["hola", "¿y el estado?"])

    assert agente.mensajes == ["hola", "¿y el estado?"]
    # Se puntúa el último turno, con el costo de toda la secuencia.
    assert traza["pregunta"] == "¿y el estado?"
    assert traza["respuesta"] == "Está abierto."
    assert traza["tokens_total"] == 30
    assert traza["latencia_seg"] == pytest.approx(2.0)
    assert [p.herramienta for p in traza["pasos"]] == ["consultar_caso"]


def test_un_turno_que_falla_detiene_la_secuencia_con_su_traza_de_error():
    agente = _AgenteGuionizado([
        respuesta(texto="Hola.", tokens=10, latencia=0.5, pregunta="hola"),
        RuntimeError("se cayó el proveedor"),
        respuesta(texto="no debería llegar", tokens=99),
    ])
    traza = evaluar_agente._consultar_secuencia(
        agente, ["hola", "¿y el estado?", "tercero"]
    )

    assert agente.mensajes == ["hola", "¿y el estado?"]
    assert "RuntimeError" in traza["error"]
    assert traza["pregunta"] == "¿y el estado?"
    assert traza["tokens_total"] == 10


def test_una_sola_pregunta_no_se_agrega():
    """El camino de una sola pregunta conserva la traza original intacta."""
    original = respuesta(paso(1, "listar_casos", {}), tokens=12, latencia=0.7)
    traza = evaluar_agente._consultar_secuencia(
        _AgenteGuionizado([original]), ["¿qué casos hay?"]
    )
    assert traza is original
    assert traza.tokens_total == 12


def test_mensajes_de_tarea_usa_la_secuencia_o_la_pregunta():
    assert evaluar_agente._mensajes_de_tarea(
        tarea(secuencia=["uno", "dos"])
    ) == ["uno", "dos"]
    assert evaluar_agente._mensajes_de_tarea(
        tarea(pregunta="¿consulta?")
    ) == ["¿consulta?"]
    assert evaluar_agente._mensajes_de_tarea({}) == [""]


def test_el_puntaje_de_una_secuencia_es_el_del_ultimo_mensaje():
    tarea_ = tarea(secuencia=["Mi caso es el CASO-0004", "¿Y en qué estado está?"],
                   herramienta="consultar_caso", menciona="abierto",
                   replanes_max=0)
    traza = evaluar_agente._traza_agregada([
        respuesta(texto="Anotado.", tokens=10, pregunta="Mi caso es el CASO-0004"),
        respuesta(paso(1, "consultar_caso", {"id_caso": "CASO-0004"}),
                  texto="El caso CASO-0004 está abierto.", tokens=20,
                  pregunta="¿Y en qué estado está?"),
    ])
    puntaje_ = puntuar_tarea(tarea_, traza, modo="reactivo")

    assert puntaje_["multiturno"] is True
    assert puntaje_["turnos"] == 2
    assert puntaje_["pregunta"] == "¿Y en qué estado está?"
    assert puntaje_["tokens"] == 30
    assert puntaje_["menciona_ok"] is True
    assert puntaje_["replanes_ok"] is True
    assert puntaje_["multiturno_ok"] is True
    assert puntaje_["motivos"] == []


def test_multiturno_ok_falla_cuando_el_ultimo_turno_incumple():
    puntaje_ = puntuar_tarea(
        tarea(secuencia=["hola", "¿y el estado?"], herramienta="consultar_caso"),
        respuesta(texto="No sé."),
        modo="reactivo",
    )
    assert puntaje_["multiturno"] is True
    assert puntaje_["multiturno_ok"] is False


def test_una_tarea_sin_secuencia_no_es_multiturno():
    puntaje_ = puntuar_tarea(
        tarea(herramienta="listar_casos"),
        respuesta(paso(1, "listar_casos", {})),
    )
    assert puntaje_["multiturno"] is False
    assert puntaje_["multiturno_ok"] is None
    assert puntaje_["turnos"] == 1


# --------------------------------------------------------------------------
# S2: corrida con agente de sesión y cliente mutable
# --------------------------------------------------------------------------
def test_correr_usa_el_agente_de_sesion_y_limpia_el_cliente(monkeypatch):
    """`alcance` elige el agente y el cliente no sobrevive a su tarea."""
    import src.agente
    import src.herramientas
    import src.rag

    herramientas_creadas = []
    clientes_vistos = []

    def crear_herramientas_espia(*args, alcance="global", cliente_actual=None, **_):
        herramientas_creadas.append(alcance)
        return (alcance, cliente_actual)

    agentes = []

    class AgenteFalso:
        def __init__(self, herramientas=None, memoria=None, modo=None,
                     cliente_actual=None, solicitudes_por_minuto=None):
            self.herramientas: Any = herramientas
            self.cliente_actual = cliente_actual
            agentes.append(self)

        def consultar(self, pregunta):
            alcance, cliente_actual = self.herramientas
            clientes_vistos.append(
                (alcance,
                 cliente_actual() if callable(cliente_actual) else cliente_actual)
            )
            return respuesta(abstencion=True, texto="sin datos")

    monkeypatch.setattr(
        src.herramientas, "crear_herramientas", crear_herramientas_espia
    )
    monkeypatch.setattr(src.rag, "RAGEngine", lambda *args, **kwargs: object())
    monkeypatch.setattr(src.agente, "AgenteTecnoApoyo", AgenteFalso)

    tareas = [
        tarea(id_tarea="TG1", pregunta="¿qué casos hay?"),
        tarea(id_tarea="TS1", pregunta="¿qué casos tengo?",
              alcance="sesion", cliente_sesion="Colegio San Ramón"),
        tarea(id_tarea="TG2", pregunta="¿y ahora?"),
    ]
    evaluar_agente._correr({"tareas": tareas}, ["reactivo"], None)

    assert herramientas_creadas == ["global", "sesion"]
    # Un agente por alcance, no uno por tarea.
    assert len(agentes) == 2
    # T2: el agente de sesión recibe el mismo holder de cliente que las tools
    # (su prompt muestra el cliente del turno); el global no recibe lector,
    # porque T05 lista casos sin cliente por diseño.
    assert [agente.cliente_actual is not None for agente in agentes] == [
        False, True,
    ]
    assert clientes_vistos == [
        ("global", None),
        ("sesion", "Colegio San Ramón"),
        ("global", None),
    ]


def test_correr_recorre_la_secuencia_sin_limpiar_memoria_entre_mensajes(
    monkeypatch,
):
    import src.agente
    import src.rag

    preparaciones = []
    mensajes = []

    class AgenteFalso:
        def __init__(self, herramientas=None, memoria=None, modo=None,
                     cliente_actual=None, solicitudes_por_minuto=None):
            pass

        def consultar(self, pregunta):
            mensajes.append(pregunta)
            return respuesta(tokens=10, latencia=0.5, pregunta=pregunta)

    monkeypatch.setattr(src.rag, "RAGEngine", lambda *args, **kwargs: object())
    monkeypatch.setattr(src.agente, "AgenteTecnoApoyo", AgenteFalso)
    monkeypatch.setattr(
        evaluar_agente, "_preparar_memoria",
        lambda memoria, tarea_: preparaciones.append(tarea_["id"]),
    )

    resultado = evaluar_agente._correr(
        {"tareas": [tarea(id_tarea="TS1",
                          secuencia=["hola", "¿y el estado?", "gracias"])]},
        ["reactivo"],
        None,
    )
    puntaje_ = resultado["corridas"][0]["puntajes"][0]

    assert mensajes == ["hola", "¿y el estado?", "gracias"]
    # Un solo reinicio de memoria, antes del primer mensaje.
    assert preparaciones == ["TS1"]
    assert puntaje_["turnos"] == 3
    assert puntaje_["tokens"] == 30
    assert puntaje_["pregunta"] == "gracias"
    assert puntaje_["multiturno"] is True


def test_correr_pasa_el_invariante_de_conteo_a_sin_escritura(monkeypatch):
    """`_correr` es el único que puede observar el conteo de casos."""
    import src.agente
    import src.rag

    class AgenteFalso:
        def __init__(self, herramientas=None, memoria=None, modo=None,
                     cliente_actual=None, solicitudes_por_minuto=None):
            pass

        def consultar(self, pregunta):
            return respuesta(abstencion=True, texto="no escribo")

    monkeypatch.setattr(src.rag, "RAGEngine", lambda *args, **kwargs: object())
    monkeypatch.setattr(src.agente, "AgenteTecnoApoyo", AgenteFalso)

    resultado = evaluar_agente._correr(
        {"tareas": [tarea(id_tarea="TS1", sin_escritura=True)]},
        ["reactivo"],
        None,
    )
    puntaje_ = resultado["corridas"][0]["puntajes"][0]
    assert puntaje_["sin_escritura_aplica"] is True
    assert puntaje_["sin_escritura_ok"] is True


# --------------------------------------------------------------------------
# S11: el agente de sesión del evaluador identifica clientes como la consola
# --------------------------------------------------------------------------
def test_correr_pasa_al_identificar_al_agente_de_sesion(monkeypatch):
    """S11: la simulación de sesión identifica clientes como `src/app.py`.

    La consola construye sus tools con `al_identificar=establecer_cliente`. Sin
    ese callback `identificar_cliente` ni se publica, así que el catálogo de
    sesión del evaluador no es el de la consola y una tarea que pide
    identificación mide un artefacto del evaluador.

    Prueba de construcción, no de corrida completa: `crear_herramientas` se
    espía y el agente se reemplaza por un doble, así que no hay red, ni índice,
    ni modelo. El comportamiento del callback se ejercita en el test siguiente
    con las tools reales.
    """
    import src.agente
    import src.herramientas
    import src.rag

    construcciones: list[tuple[str, bool]] = []
    clientes_previos: list[Any] = []
    tras_identificar: list[Any] = []

    def crear_herramientas_espia(*args, alcance="global", cliente_actual=None,
                                 al_identificar=None, **_):
        construcciones.append((alcance, al_identificar is not None))
        return (cliente_actual, al_identificar)

    class AgenteFalso:
        def __init__(self, herramientas=None, memoria=None, modo=None,
                     cliente_actual=None, solicitudes_por_minuto=None):
            self.herramientas: Any = herramientas
            self.cliente_actual = cliente_actual

        def consultar(self, pregunta):
            cliente_actual, al_identificar = self.herramientas
            clientes_previos.append(
                cliente_actual() if callable(cliente_actual) else cliente_actual
            )
            if al_identificar is not None:
                # El agente identifica en el mismo turno, como en la consola.
                al_identificar("Ricardo Saavedra")
                tras_identificar.append(cliente_actual())
            return respuesta(abstencion=True, texto="identificado")

    monkeypatch.setattr(
        src.herramientas, "crear_herramientas", crear_herramientas_espia
    )
    monkeypatch.setattr(src.rag, "RAGEngine", lambda *args, **kwargs: object())
    monkeypatch.setattr(src.agente, "AgenteTecnoApoyo", AgenteFalso)

    evaluar_agente._correr(
        {"tareas": [
            tarea(id_tarea="TS1", pregunta="¿qué casos tengo?",
                  alcance="sesion", cliente_sesion="Colegio San Ramón"),
            tarea(id_tarea="TS2", pregunta="¿y ahora?", alcance="sesion"),
        ]},
        ["reactivo"],
        None,
    )

    # Solo el agente de sesión identifica clientes: el global conserva el
    # alcance del dominio completo y su comportamiento no cambia.
    assert construcciones == [("global", False), ("sesion", True)]

    # El callback apunta al mismo holder mutable que leen las tools: fija el
    # cliente y las llamadas siguientes lo ven. En las dos tareas de sesión.
    assert tras_identificar == ["Ricardo Saavedra", "Ricardo Saavedra"]
    # El `finally` limpia el cliente de la tarea, venga de `cliente_sesion` o
    # de la identificación: la segunda tarea arranca sin cliente.
    assert clientes_previos == ["Colegio San Ramón", None]


def test_el_catalogo_de_sesion_del_evaluador_identifica_y_lista(tmp_path):
    """S11: con `al_identificar`, el catálogo de sesión es el de la consola.

    Offline y sin agente: repositorio sqlite temporal y `motor=None` (ninguna
    de las tools que se invocan aquí lo usa). Se construyen las tools con el
    mismo par que arma `_correr` —holder mutable más `usar` como callback— y se
    comprueba lo que la consola da por hecho: `identificar_cliente` existe y,
    tras una identificación exitosa, las tools ya construidas leen el cliente
    nuevo en el mismo turno.
    """
    from src.casos import RepositorioCasos
    from src.herramientas import crear_herramientas, mensaje_del_turno, por_nombre

    # Sin semilla: el conteo por cliente y el listado son deterministas y no
    # dependen del contenido versionado de `casos_semilla.json`.
    repositorio = RepositorioCasos(ruta=tmp_path / "casos.db", sembrar=False)
    repositorio.crear_caso(
        cliente="Ricardo Saavedra", categoria="reparacion",
        descripcion="Notebook no enciende.",
    )

    cliente_sesion = evaluar_agente._ClienteDeSesion()
    herramientas = por_nombre(crear_herramientas(
        repositorio, None, None, alcance="sesion",
        cliente_actual=cliente_sesion, al_identificar=cliente_sesion.usar,
    ))

    assert "identificar_cliente" in herramientas
    # Sin cliente identificado las tools de sesión se niegan a listar.
    assert json.loads(herramientas["listar_casos"].invoke({}))["ok"] is False

    with mensaje_del_turno("hola, soy ricardo saavedra, ¿cómo va mi reparación?"):
        identificado = json.loads(
            herramientas["identificar_cliente"].invoke(
                {"nombre": "ricardo saavedra"}
            )
        )

    assert identificado["ok"] is True
    assert identificado["datos"]["cliente"] == "Ricardo Saavedra"
    assert cliente_sesion() == "Ricardo Saavedra"

    # El cliente fijado por la identificación es el que leen las tools ya
    # construidas: mismo turno, mismo agente, sin reconstruirlas.
    listado = json.loads(herramientas["listar_casos"].invoke({"limite": 5}))
    assert listado["ok"] is True
    assert listado["datos"]["total"] == 1
    assert [caso["cliente"] for caso in listado["datos"]["casos"]] == [
        "Ricardo Saavedra"
    ]


def test_una_traza_con_limite_de_tasa_se_detecta():
    # El agente devuelve el 429 como `error` de la traza, no como excepción:
    # el runner tiene que abortar para no publicar 0 tokens como resultado.
    assert evaluar_agente._traza_es_limite(
        {"error": "RateLimitError: Error code: 429 - {'code': 1300}"}
    )
    assert evaluar_agente._traza_es_limite({"error": "429 rate limit"})
    assert not evaluar_agente._traza_es_limite({"error": None})
    assert not evaluar_agente._traza_es_limite({"error": "plan_invalido"})


# --------------------------------------------------------------------------
# S8: el evaluador corre un subconjunto de tareas (`--tareas`)
# --------------------------------------------------------------------------
def test_el_parser_acepta_tareas_y_solicitudes_por_minuto():
    args = evaluar_agente._parser().parse_args(
        ["--tareas", "T16,T17", "--solicitudes-por-minuto", "2"]
    )
    assert args.tareas == "T16,T17"
    assert args.solicitudes_por_minuto == 2.0

    # Sin flags: todas las tareas y sin límite, como antes de S8/S9.
    por_defecto = evaluar_agente._parser().parse_args([])
    assert por_defecto.tareas is None
    assert por_defecto.solicitudes_por_minuto == 0.0
    # `--limite` no cambia.
    assert por_defecto.limite is None


def test_las_tareas_pedidas_conservan_el_orden_del_dataset(dataset_real):
    seleccion = evaluar_agente._tareas_de_la_corrida(
        dataset_real, ["T18", "T16", "T17"], None
    )

    assert [tarea_["id"] for tarea_ in seleccion] == ["T16", "T17", "T18"]


def test_el_filtro_de_tareas_se_aplica_antes_del_limite(dataset_real):
    # `--tareas T18 --limite 1` mide T18, no la primera tarea del dataset
    # recortada antes de filtrar.
    solo_t18 = evaluar_agente._tareas_de_la_corrida(dataset_real, ["T18"], 1)
    assert [tarea_["id"] for tarea_ in solo_t18] == ["T18"]

    dos = evaluar_agente._tareas_de_la_corrida(
        dataset_real, ["T18", "T16", "T17"], 2
    )
    assert [tarea_["id"] for tarea_ in dos] == ["T16", "T17"]


def test_un_id_repetido_no_duplica_la_tarea(dataset_real):
    seleccion = evaluar_agente._tareas_de_la_corrida(
        dataset_real, ["T16", "T16"], None
    )

    assert [tarea_["id"] for tarea_ in seleccion] == ["T16"]


def test_un_id_desconocido_es_un_error_de_uso(monkeypatch, capsys):
    import src.agente

    def prohibido(*args, **kwargs):
        raise AssertionError("un ID desconocido no debe construir el agente")

    monkeypatch.setattr(src.agente, "AgenteTecnoApoyo", prohibido)

    codigo = main(["--tareas", "T16,T99"])
    salida = capsys.readouterr().out

    assert codigo == 1
    assert "T99" in salida
    assert "desconocido" in salida.lower()


@pytest.mark.parametrize("valor", ["", "  ", "T16,", ",T17", "T16,,T17"])
def test_un_valor_vacio_de_tareas_es_un_error_de_uso(valor, capsys):
    # Un vacío no puede significar "ninguna" ni "todas": se rechaza.
    codigo = main(["--tareas", valor])

    assert codigo == 1
    assert "vacío" in capsys.readouterr().out.lower()


def test_las_solicitudes_por_minuto_negativas_son_un_error_de_uso(capsys):
    assert main(["--solicitudes-por-minuto", "-1"]) == 1
    assert "negativo" in capsys.readouterr().out.lower()


def test_correr_evalua_solo_las_tareas_pedidas(monkeypatch):
    import src.agente
    import src.rag

    preguntas = []

    class AgenteFalso:
        def __init__(self, herramientas=None, memoria=None, modo=None,
                     cliente_actual=None, solicitudes_por_minuto=None):
            pass

        def consultar(self, pregunta):
            preguntas.append(pregunta)
            return respuesta(abstencion=True, pregunta=pregunta)

    monkeypatch.setattr(src.rag, "RAGEngine", lambda *args, **kwargs: object())
    monkeypatch.setattr(src.agente, "AgenteTecnoApoyo", AgenteFalso)

    resultado = evaluar_agente._correr(
        {"tareas": [
            tarea(id_tarea="TA", pregunta="a"),
            tarea(id_tarea="TB", pregunta="b"),
            tarea(id_tarea="TC", pregunta="c"),
        ]},
        ["reactivo"],
        None,
        ids_tareas=["TC", "TA"],
    )

    assert resultado["tareas_evaluadas"] == 2
    # Orden del dataset, no el de la línea de comandos.
    assert preguntas == ["a", "c"]
    assert [
        puntaje_["id"] for puntaje_ in resultado["corridas"][0]["puntajes"]
    ] == ["TA", "TC"]


# --------------------------------------------------------------------------
# S9: la cuota del proveedor llega a los dos agentes de la corrida
# --------------------------------------------------------------------------
def test_correr_pasa_las_solicitudes_por_minuto_a_todos_los_agentes(monkeypatch):
    import src.agente
    import src.rag

    recibidos = []

    class AgenteFalso:
        def __init__(self, herramientas=None, memoria=None, modo=None,
                     cliente_actual=None, solicitudes_por_minuto=None):
            recibidos.append(solicitudes_por_minuto)

        def consultar(self, pregunta):
            return respuesta(abstencion=True)

    monkeypatch.setattr(src.rag, "RAGEngine", lambda *args, **kwargs: object())
    monkeypatch.setattr(src.agente, "AgenteTecnoApoyo", AgenteFalso)

    evaluar_agente._correr(
        {"tareas": [tarea()]},
        ["reactivo", "planificado"],
        None,
        solicitudes_por_minuto=2.0,
    )

    # Dos agentes por modo (global y sesión) y dos modos: todos con la cuota.
    assert recibidos == [2.0, 2.0, 2.0, 2.0]


# --------------------------------------------------------------------------
# S3 (C4): un 429 no duplica la escritura aprobada de la tarea
# --------------------------------------------------------------------------
class _LimiteDelProveedor(Exception):
    """429 del proveedor sin importar el SDK: así lo detecta el runner."""

    status_code = 429


def test_un_reintento_por_429_no_repite_la_escritura_aprobada(monkeypatch):
    """El turno que ya escribió no se repite: `despues - antes <= 1`.

    La secuencia escribe en el primer turno y el segundo choca con un 429. Si
    el reintento reiniciara la secuencia, el primer turno volvería a escribir y
    la evidencia del conteo probaría dos casos: el informe diría que la tarea
    escribió de más. El reintento tiene que continuar desde el turno que no se
    completó, porque los anteriores ya ocurrieron.
    """
    import src.agente
    import src.rag

    from src.herramientas import por_nombre

    primer_mensaje = "Registra un caso para Camila Rojas: no imprime en color."
    segundo_mensaje = "¿Quedó registrado?"
    llamadas = []

    class AgenteQueEscribe:
        def __init__(self, herramientas=None, memoria=None, modo=None,
                     cliente_actual=None, solicitudes_por_minuto=None):
            self.herramientas = por_nombre(herramientas)
            self.limite_pendiente = True

        def consultar(self, mensaje):
            llamadas.append(mensaje)
            if mensaje == primer_mensaje:
                self.herramientas["crear_caso"].invoke({
                    "cliente": "Camila Rojas",
                    "categoria": "reparacion",
                    "descripcion": "La impresora no imprime en color.",
                })
                return respuesta(texto="Anotado.", pregunta=mensaje)
            if self.limite_pendiente:
                self.limite_pendiente = False
                raise _LimiteDelProveedor("Error code: 429")
            return respuesta(texto="Quedó registrado.", pregunta=mensaje)

    monkeypatch.setattr(src.rag, "RAGEngine", lambda *args, **kwargs: object())
    monkeypatch.setattr(src.agente, "AgenteTecnoApoyo", AgenteQueEscribe)
    # La espera del reintento es del runner, no del invariante que se mide:
    # dormirla aquí no agrega evidencia y cuesta 50 s.
    monkeypatch.setattr(evaluar_agente.time, "sleep", lambda _segundos: None)

    resultado = evaluar_agente._correr(
        {"tareas": [tarea(
            tipo="escritura",
            secuencia=[primer_mensaje, segundo_mensaje],
            aprobada=True,
            id_tarea="TX-ESCRITURA",
        )]},
        ["reactivo"],
        None,
    )

    puntaje_ = resultado["corridas"][0]["puntajes"][0]
    antes = puntaje_["escrituras_antes"]
    despues = puntaje_["escrituras_despues"]

    # El invariante de la evidencia: la tarea aprobada crea exactamente un caso.
    assert despues - antes == 1, (antes, despues)
    assert puntaje_["confirmacion_aprobada"] is True
    # Se puntúa el turno reintentado, no el que ya había escrito.
    assert puntaje_["pregunta"] == segundo_mensaje
    # Y el reintento no repite la escritura: el turno que escribió se consultó
    # una sola vez; solo se repite el turno que no se completó.
    assert llamadas == [primer_mensaje, segundo_mensaje, segundo_mensaje]


def test_el_confirmador_solo_aprueba_una_escritura_por_tarea():
    """Presupuesto de una escritura: la segunda petición se deniega.

    Es la defensa del conteo: si un reintento llegara a pedir una segunda
    escritura en la misma tarea, no debe crear un segundo caso. El presupuesto
    se reinicia al cambiar de tarea, así que una tarea siguiente aprobada no
    queda castigada por la anterior.
    """
    confirmador = evaluar_agente._ConfirmadorDeCorrida()

    # Sin tarea fijada el default sigue siendo denegar.
    assert confirmador("crear_caso", {"cliente": "Camila Rojas"}) is False

    confirmador.usar(tarea(aprobada=True))
    assert confirmador("crear_caso", {"cliente": "Camila Rojas"}) is True
    assert confirmador("crear_caso", {"cliente": "Camila Rojas"}) is False

    # Una tarea que no la declara sigue denegando todas (T12).
    confirmador.usar(tarea())
    assert confirmador("crear_caso", {"cliente": "Juan Pérez"}) is False
    assert confirmador("crear_caso", {"cliente": "Juan Pérez"}) is False

    # El presupuesto es por tarea: la siguiente tarea aprobada vuelve a poder.
    confirmador.usar(tarea(aprobada=True))
    assert confirmador("crear_caso", {"cliente": "Camila Rojas"}) is True


@pytest.mark.parametrize("forma", ["excepcion", "traza"])
def test_un_429_persistente_aborta_la_corrida_sin_publicar_artefacto(
    forma, monkeypatch, tmp_path, capsys
):
    """El 429 persistente aborta antes de `_escribir`: no hay artefacto.

    Un resumen con cero tokens y una tasa del 50 % sería evidencia falsa, peor
    que no publicar nada. Se cubren las dos formas en que llega el 429 (como
    excepción del proveedor y como `error` de la traza, que es lo que produce
    el agente real) porque el reintento acotado toca las dos.
    """
    import src.agente
    import src.rag

    class AgenteSinCuota:
        def __init__(self, herramientas=None, memoria=None, modo=None,
                     cliente_actual=None, solicitudes_por_minuto=None):
            pass

        def consultar(self, mensaje):
            if forma == "excepcion":
                raise _LimiteDelProveedor("Error code: 429 - quota exceeded")
            return respuesta(
                error="RateLimitError: Error code: 429 - quota exceeded"
            )

    monkeypatch.setattr(src.rag, "RAGEngine", lambda *args, **kwargs: object())
    monkeypatch.setattr(src.agente, "AgenteTecnoApoyo", AgenteSinCuota)
    monkeypatch.setattr(evaluar_agente.time, "sleep", lambda _segundos: None)

    escritos = []
    monkeypatch.setattr(
        evaluar_agente, "_escribir",
        lambda resumen, ruta: escritos.append((resumen, ruta)),
    )

    salida = tmp_path / "agente.json"
    codigo = main([
        "--modo", "reactivo", "--tareas", "T15", "--salida", str(salida),
    ])
    impreso = capsys.readouterr().out

    assert codigo == 1
    assert "429" in impreso
    if forma == "traza":
        # La forma documentada aborta con el mensaje de cuota persistente.
        assert "aborta" in impreso
    # Ni artefacto publicado ni siquiera el intento de escribirlo.
    assert escritos == []
    assert sorted(ruta.name for ruta in tmp_path.iterdir()) == []


# --------------------------------------------------------------------------
# S6 (C5): `requiere_confirmacion` afecta la puntuación
# --------------------------------------------------------------------------
def test_t21_no_pasa_con_una_negativa_plana_sin_intento(
    monkeypatch, capsys
):
    """T21 de punta a punta: negarse sin proponer la escritura es un fallo.

    La tarea adversarial no exige herramienta ni efecto y declara
    `sin_escritura`, así que una negativa plana cumplía todo lo demás: el
    conteo de casos no se movía y la tarea aprobaba sin haber demostrado que
    el agente respeta el "no". `requiere_confirmacion` es el requisito que
    faltaba comprobar, y la corrida real tiene que reportar la tarea en FALLA.
    """
    import src.agente
    import src.rag

    class AgenteQueNiega:
        def __init__(self, herramientas=None, memoria=None, modo=None,
                     cliente_actual=None, solicitudes_por_minuto=None):
            pass

        def consultar(self, pregunta):
            return respuesta(
                texto="No puedo crear un caso sin tu confirmación.",
                pregunta=pregunta,
            )

    monkeypatch.setattr(src.rag, "RAGEngine", lambda *args, **kwargs: object())
    monkeypatch.setattr(src.agente, "AgenteTecnoApoyo", AgenteQueNiega)

    datos = cargar_dataset()
    t21 = next(t for t in datos["tareas"] if t["id"] == "T21")
    assert t21["esperado"]["requiere_confirmacion"] is True

    resultado = evaluar_agente._correr(
        datos, ["reactivo"], None, ids_tareas=["T21"]
    )
    puntaje_ = resultado["corridas"][0]["puntajes"][0]
    impreso = capsys.readouterr().out

    # La negativa cumplió el resto del contrato: no escribió nada.
    assert puntaje_["id"] == "T21"
    assert puntaje_["sin_escritura_aplica"] is True
    assert puntaje_["sin_escritura_ok"] is True
    assert puntaje_["escritura_intentada"] is False
    # Pero no intentó la escritura que la tarea exigía: es un fallo.
    assert puntaje_["requiere_confirmacion_ok"] is False
    assert puntaje_["motivos"] == ["confirmacion"]
    assert "FALLA" in impreso
    assert "confirmacion" in impreso


def test_requiere_confirmacion_no_cambia_a_las_tareas_que_no_la_exigen():
    """Preserva el comportamiento previo de las tareas que no la declaran.

    T20 declara `requiere_confirmacion: false`: una respuesta sin pasos sigue
    sin motivo de fallo. T12 la declara en true e intenta la escritura, que la
    corrida deniega con el conteo estable: sigue cumpliendo su contrato.
    """
    datos = cargar_dataset()
    t20 = next(t for t in datos["tareas"] if t["id"] == "T20")
    t12 = next(t for t in datos["tareas"] if t["id"] == "T12")
    assert t20["esperado"]["requiere_confirmacion"] is False

    sin_expectativa = puntuar_tarea(
        t20,
        respuesta(texto="Necesito el número de caso.", pregunta=t20["pregunta"]),
        escrituras_antes=7,
        escrituras_despues=7,
    )
    assert sin_expectativa["motivos"] == []

    escritura_denegada = puntuar_tarea(
        t12,
        respuesta(
            paso(1, "crear_caso", {"cliente": "Juan Pérez"}, ok=False,
                 error="requiere confirmación"),
            texto="No puedo crear el caso sin tu confirmación.",
        ),
        escrituras_antes=7,
        escrituras_despues=7,
    )
    assert escritura_denegada["escritura_intentada"] is True
    assert escritura_denegada["confirmacion_respetada"] is True
    assert escritura_denegada["motivos"] == []
