"""Puente entre el agente y la observabilidad (RA2 / IL2.4).

Verifica lo que la consola consume: fuentes legibles desde un paso real de
`buscar_documentacion` y el armado de la traza a partir de la respuesta del
agente. Sin red.
"""
import json

from src.agente import PasoAgente, RespuestaAgente
from src.observabilidad import (
    escribir_traza,
    fuentes_de,
    leer_trazas,
    resumen_stats,
    traza_de_turno,
)


def salida_busqueda(fragmentos):
    return json.dumps({"ok": True, "datos": {"fragmentos": fragmentos,
                                             "evidencia_suficiente": True},
                       "error": None}, ensure_ascii=False)


def paso_busqueda(fragmentos, ok=True):
    return PasoAgente(
        numero=1, herramienta="buscar_documentacion",
        argumentos={"pregunta": "¿cuánto demora?"},
        salida=salida_busqueda(fragmentos), ok=ok, duracion_seg=0.12,
    )


FRAGMENTO = {"document_id": "PROC-004",
             "titulo": "Política de presupuestos y diagnósticos",
             "score": 0.88}


def respuesta(pasos=(), abstencion=False, plan=None):
    return RespuestaAgente(
        pregunta="¿cuánto demora el diagnóstico?",
        respuesta="Hasta 48 horas hábiles.",
        abstencion=abstencion,
        pasos=list(pasos),
        iteraciones=1,
        tokens_total=10,
        latencia_seg=0.5,
        error=None,
        plan=plan,
    )


# --- fuentes -------------------------------------------------------------
def test_fuentes_de_extrae_documento_titulo_y_score():
    assert fuentes_de([paso_busqueda([FRAGMENTO])]) == [FRAGMENTO]


def test_fuentes_de_deduplica_y_conserva_el_mejor_score():
    peor = {"document_id": "PROC-004", "titulo": "Política", "score": 0.4}
    mejor = {"document_id": "PROC-004", "titulo": "Política", "score": 0.9}
    otro = {"document_id": "FAQ-001", "titulo": "Horarios", "score": 0.7}
    fuentes = fuentes_de([paso_busqueda([peor]), paso_busqueda([mejor, otro])])
    por_documento = {fuente["document_id"]: fuente["score"] for fuente in fuentes}
    assert por_documento == {"PROC-004": 0.9, "FAQ-001": 0.7}
    # Ordenadas de mayor a menor relevancia.
    assert [fuente["document_id"] for fuente in fuentes] == ["PROC-004", "FAQ-001"]


def test_fuentes_de_ignora_pasos_fallidos_y_salidas_no_json():
    fallido = paso_busqueda([FRAGMENTO], ok=False)
    roto = PasoAgente(numero=2, herramienta="buscar_documentacion",
                      argumentos={}, salida="no es json", ok=True)
    otro_tipo = PasoAgente(numero=3, herramienta="listar_casos",
                           argumentos={}, salida=salida_busqueda([FRAGMENTO]),
                           ok=True)
    assert fuentes_de([fallido, roto, otro_tipo]) == []


def test_fuentes_de_sin_pasos_devuelve_lista_vacia():
    assert fuentes_de([]) == []
    assert fuentes_de(None) == []


# --- armado de traza -----------------------------------------------------
def test_traza_de_turno_reubica_los_campos_del_agente():
    traza = traza_de_turno(
        respuesta(pasos=[paso_busqueda([FRAGMENTO])]),
        modo="planificado",
        estrategia_memoria="ventana",
        hechos_inyectados=2,
    )
    assert traza["modo"] == "planificado"
    assert traza["pregunta"] == "¿cuánto demora el diagnóstico?"
    assert traza["abstencion"] is False
    assert traza["tokens"] == 10
    assert traza["latencia_seg"] == 0.5
    assert traza["replanes"] == 0
    assert traza["presupuesto_usado"] == 1
    assert traza["estrategia_memoria"] == "ventana"
    assert traza["hechos_inyectados"] == 2
    assert traza["pasos"][0]["herramienta"] == "buscar_documentacion"
    assert traza["pasos"][0]["duracion_seg"] == 0.12
    assert traza["pasos"][0]["ok"] is True
    assert traza["timestamp"]


def test_traza_de_turno_serializa_el_plan():
    from src.planificador import PasoPlan, Plan

    plan = Plan(objetivo="resolver", pasos=[
        PasoPlan(id="p1", herramienta="buscar_documentacion",
                 argumentos={"pregunta": "x"}),
    ])
    traza = traza_de_turno(respuesta(plan=plan), modo="planificado")
    # `plan` viaja como dict, no como modelo: el JSONL no sabe de Pydantic.
    assert traza["plan"]["objetivo"] == "resolver"
    assert traza["plan"]["pasos"][0]["id"] == "p1"


def test_la_traza_armada_se_escribe_y_se_relee(tmp_path):
    traza = traza_de_turno(respuesta(pasos=[paso_busqueda([FRAGMENTO])]),
                           modo="reactivo", estrategia_memoria="buffer")
    ruta = escribir_traza(traza, ruta=tmp_path)
    assert ruta is not None
    leidas = leer_trazas(limite=10, ruta=tmp_path)
    assert len(leidas) == 1
    assert leidas[0]["pregunta"] == traza["pregunta"]
    assert leidas[0]["pasos"][0]["herramienta"] == "buscar_documentacion"


def test_resumen_stats_cuenta_las_trazas_del_agente(tmp_path):
    escribir_traza(traza_de_turno(respuesta(pasos=[paso_busqueda([FRAGMENTO])]),
                                  modo="reactivo"), ruta=tmp_path)
    escribir_traza(traza_de_turno(
        respuesta(pasos=[paso_busqueda([FRAGMENTO])], abstencion=True),
        modo="planificado"), ruta=tmp_path)

    datos = resumen_stats(ruta=tmp_path)
    assert datos["consultas"] == 2
    assert datos["abstenciones"] == 1
    assert datos["pasos"] == 2
    assert datos["por_herramienta"]["buscar_documentacion"] == 2
    assert datos["por_modo"] == {"reactivo": 1, "planificado": 1}


def test_un_error_de_escritura_no_rompe_el_turno(tmp_path, monkeypatch):
    import src.observabilidad as observabilidad

    def escribir_que_falla(*args, **kwargs):
        return None

    monkeypatch.setattr(observabilidad, "escribir_traza", escribir_que_falla)
    # El turno se arma igual y la respuesta del agente no depende de la traza.
    traza = traza_de_turno(respuesta(), modo="reactivo")
    assert traza["pregunta"]
    assert observabilidad.escribir_traza(traza, ruta=tmp_path) is None
