"""Contratos de la observabilidad: trazas JSONL por día, saneado y agregados.

Sin red y sin directorio compartido: cada test escribe en su propio `tmp_path`,
así que el estado de un test nunca condiciona al siguiente.
"""
import json
from datetime import date

from src import observabilidad


def _turno(pregunta="¿Cuál es el plazo de reparación?", **extra):
    turno = {
        "pregunta": pregunta,
        "respuesta": "De 5 días hábiles.",
        "abstencion": False,
        "modo": "reactivo",
        "tokens": 120,
        "latencia_seg": 0.4,
        "pasos": [
            {"numero": 1, "herramienta": "buscar_documentacion", "ok": True},
        ],
    }
    turno.update(extra)
    return turno


def _lineas(archivo):
    return archivo.read_text(encoding="utf-8").splitlines()


def test_escribir_dos_trazas_y_leerlas_en_orden(tmp_path):
    primera = _turno(pregunta="Primera pregunta")
    segunda = _turno(pregunta="Segunda pregunta", abstencion=True)

    ruta_primera = observabilidad.escribir_traza(primera, ruta=tmp_path)
    ruta_segunda = observabilidad.escribir_traza(segunda, ruta=tmp_path)

    assert ruta_primera == ruta_segunda
    assert ruta_primera is not None
    assert ruta_primera.exists()

    trazas = observabilidad.leer_trazas(ruta=tmp_path)
    assert [traza["pregunta"] for traza in trazas] == [
        "Primera pregunta",
        "Segunda pregunta",
    ]
    # Contenido íntegro: la traza vuelve igual a como se escribió.
    assert trazas[0]["respuesta"] == primera["respuesta"]
    assert trazas[0]["modo"] == "reactivo"
    assert trazas[0]["tokens"] == 120
    assert trazas[0]["latencia_seg"] == 0.4
    assert trazas[0]["pasos"] == primera["pasos"]
    assert trazas[1]["abstencion"] is True


def test_el_archivo_es_por_dia_y_lleva_la_fecha_de_hoy(tmp_path):
    archivo = observabilidad.escribir_traza(_turno(), ruta=tmp_path)

    hoy = date.today().isoformat()
    assert archivo is not None
    assert archivo.name == f"trazas_{hoy}.jsonl"
    assert archivo.parent == tmp_path
    # Una línea por traza: el archivo es JSONL, no un JSON acumulado.
    observabilidad.escribir_traza(_turno(pregunta="Otra"), ruta=tmp_path)
    assert len(_lineas(archivo)) == 2
    for linea in _lineas(archivo):
        assert json.loads(linea)["pregunta"]


def test_crea_el_directorio_de_trazas_si_falta(tmp_path):
    destino = tmp_path / "output" / "trazas"

    archivo = observabilidad.escribir_traza(_turno(), ruta=destino)

    assert archivo is not None
    assert destino.is_dir()
    assert archivo.parent == destino


def test_fallo_de_escritura_no_propaga_y_devuelve_none(tmp_path):
    # Un archivo existente usado como directorio: `mkdir` y `open` fallan.
    archivo_bloqueante = tmp_path / "no_es_directorio"
    archivo_bloqueante.write_text("ocupado", encoding="utf-8")

    resultado = observabilidad.escribir_traza(_turno(), ruta=archivo_bloqueante)

    assert resultado is None
    assert archivo_bloqueante.read_text(encoding="utf-8") == "ocupado"


def test_valor_no_serializable_tampoco_propaga(tmp_path):
    class Exotico:
        def __repr__(self):
            return "Exotico()"

    archivo = observabilidad.escribir_traza(
        _turno(extra=Exotico()), ruta=tmp_path
    )

    assert archivo is not None
    # `default=str` conserva la traza en vez de perderla entera.
    assert json.loads(_lineas(archivo)[0])["extra"] == "Exotico()"


def test_saneado_descarta_api_key_y_password(tmp_path):
    # Las claves sensibles y sus valores viven en un dict, no como kwargs: son
    # datos de prueba que nombran las marcas de `CLAVES_SENSIBLES`, no
    # credenciales del proyecto.
    sensibles = {
        "api_key": "VALOR_API_KEY",
        "password": "VALOR_PASSWORD",
        "APIKEY": "VALOR_APIKEY",
        "token_secret": "VALOR_TOKEN_SECRET",
        "mi_secret_guardado": "VALOR_SECRET",
    }
    turno = _turno(
        **sensibles,
        pasos=[
            {
                "herramienta": "consultar_caso",
                "argumentos": {"id": "CASO-0001", **sensibles},
                "ok": True,
            }
        ],
    )

    archivo = observabilidad.escribir_traza(turno, ruta=tmp_path)

    assert archivo is not None
    texto = archivo.read_text(encoding="utf-8")
    for filtrada in sensibles.values():
        assert filtrada not in texto

    traza = json.loads(_lineas(archivo)[0])
    for clave in sensibles:
        assert clave not in traza
    # El saneado es recursivo: los argumentos del paso siguen ahí, sin la clave.
    assert traza["pasos"][0]["argumentos"] == {"id": "CASO-0001"}
    assert traza["pasos"][0]["herramienta"] == "consultar_caso"


def test_saneado_redacta_credenciales_sin_borrar_los_contadores_de_tokens(tmp_path):
    """S10: `token`, `access_token`, `bearer` y `authorization` no se escriben.

    La comparación por subcadena tiene trampa: `token` es prefijo de
    `tokens_total`, que el resumen de la traza necesita. Los contadores de
    tokens se eximen por nombre exacto y sobreviven con su valor.
    """
    credenciales = {
        "token": "VALOR_TOKEN",
        "access_token": "VALOR_ACCESS_TOKEN",
        "bearer": "VALOR_BEARER",
        "authorization": "VALOR_AUTHORIZATION",
    }
    turno = _turno(
        tokens=120,
        tokens_total=4321,
        **credenciales,
        pasos=[
            {
                "herramienta": "listar_casos",
                "argumentos": {"estado": "abierto", "authorization": "VALOR_ANIDADO"},
                "ok": True,
            }
        ],
    )

    archivo = observabilidad.escribir_traza(turno, ruta=tmp_path)

    assert archivo is not None
    texto = archivo.read_text(encoding="utf-8")
    for filtrada in (*credenciales.values(), "VALOR_ANIDADO"):
        assert filtrada not in texto

    traza = json.loads(_lineas(archivo)[0])
    for clave in credenciales:
        assert clave not in traza
    # Los contadores de tokens son métricas del turno, no credenciales.
    assert traza["tokens"] == 120
    assert traza["tokens_total"] == 4321
    # El saneado es recursivo: cae la credencial anidada y el resto del paso queda.
    assert traza["pasos"][0]["argumentos"] == {"estado": "abierto"}
    assert traza["pasos"][0]["herramienta"] == "listar_casos"


def test_saneado_no_muta_el_turno_del_llamador(tmp_path):
    sensibles = {"api_key": "VALOR_API_KEY"}
    turno = _turno(**sensibles)

    observabilidad.escribir_traza(turno, ruta=tmp_path)

    assert turno["api_key"] == sensibles["api_key"]
    assert "pregunta" in turno


def test_resumen_stats_agrega_consultas_pasos_y_modos(tmp_path):
    observabilidad.escribir_traza(
        _turno(
            modo="reactivo",
            tokens=100,
            pasos=[
                {"herramienta": "buscar_documentacion", "ok": True},
                {"herramienta": "calcular_plazo", "ok": True},
            ],
        ),
        ruta=tmp_path,
    )
    observabilidad.escribir_traza(
        _turno(
            pregunta="¿Cubren pantallas quebradas?",
            abstencion=True,
            modo="reactivo",
            tokens=50,
            error="tope_pasos: se agotó el presupuesto",
            pasos=[{"herramienta": "buscar_documentacion", "ok": False}],
        ),
        ruta=tmp_path,
    )
    observabilidad.escribir_traza(
        _turno(
            pregunta="Agenda una recepción",
            modo="planificado",
            tokens=200,
            pasos=[
                {"herramienta": "crear_caso", "ok": True},
                {"herramienta": "calcular_plazo", "ok": True},
                # Paso sin nombre: cuenta en `pasos`, no en `por_herramienta`.
                {"ok": False},
            ],
        ),
        ruta=tmp_path,
    )

    stats = observabilidad.resumen_stats(ruta=tmp_path)

    assert stats["consultas"] == 3
    assert stats["abstenciones"] == 1
    assert stats["pasos"] == 6
    assert stats["por_herramienta"] == {
        "buscar_documentacion": 2,
        "calcular_plazo": 2,
        "crear_caso": 1,
    }
    assert stats["por_modo"] == {"reactivo": 2, "planificado": 1}
    assert stats["tokens"] == 350
    assert stats["error"] == 1


def test_resumen_stats_sin_trazas_devuelve_ceros(tmp_path):
    stats = observabilidad.resumen_stats(ruta=tmp_path / "inexistente")

    assert stats == {
        "consultas": 0,
        "abstenciones": 0,
        "pasos": 0,
        "por_herramienta": {},
        "por_modo": {},
        "tokens": 0,
        "error": 0,
    }


def test_leer_trazas_ignora_linea_corrupta_intercalada(tmp_path):
    archivo = tmp_path / f"trazas_{date.today().isoformat()}.jsonl"
    archivo.write_text(
        json.dumps({"pregunta": "antes"}, ensure_ascii=False) + "\n"
        + '{"pregunta": "rota", "pasos": [}\n'
        + "no es json\n"
        + json.dumps([1, 2, 3]) + "\n"
        + "\n"
        + json.dumps({"pregunta": "después"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    trazas = observabilidad.leer_trazas(ruta=tmp_path)

    assert [traza["pregunta"] for traza in trazas] == ["antes", "después"]


def test_leer_trazas_respeta_el_limite_quedandose_con_las_ultimas(tmp_path):
    for numero in range(5):
        observabilidad.escribir_traza(_turno(pregunta=f"pregunta {numero}"), ruta=tmp_path)

    trazas = observabilidad.leer_trazas(limite=2, ruta=tmp_path)

    assert [traza["pregunta"] for traza in trazas] == ["pregunta 3", "pregunta 4"]


def test_leer_trazas_sin_directorio_devuelve_lista_vacia(tmp_path):
    assert observabilidad.leer_trazas(ruta=tmp_path / "no_existe") == []
    assert observabilidad.leer_trazas(limite=0, ruta=tmp_path) == []


def test_las_tildes_se_escriben_legibles_no_escapadas(tmp_path):
    archivo = observabilidad.escribir_traza(
        _turno(pregunta="¿Reparan pantallas quebradas en Ñuñoa?"), ruta=tmp_path
    )

    assert archivo is not None
    texto = archivo.read_text(encoding="utf-8")
    assert "Ñuñoa" in texto
    assert "\\u" not in texto


def test_la_traza_incluye_los_pensamientos_del_turno():
    from types import SimpleNamespace

    respuesta = SimpleNamespace(
        pregunta="¿hay casos?",
        respuesta="Sí.",
        pasos=[],
        plan=None,
        replanes=0,
        tokens_total=10,
        latencia_seg=0.2,
        abstencion=False,
        error=None,
        pensamientos=["Conviene listar antes de responder."],
    )

    traza = observabilidad.traza_de_turno(respuesta, modo="reactivo")

    assert traza["pensamientos"] == ["Conviene listar antes de responder."]


def test_un_paso_sin_medicion_se_escribe_como_null_y_no_como_cero():
    """S9/T13: la traza no rellena con 0.0 una duración que no se midió.

    Un 0.0 es indistinguible de una llamada tan rápida que no cruzó el reloj:
    rellenar con él lo no medido es lo que hizo pasar por medida una llamada a
    herramienta que nunca se midió. La medición real se conserva tal cual.
    """
    from types import SimpleNamespace

    respuesta = SimpleNamespace(
        pregunta="¿reparan tablets?",
        respuesta="No hay evidencia suficiente.",
        pasos=[
            SimpleNamespace(herramienta="buscar_documentacion", argumentos={},
                            ok=True, error=None, duracion_seg=0.000482),
            SimpleNamespace(herramienta="capacidades", argumentos={},
                            ok=True, error=None, duracion_seg=None),
        ],
        plan=None,
        replanes=0,
        tokens_total=10,
        latencia_seg=1.2,
        abstencion=True,
        error=None,
        pensamientos=[],
    )

    traza = observabilidad.traza_de_turno(respuesta, modo="reactivo")

    assert traza["pasos"][0]["duracion_seg"] == 0.000482
    assert traza["pasos"][1]["duracion_seg"] is None


def test_saneado_conserva_los_contadores_anidados_y_redacta_en_profundidad(tmp_path):
    """S10: el saneado es recursivo y `tokens`/`tokens_total` sobreviven anidados."""
    turno = _turno(metadatos={
        "token": "VALOR_TOKEN_ANIDADO",
        "access_token": "VALOR_ACCESS_ANIDADO",
        "bearer": "VALOR_BEARER_ANIDADO",
        "authorization": "VALOR_AUTHZ_ANIDADO",
        "tokens": 33,
        "tokens_total": 44,
        "pasos": [{"herramienta": "listar_casos", "tokens": 5}],
    })
    archivo = observabilidad.escribir_traza(turno, ruta=tmp_path)
    texto = archivo.read_text(encoding="utf-8")
    for filtrada in ("VALOR_TOKEN_ANIDADO", "VALOR_ACCESS_ANIDADO",
                     "VALOR_BEARER_ANIDADO", "VALOR_AUTHZ_ANIDADO"):
        assert filtrada not in texto
    assert json.loads(_lineas(archivo)[0])["metadatos"] == {
        "tokens": 33, "tokens_total": 44,
        "pasos": [{"herramienta": "listar_casos", "tokens": 5}],
    }
