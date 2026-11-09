"""Consola con streaming y confirmación por menú (RA2 / IL2.1).

Estos tests fijan cuatro defectos de experiencia que no tocan la red:

  - la síntesis del modo planificado y el bucle reactivo pueden emitir el texto
    por trozos y, aun así, contabilizar tokens (el callback de LangChain no
    recibe nada en un `.stream()` directo, así que el uso se suma a mano);
  - la respuesta se tipografía como Markdown mientras llega, en un solo `Live`,
    y no se imprime dos veces cuando el texto transmitido es el final;
  - la confirmación de escritura es un menú (`src.seleccion`) con la opción
    segura preseleccionada, y nunca se lee con un `Live` abierto;
  - una confirmación en medio del stream corta el `Live` y deja que la
    respuesta final se imprima una sola vez en el panel.

Todo corre con dobles: el modelo expone `.stream()` y `.invoke()` falsos y el
agente se construye sin herramientas reales. La consola se reemplaza por una
instancia con salida a `StringIO`.
"""
import importlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console

from src import config, credenciales
from src.agente import AgenteTecnoApoyo, PasoAgente, RespuestaAgente

RAIZ_PROYECTO = Path(__file__).resolve().parents[1]

# Variables nuevas que resuelve `src.config` y que el `.env` puede definir.
VARIABLES_NUEVAS = ("LLM_TIMEOUT_SEG", "LLM_MAX_REINTENTOS", "LLM_STREAM")


# --- dobles ---------------------------------------------------------------
class ModeloStreamFalso:
    """Doble con `.stream()` y `.invoke()`: registra cómo se le llamó."""

    def __init__(self, trozos, texto_invoke="Respuesta del invoke.",
                 falla_despues=None):
        self.trozos = list(trozos)
        self.texto_invoke = texto_invoke
        self.falla_despues = falla_despues
        self.invocaciones = 0
        self.llamadas_stream = []

    def invoke(self, mensajes, config=None):
        self.invocaciones += 1
        self.mensajes_invoke = list(mensajes)
        return SimpleNamespace(content=self.texto_invoke)

    def stream(self, mensajes, *args, **kwargs):
        self.llamadas_stream.append((list(mensajes), args, kwargs))
        yield from self.trozos
        if self.falla_despues is not None:
            raise self.falla_despues


def agente_sin_modelo():
    """Agente con el modelo inyectado en cada llamada y sin herramientas."""
    return AgenteTecnoApoyo(herramientas=[], llm=None)


def resultado_falso(salida='{"ok": true, "datos": {"casos": []}}'):
    return SimpleNamespace(pasos=[SimpleNamespace(
        herramienta="buscar_documentacion", ok=True, salida=salida, error=None,
    )])


def respuesta_falsa(texto="El caso **CASO-0011** sigue abierto.",
                    tokens=5, pasos=None, error=None):
    return RespuestaAgente(
        pregunta="¿en qué estado está el caso?",
        respuesta=texto,
        abstencion=False,
        pasos=list(pasos or []),
        iteraciones=1,
        tokens_total=tokens,
        latencia_seg=0.3,
        error=error,
    )


# --- síntesis con streaming ----------------------------------------------
def test_sintetizar_con_callback_emite_los_trozos_y_ensambla_el_texto():
    modelo = ModeloStreamFalso([
        SimpleNamespace(content="**CASO-0011**"),
        SimpleNamespace(content=" sigue abierto."),
    ])
    emitidos = []
    agente = agente_sin_modelo()

    texto, error = agente._sintetizar(
        modelo, "¿en qué estado está el caso?", resultado_falso(),
        al_emitir=emitidos.append,
    )

    assert error is None
    assert texto == "**CASO-0011** sigue abierto."
    # Se emite cada trozo con contenido, tal como llega (sin recomponer).
    assert emitidos == ["**CASO-0011**", " sigue abierto."]
    # Un `.stream()` directo no reenvía el callback de LangChain: se comprueba
    # que no se le pasa, porque el uso se suma a mano.
    assert len(modelo.llamadas_stream) == 1
    _, _, kwargs = modelo.llamadas_stream[0]
    assert kwargs == {}
    assert modelo.invocaciones == 0


def test_sintetizar_con_stream_suma_los_tokens_del_ultimo_trozo():
    modelo = ModeloStreamFalso([
        SimpleNamespace(content="El diagnóstico"),
        SimpleNamespace(content=" demora 48 horas."),
        SimpleNamespace(content="", usage_metadata={
            "input_tokens": 11, "output_tokens": 7, "total_tokens": 18,
        }),
    ])
    agente = agente_sin_modelo()

    texto, error = agente._sintetizar(
        modelo, "¿cuánto demora?", resultado_falso(), al_emitir=lambda _: None,
    )

    assert error is None
    assert texto == "El diagnóstico demora 48 horas."
    # El callback no ve nada en un stream directo: si no se sumara a mano, la
    # contabilidad quedaría en 0/0/0 y el reporte de tokens mentiría.
    assert agente.colector_tokens.prompt_tokens == 11
    assert agente.colector_tokens.completion_tokens == 7
    assert agente.colector_tokens.total_tokens == 18


def test_sintetizar_tolera_trozos_sin_usage_metadata():
    modelo = ModeloStreamFalso([
        SimpleNamespace(content="Hola."),
    ])
    agente = agente_sin_modelo()

    texto, error = agente._sintetizar(
        modelo, "pregunta", resultado_falso(), al_emitir=lambda _: None,
    )

    assert (texto, error) == ("Hola.", None)
    assert agente.colector_tokens.total_tokens == 0


def test_sintetizar_sin_callback_usa_invoke_y_no_emite():
    modelo = ModeloStreamFalso(
        [SimpleNamespace(content="no debería leerse")],
        texto_invoke="Respuesta sin streaming.",
    )
    agente = agente_sin_modelo()

    texto, error = agente._sintetizar(modelo, "pregunta", resultado_falso())

    assert (texto, error) == ("Respuesta sin streaming.", None)
    assert modelo.invocaciones == 1
    # La ruta de evaluación (`scripts/evaluar_agente.py`) no toca `.stream()` ni
    # construye emisores: sigue exactamente igual que antes.
    assert modelo.llamadas_stream == []


def test_sintetizar_con_fallo_a_medio_stream_conserva_lo_ya_mostrado():
    modelo = ModeloStreamFalso(
        [SimpleNamespace(content="No tengo información")],
        falla_despues=RuntimeError("conexión cortada"),
    )
    emitidos = []
    agente = agente_sin_modelo()

    texto, error = agente._sintetizar(
        modelo, "pregunta", resultado_falso(), al_emitir=emitidos.append,
    )

    # El texto parcial ya se mostró en pantalla: perderlo dejaría al usuario con
    # una línea incompleta y sin forma de saber qué alcanzó a responder.
    assert texto == "No tengo información"
    assert error is not None and "conexión cortada" in error
    assert emitidos == ["No tengo información"]


# --- configuración --------------------------------------------------------
@pytest.fixture
def recargar_config(monkeypatch):
    """Recarga `src.config` con `load_dotenv` neutralizado y entorno controlado."""
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: False)

    def recargar(**valores):
        for nombre in VARIABLES_NUEVAS:
            monkeypatch.delenv(nombre, raising=False)
        for nombre, valor in valores.items():
            monkeypatch.setenv(nombre, str(valor))
        importlib.reload(config)
        return config

    try:
        yield recargar
    finally:
        # Restaurar el entorno y recargar antes de soltar el monkeypatch: si se
        # recargara con el entorno limpio, el resto de la suite heredaría los
        # defaults y no la configuración real.
        monkeypatch.undo()
        importlib.reload(config)


def test_config_stream_y_tiempos_tienen_defaults_seguros(recargar_config):
    modulo = recargar_config()
    assert modulo.LLM_TIMEOUT_SEG == 60.0
    assert isinstance(modulo.LLM_TIMEOUT_SEG, float)
    assert modulo.LLM_MAX_REINTENTOS == 1
    assert modulo.LLM_STREAM is True


def test_config_lee_los_tres_valores_del_entorno(recargar_config):
    modulo = recargar_config(
        LLM_TIMEOUT_SEG="12.5", LLM_MAX_REINTENTOS="4", LLM_STREAM="0",
    )
    assert modulo.LLM_TIMEOUT_SEG == 12.5
    assert modulo.LLM_MAX_REINTENTOS == 4
    assert modulo.LLM_STREAM is False


@pytest.mark.parametrize("valor", ["0", "false", "False", "NO", "off", " OFF "])
def test_config_desactiva_stream_con_los_valores_falsy(recargar_config, valor):
    assert recargar_config(LLM_STREAM=valor).LLM_STREAM is False


@pytest.mark.parametrize("valor", ["1", "true", "sí", "apagado", ""])
def test_config_mantiene_stream_activo_con_cualquier_otro_valor(recargar_config, valor):
    # Solo 0/false/no/off desactivan: un valor no reconocido no deja la consola
    # muda sin que nadie lo haya pedido.
    assert recargar_config(LLM_STREAM=valor).LLM_STREAM is True


def test_config_nunca_deja_reintentos_negativos(recargar_config):
    assert recargar_config(LLM_MAX_REINTENTOS="-3").LLM_MAX_REINTENTOS == 0


# --- clientes del proveedor ----------------------------------------------
def test_los_clientes_del_motor_llevan_timeout_reintentos_y_stream_usage(
    monkeypatch,
):
    from src import rag

    chats: list[dict] = []
    embeddings: list[dict] = []

    class ChatFalso:
        def __init__(self, **kwargs):
            chats.append(kwargs)

    class EmbeddingsFalso:
        def __init__(self, **kwargs):
            embeddings.append(kwargs)

    monkeypatch.setattr(config, "LLM_API_KEY", "clave-de-prueba")
    monkeypatch.setattr(rag, "ChatOpenAI", ChatFalso)
    monkeypatch.setattr(rag, "OpenAIEmbeddings", EmbeddingsFalso)

    rag.RAGEngine(carga_indice=False)

    assert len(chats) == 1 and len(embeddings) == 1
    assert chats[0]["timeout"] == config.LLM_TIMEOUT_SEG
    assert chats[0]["max_retries"] == config.LLM_MAX_REINTENTOS
    assert chats[0]["stream_usage"] is True
    assert embeddings[0]["timeout"] == config.LLM_TIMEOUT_SEG
    assert embeddings[0]["max_retries"] == config.LLM_MAX_REINTENTOS


def test_los_clientes_del_agente_llevan_timeout_reintentos_y_stream_usage(
    monkeypatch,
):
    import langchain_openai

    registrados: list[dict] = []

    class ChatFalso:
        def __init__(self, **kwargs):
            registrados.append(kwargs)

        def bind_tools(self, herramientas, **kwargs):
            return self

    monkeypatch.setattr(config, "LLM_API_KEY", "clave-de-prueba")
    monkeypatch.setattr(langchain_openai, "ChatOpenAI", ChatFalso)

    agente = AgenteTecnoApoyo(herramientas=[], llm=None, modo="planificado")
    agente._modelo()
    agente._modelo_de_planificacion()

    assert len(registrados) == 2
    for kwargs in registrados:
        assert kwargs["timeout"] == config.LLM_TIMEOUT_SEG
        assert kwargs["max_retries"] == config.LLM_MAX_REINTENTOS
        assert kwargs["stream_usage"] is True


# --- arranque: importar no arranca ---------------------------------------
@pytest.fixture(scope="module")
def app_mod():
    """`src.app` se importa sin efectos: la clave y el motor se piden en `main()`."""
    import src.app

    return src.app


def test_importar_la_app_no_ejecuta_el_arranque():
    """Importar la consola no puede exigir clave ni motor.

    La guarda de credenciales y la construcción del motor viven en `main()`:
    un checkout limpio, sin clave, tiene que poder importar el módulo.
    """
    # El hijo importa numpy/faiss: con todos los núcleos OpenBLAS puede fallar
    # antes del import bajo la huella de la suite, como en test_credenciales.
    entorno = {
        **os.environ,
        "LLM_API_KEY": "",
        "OPENBLAS_NUM_THREADS": "1",
    }

    proceso = subprocess.run(
        [sys.executable, "-c", "import src.app"],
        cwd=RAIZ_PROYECTO,
        env=entorno,
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert proceso.returncode == 0, proceso.stderr
    assert "Traceback" not in proceso.stderr


def test_preparar_entorno_sin_clave_instruye_como_configurarla(app_mod, monkeypatch):
    monkeypatch.setattr(app_mod, "engine", None)
    monkeypatch.setattr(app_mod, "memoria", None)
    monkeypatch.setattr(app_mod.config, "LLM_API_KEY", "")

    with pytest.raises(SystemExit) as error:
        app_mod.preparar_entorno()

    assert str(error.value) == credenciales.MENSAJE_FALTA_CLAVE


def test_preparar_entorno_sin_indice_instruye_como_construirlo(
    app_mod, monkeypatch, tmp_path
):
    monkeypatch.setattr(app_mod, "engine", None)
    monkeypatch.setattr(app_mod, "memoria", None)
    monkeypatch.setattr(app_mod.config, "LLM_API_KEY", "clave-de-prueba")
    monkeypatch.setattr(app_mod.config, "RUTA_INDICE", tmp_path / "sin_indice")

    with pytest.raises(SystemExit) as error:
        app_mod.preparar_entorno()

    mensaje = str(error.value)
    assert "No existe el índice FAISS" in mensaje
    assert "scripts/build_index.py" in mensaje


def consola_capturada(monkeypatch, app_mod, **kwargs):
    salida = io.StringIO()
    parametros = {"file": salida, "width": 100, "force_terminal": False}
    parametros.update(kwargs)
    consola = Console(**parametros)
    monkeypatch.setattr(app_mod, "console", consola)
    return salida, consola


def elegir_espia(monkeypatch, app_mod, valor="si"):
    """Sustituye el menú de selección y registra cómo se le llamó.

    También registra si había un `Live` de streaming abierto: la confirmación
    se lee del teclado y Rich Live no admite entrada mientras redibuja.
    """
    registro: dict = {"llamado": False}

    def elegir(consola, pregunta, opciones, indice_inicial=0,
               lector_teclas=None):
        registro["llamado"] = True
        registro["consola"] = consola
        registro["pregunta"] = pregunta
        registro["opciones"] = tuple(opciones)
        registro["indice_inicial"] = indice_inicial
        registro["lector_teclas"] = lector_teclas
        transmision = getattr(app_mod, "_transmision_actual", None)
        registro["live_activo"] = bool(
            transmision is not None and transmision.activo
        )
        for opcion in registro["opciones"]:
            if opcion.valor == valor:
                return opcion
        raise AssertionError(f"El menú no ofrece la opción {valor!r}")

    monkeypatch.setattr(app_mod.seleccion, "elegir", elegir)
    return registro


def test_la_confirmacion_ofrece_un_menu_con_el_default_seguro(app_mod, monkeypatch):
    _, consola = consola_capturada(monkeypatch, app_mod)
    registro = elegir_espia(monkeypatch, app_mod, "si")

    assert app_mod._confirmar_en_consola(
        "crear_caso", {"cliente": "ACME"}
    ) is True

    assert registro["llamado"] is True
    assert registro["consola"] is consola
    assert [(o.etiqueta, o.valor) for o in registro["opciones"]] == [
        ("Sí, escribir", "si"), ("No, cancelar", "no"),
    ]
    # El cursor arranca en la opción segura: un Enter accidental no escribe.
    seguro = registro["opciones"][registro["indice_inicial"]]
    assert seguro.valor == "no"
    assert seguro.predeterminada is True


def test_la_confirmacion_solo_aprueba_con_si(app_mod, monkeypatch):
    consola_capturada(monkeypatch, app_mod)
    elegir_espia(monkeypatch, app_mod, "no")

    assert app_mod._confirmar_en_consola(
        "crear_caso", {"cliente": "ACME"}
    ) is False


def test_la_confirmacion_conserva_el_panel_de_escritura_propuesta(app_mod, monkeypatch):
    salida, _ = consola_capturada(monkeypatch, app_mod)
    elegir_espia(monkeypatch, app_mod, "si")

    app_mod._confirmar_en_consola(
        "crear_caso", {"cliente": "ACME", "prioridad": "media"}
    )

    texto = salida.getvalue()
    assert "Escritura propuesta" in texto
    assert "crear_caso" in texto
    assert "ACME" in texto and "media" in texto


def test_el_menu_real_aprueba_con_flecha_y_enter(app_mod, monkeypatch):
    salida, _ = consola_capturada(monkeypatch, app_mod)
    teclas = ["arriba", "enter"]
    monkeypatch.setattr(app_mod.seleccion, "leer_tecla", lambda: teclas.pop(0))

    assert app_mod._confirmar_en_consola(
        "crear_caso", {"cliente": "ACME"}
    ) is True
    assert "Sí, escribir" in salida.getvalue()


def test_el_menu_real_niega_con_enter_a_secas(app_mod, monkeypatch):
    consola_capturada(monkeypatch, app_mod)
    teclas = ["enter"]  # Enter en el arranque cae en la opción segura.
    monkeypatch.setattr(app_mod.seleccion, "leer_tecla", lambda: teclas.pop(0))

    assert app_mod._confirmar_en_consola(
        "crear_caso", {"cliente": "ACME"}
    ) is False


# --- render del panel -----------------------------------------------------
def test_el_panel_final_renderiza_markdown(app_mod, monkeypatch):
    salida, _ = consola_capturada(monkeypatch, app_mod)
    monkeypatch.setattr(app_mod, "_activo_trazo", False)

    app_mod.mostrar_respuesta_agente(respuesta_falsa(
        "El caso **CASO-0011** sigue abierto [NO DISPONIBLE]."
    ))

    texto = salida.getvalue()
    # El Markdown se renderiza: ni los asteriscos literales ni el corchete se
    # interpretan como marcado de Rich.
    assert "**" not in texto
    assert "CASO-0011" in texto
    assert "[NO DISPONIBLE]" in texto


def test_el_panel_final_aplica_negrita_de_markdown(app_mod, monkeypatch):
    salida, _ = consola_capturada(
        monkeypatch, app_mod, force_terminal=True, color_system="standard",
    )
    monkeypatch.setattr(app_mod, "_activo_trazo", False)

    app_mod.mostrar_respuesta_agente(respuesta_falsa("El caso **CASO-0011**."))

    assert "\x1b[1m" in salida.getvalue()


def test_el_panel_de_error_renderiza_markdown(app_mod, monkeypatch):
    salida, _ = consola_capturada(monkeypatch, app_mod)
    monkeypatch.setattr(app_mod, "_activo_trazo", False)

    app_mod.mostrar_respuesta_agente(respuesta_falsa(
        "", error="TimeoutError: [ERROR] **sin respuesta** del proveedor.",
    ))

    texto = salida.getvalue()
    # El panel del aviso se renderiza como Markdown: no queda ningún asterisco
    # de marcado alrededor del texto.
    assert "(sin respuesta)" in texto
    # El mensaje del proveedor se imprime escapado: un `[...]` no se interpreta
    # como etiqueta de Rich (no lanza MarkupError) y se ve tal cual.
    assert "[ERROR]" in texto
    assert "TimeoutError" in texto
    assert "del proveedor." in texto


def test_la_respuesta_ya_transmitida_no_se_reimprime(app_mod, monkeypatch):
    salida, _ = consola_capturada(monkeypatch, app_mod)
    monkeypatch.setattr(app_mod, "_activo_trazo", False)

    app_mod.mostrar_respuesta_agente(
        respuesta_falsa("Texto ya transmitido por la consola.", tokens=9),
        ya_mostrada=True,
    )

    texto = salida.getvalue()
    assert "Texto ya transmitido por la consola." not in texto
    # El pie con métricas sigue apareciendo: es el resumen del turno.
    assert "tokens: 9" in texto


def test_la_traza_de_fuentes_sigue_visible_cuando_hay_streaming(app_mod, monkeypatch):
    salida, _ = consola_capturada(monkeypatch, app_mod)
    monkeypatch.setattr(app_mod, "_activo_trazo", True)
    paso = PasoAgente(
        numero=1, herramienta="buscar_documentacion", argumentos={},
        salida=json.dumps({"ok": True, "datos": {"fragmentos": [{
            "document_id": "PROC-004",
            "titulo": "Política de presupuestos y diagnósticos",
            "score": 0.88,
        }]}}),
        ok=True,
    )

    app_mod.mostrar_respuesta_agente(
        respuesta_falsa("Ya mostrado.", pasos=[paso]), ya_mostrada=True,
    )

    # La tabla de fuentes se imprime aunque el texto ya se haya transmitido.
    assert "PROC-004" in salida.getvalue()


# --- main: sin console.status, con pasos y streaming ---------------------
class ConsolaEspia(Console):
    """Console que cuenta las llamadas a `status` (Rich Live)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.status_llamados = 0

    def status(self, *args, **kwargs):
        self.status_llamados += 1
        return super().status(*args, **kwargs)


class AgenteEspia:
    """Agente falso que ejecuta un guion de pasos y trozos, sin red."""

    def __init__(self, respuesta, pasos=(), trozos=(), antes_de_trozo=None,
                 al_consultar=None):
        self.respuesta = respuesta
        self.pasos = list(pasos)
        self.trozos = list(trozos)
        # Gancho para intercalar una confirmación de escritura entre dos trozos.
        self.antes_de_trozo = antes_de_trozo
        # Gancho para fotografiar el estado del módulo al empezar la consulta.
        self.al_consultar = al_consultar
        self.consultas = []
        self.al_emitir_recibido = "sin definir"
        self.al_pensamiento_recibido = "sin definir"
        self.transmision_al_consultar = "sin definir"

    def consultar(self, pregunta, al_paso=None, al_emitir=None,
                  al_pensamiento=None):
        self.consultas.append(pregunta)
        self.al_emitir_recibido = al_emitir
        self.al_pensamiento_recibido = al_pensamiento
        if self.al_consultar is not None:
            self.transmision_al_consultar = self.al_consultar()
        for paso in self.pasos:
            if al_paso is not None:
                al_paso(paso)
        for indice, trozo in enumerate(self.trozos):
            if self.antes_de_trozo is not None:
                self.antes_de_trozo(indice, self)
            if al_emitir is not None:
                al_emitir(trozo)
        return self.respuesta


def preparar_main(monkeypatch, app_mod, agente, entradas, **opciones_consola):
    salida = io.StringIO()
    parametros = {"file": salida, "width": 100, "force_terminal": False}
    parametros.update(opciones_consola)
    consola = ConsolaEspia(**parametros)
    monkeypatch.setattr(app_mod, "console", consola)
    pendientes = iter(list(entradas))

    def entrada(mensaje=""):
        try:
            return next(pendientes)
        except StopIteration:
            raise EOFError from None

    monkeypatch.setattr(consola, "input", entrada)
    monkeypatch.setattr(app_mod, "obtener_agente", lambda: agente)
    monkeypatch.setattr(app_mod, "engine", SimpleNamespace(historial=[]))
    monkeypatch.setattr(app_mod, "memoria", SimpleNamespace(
        largo=SimpleNamespace(hechos_relevantes=lambda pregunta: []),
    ))
    monkeypatch.setattr(app_mod, "escribir_traza", lambda *args, **kwargs: None)
    monkeypatch.setattr(app_mod, "traza_de_turno", lambda *args, **kwargs: None)
    return salida, consola


def test_la_consulta_ya_no_se_envuelve_en_console_status(app_mod, monkeypatch):
    monkeypatch.setattr(app_mod.config, "LLM_STREAM", True)
    respuesta = respuesta_falsa("El diagnóstico cuesta $10.000.")
    agente = AgenteEspia(respuesta, pasos=[PasoAgente(
        numero=1, herramienta="buscar_documentacion", argumentos={},
        salida="{}", ok=True,
    )])
    salida, consola = preparar_main(
        monkeypatch, app_mod, agente, ["¿cuánto cuesta?", "/salir"],
    )

    app_mod.main()

    # Rich Live no soporta entrada: la confirmación de escritura quedaba
    # enterrada bajo el spinner. La consulta ya no abre un `status`.
    assert consola.status_llamados == 0
    assert agente.consultas == ["¿cuánto cuesta?"]
    texto = salida.getvalue()
    assert "Consultando" in texto
    assert "→ buscar_documentacion | ok=sí" in texto


def test_el_texto_del_modelo_se_transmite_una_sola_vez(app_mod, monkeypatch):
    monkeypatch.setattr(app_mod.config, "LLM_STREAM", True)
    monkeypatch.setattr(app_mod, "_activo_trazo", False)
    respuesta = respuesta_falsa("Texto final del modelo.")
    agente = AgenteEspia(respuesta, trozos=["Texto final ", "del modelo."])
    salida, _ = preparar_main(
        monkeypatch, app_mod, agente, ["consulta", "/salir"],
    )

    app_mod.main()

    texto = salida.getvalue()
    # El emisor llegó al agente y el panel no repitió el mismo texto.
    assert callable(agente.al_emitir_recibido)
    assert texto.count("Texto final del modelo.") == 1
    # Los trozos se escriben sin interpretar marcado: así un `[...]` del modelo
    # no se convierte en una etiqueta de Rich.
    assert "Texto final del modelo." in texto


def test_con_stream_desactivado_el_panel_sigue_mostrando_la_respuesta(app_mod, monkeypatch):
    monkeypatch.setattr(app_mod.config, "LLM_STREAM", False)
    monkeypatch.setattr(app_mod, "_activo_trazo", False)
    respuesta = respuesta_falsa("El caso **CASO-0011** sigue abierto.")
    agente = AgenteEspia(respuesta, al_consultar=lambda: app_mod._transmision_actual)
    salida, _ = preparar_main(
        monkeypatch, app_mod, agente, ["consulta", "/salir"],
    )

    app_mod.main()

    # Sin streaming se mantiene el panel único, ya renderizado como Markdown, y
    # no se abre ningún `Live`.
    assert agente.al_emitir_recibido is None
    assert agente.transmision_al_consultar is None
    texto = salida.getvalue()
    assert "**" not in texto
    assert "CASO-0011" in texto
    # Un solo panel: el texto de la respuesta aparece una única vez.
    assert texto.count("El caso CASO-0011 sigue abierto.") == 1
    assert "\u2500 Asistente \u2500" in texto


def test_si_el_texto_final_difiere_del_transmitido_el_panel_se_imprime(app_mod, monkeypatch):
    # El guard de abstención del agente puede reemplazar la síntesis ya
    # transmitida. Omitir el panel dejaría al usuario con el texto del modelo y
    # sin la respuesta final, así que solo se omite cuando el texto coincide.
    monkeypatch.setattr(app_mod.config, "LLM_STREAM", True)
    monkeypatch.setattr(app_mod, "_activo_trazo", False)
    respuesta = respuesta_falsa(
        "No tengo información suficiente en la documentación de TecnoApoyo "
        "para responder eso."
    )
    agente = AgenteEspia(respuesta, trozos=["Sí, reparamos tablets."])
    salida, _ = preparar_main(
        monkeypatch, app_mod, agente, ["consulta", "/salir"],
    )

    app_mod.main()

    texto = salida.getvalue()
    assert "Sí, reparamos tablets." in texto
    assert "No tengo información suficiente" in texto


# --- Markdown mientras llega: un solo `Live` ------------------------------
def test_el_stream_tipografia_markdown_mientras_llega(app_mod, monkeypatch):
    monkeypatch.setattr(app_mod.config, "LLM_STREAM", True)
    monkeypatch.setattr(app_mod, "_activo_trazo", False)
    respuesta = respuesta_falsa("**hola** de nuevo.")
    # El corte va fuera del marcador: un `**` partido entre dos trozos solo es
    # texto incompleto para el renderizador, no un fallo del `Live`.
    agente = AgenteEspia(respuesta, trozos=["**hola**", " de nuevo."])
    salida, _ = preparar_main(
        monkeypatch, app_mod, agente, ["consulta", "/salir"],
        force_terminal=True, color_system="standard",
    )

    app_mod.main()

    texto = salida.getvalue()
    # El texto se tipografía mientras llega: nunca se ven los asteriscos del
    # modelo acumulados en pantalla.
    assert "**" not in texto
    assert "\x1b[1m" in texto


def test_el_stream_no_repite_el_texto_en_el_panel(app_mod, monkeypatch):
    # El texto ya quedó renderizado por el `Live`: el panel final no lo repite.
    monkeypatch.setattr(app_mod.config, "LLM_STREAM", True)
    monkeypatch.setattr(app_mod, "_activo_trazo", False)
    respuesta = respuesta_falsa("El caso **CASO-0011** está abierto.")
    agente = AgenteEspia(
        respuesta, trozos=["El caso **CASO-0011**", " está abierto."],
    )
    salida, _ = preparar_main(
        monkeypatch, app_mod, agente, ["consulta", "/salir"],
    )

    app_mod.main()

    texto = salida.getvalue()
    assert "**" not in texto
    assert texto.count("El caso CASO-0011 está abierto.") == 1
    # El panel del asistente no aparece: ya había un bloque renderizado.
    assert "\u2500 Asistente \u2500" not in texto


def test_el_stream_deja_el_bloque_final_en_pantalla(app_mod, monkeypatch):
    # `transient=False`: al cerrar el `Live`, lo renderizado queda visible en vez
    # de borrarse como un spinner.
    monkeypatch.setattr(app_mod.config, "LLM_STREAM", True)
    monkeypatch.setattr(app_mod, "_activo_trazo", False)
    respuesta = respuesta_falsa("El horario es de 9 a 18.")
    agente = AgenteEspia(respuesta, trozos=["El horario ", "es de 9 a 18."])
    salida, _ = preparar_main(
        monkeypatch, app_mod, agente, ["consulta", "/salir"],
    )

    app_mod.main()

    assert salida.getvalue().count("El horario es de 9 a 18.") == 1


# --- confirmación de escritura en medio del stream ------------------------
def test_la_confirmacion_detiene_el_live_antes_de_leer(app_mod, monkeypatch):
    monkeypatch.setattr(app_mod.config, "LLM_STREAM", True)
    monkeypatch.setattr(app_mod, "_activo_trazo", False)
    registro = elegir_espia(monkeypatch, app_mod, "si")
    respuesta = respuesta_falsa("Voy a crear el caso. Caso **CASO-0011** creado.")
    aprobaciones: list = []

    def confirmar(indice, _agente):
        if indice == 1:
            aprobaciones.append(app_mod._confirmar_en_consola(
                "crear_caso", {"cliente": "ACME"},
            ))

    agente = AgenteEspia(
        respuesta,
        trozos=["Voy a crear el caso. ", "Caso **CASO-0011** creado."],
        antes_de_trozo=confirmar,
    )
    salida, _ = preparar_main(
        monkeypatch, app_mod, agente, ["consulta", "/salir"],
    )

    app_mod.main()

    # Ningún `Live` puede estar redibujando mientras se lee una confirmación:
    # Rich no admite entrada en ese estado y la respuesta se perdía.
    assert registro["llamado"] is True
    assert registro["live_activo"] is False
    assert aprobaciones == [True]
    texto = salida.getvalue()
    # Con texto ya en pantalla el `Live` no se reabre: lo pintado se borra y el
    # turno completo se imprime una sola vez, como Markdown, en el panel final.
    assert "**" not in texto
    assert texto.count("Caso CASO-0011 creado.") == 1
    assert texto.count("Voy a crear el caso.") == 1
    assert "Escritura propuesta" in texto


def test_si_el_live_estaba_vacio_se_reanuda_tras_la_confirmacion(app_mod, monkeypatch):
    monkeypatch.setattr(app_mod.config, "LLM_STREAM", True)
    monkeypatch.setattr(app_mod, "_activo_trazo", False)
    registro = elegir_espia(monkeypatch, app_mod, "si")
    respuesta = respuesta_falsa("Caso CASO-0011 creado.")
    transmisiones: list = []

    def confirmar(indice, _agente):
        if indice == 1:
            # Se fotografía el estado en el momento de la confirmación: el
            # `Live` del turno se cierra igual al terminar la consulta.
            transmisiones.append((app_mod._transmision_actual.activo,
                                  app_mod._transmision_actual.interrumpida,
                                  app_mod._transmision_actual.texto()))
            app_mod._confirmar_en_consola("crear_caso", {"cliente": "ACME"})
            # Tras la confirmación el `Live` vuelve a estar abierto: el turno
            # sigue transmitiéndose y no se pierde el resto de la respuesta.
            transmisiones.append((app_mod._transmision_actual.activo,
                                  app_mod._transmision_actual.interrumpida,
                                  app_mod._transmision_actual.texto()))

    agente = AgenteEspia(
        respuesta,
        # El primer trozo es solo un salto de línea: el `Live` existe, pero no
        # hay texto pintado que no se pueda rehacer.
        trozos=["\n\n", "Caso CASO-0011 creado."],
        antes_de_trozo=confirmar,
    )
    salida, _ = preparar_main(
        monkeypatch, app_mod, agente, ["consulta", "/salir"],
    )

    app_mod.main()

    assert registro["live_activo"] is False
    # Antes de la confirmación el `Live` estaba abierto con el salto de línea;
    # el menú lo encontró cerrado y lo reabrió al volver.
    assert transmisiones[0][0] is True
    assert transmisiones[1][0] is True
    assert transmisiones[1][1] is False
    texto = salida.getvalue()
    # El texto se transmitió una sola vez y el panel no lo repitió.
    assert texto.count("Caso CASO-0011 creado.") == 1
    # El panel del asistente no aparece: el `Live` quedó con el bloque final.
    assert "\u2500 Asistente \u2500" not in texto



# --- cliente de la sesión (/cliente) --------------------------------------
def preparar_consola(monkeypatch, app_mod, entradas, **opciones_consola):
    """Como `preparar_main`, pero dejando en pie `obtener_agente`.

    El comando `/cliente` tiene que poder reconstruir el agente de verdad, así
    que aquí no se sustituye: solo se reemplazan las dependencias caras.
    """
    salida = io.StringIO()
    parametros = {"file": salida, "width": 100, "force_terminal": False}
    parametros.update(opciones_consola)
    consola = ConsolaEspia(**parametros)
    monkeypatch.setattr(app_mod, "console", consola)
    pendientes = iter(list(entradas))

    def entrada(mensaje=""):
        try:
            return next(pendientes)
        except StopIteration:
            raise EOFError from None

    monkeypatch.setattr(consola, "input", entrada)
    monkeypatch.setattr(app_mod, "engine", SimpleNamespace(historial=[]))
    monkeypatch.setattr(app_mod, "memoria", SimpleNamespace(
        largo=SimpleNamespace(hechos_relevantes=lambda pregunta: []),
    ))
    monkeypatch.setattr(app_mod, "escribir_traza", lambda *args, **kwargs: None)
    monkeypatch.setattr(app_mod, "traza_de_turno", lambda *args, **kwargs: None)
    # El agente es estado de módulo: cada test arranca sin agente armado y sin
    # cliente heredado de otro test.
    monkeypatch.setattr(app_mod, "_agente", None)
    monkeypatch.setattr(app_mod, "_agente_modo", None)
    monkeypatch.setattr(app_mod.config, "CLIENTE_ACTUAL", None, raising=False)
    return salida, consola


def test_el_comando_cliente_muestra_define_y_borra_el_cliente(app_mod, monkeypatch):
    monkeypatch.setattr(app_mod.config, "CLIENTE_ACTUAL", None, raising=False)
    salida, _ = preparar_consola(
        monkeypatch, app_mod,
        ["/cliente", "/cliente Colegio San Ramón", "/cliente",
         "/cliente borrar", "/cliente", "/salir"],
    )

    app_mod.main()

    texto = salida.getvalue()
    # Antes y después de borrar, `/cliente` a secas informa que no hay cliente.
    assert texto.count("Cliente de la sesión: sin identificar") == 2
    assert "Colegio San Ramón" in texto
    assert getattr(app_mod.config, "CLIENTE_ACTUAL", None) is None


def test_cliente_advierte_que_no_autentica_ni_aisla_memoria(app_mod, monkeypatch):
    salida, _ = preparar_consola(
        monkeypatch, app_mod, ["/cliente Valentina Soto", "/salir"]
    )

    app_mod.main()

    texto = salida.getvalue()
    assert "no verifica identidad" in texto
    assert "memoria" in texto and "compartid" in texto


def test_cambiar_el_cliente_vuelve_a_acotar_las_herramientas(app_mod, monkeypatch):
    construcciones: list = []

    def fabrica(repositorio, motor, confirmador=None, **kwargs):
        # El cliente viaja como lector en vivo, no como valor congelado: se
        # evalúa al construir para comprobar qué veía cada agente.
        kwargs["cliente_vivo"] = kwargs["cliente_actual"]() or None
        construcciones.append(kwargs)
        return []

    monkeypatch.setattr(app_mod, "crear_herramientas", fabrica)
    monkeypatch.setattr(app_mod, "RepositorioCasos", lambda: SimpleNamespace())
    agente = AgenteEspia(respuesta_falsa("ok"))
    monkeypatch.setattr(app_mod, "AgenteTecnoApoyo", lambda **kwargs: agente)
    salida, _ = preparar_consola(
        monkeypatch, app_mod,
        ["consulta uno", "/cliente Valentina Soto", "consulta dos",
         "/cliente borrar", "consulta tres", "/salir"],
    )

    app_mod.main()

    # Cada cambio de cliente reconstruye las tools con el alcance de sesión y
    # un lector en vivo: la tool no puede quedarse con el cliente anterior
    # grabado, e `identificar_cliente` escribe sobre la misma sesión.
    assert [c["cliente_vivo"] for c in construcciones] == [
        None, "Valentina Soto", None,
    ]
    assert all(c["alcance"] == "sesion" for c in construcciones)
    assert all(c["al_identificar"] is app_mod.establecer_cliente
               for c in construcciones)
    assert agente.consultas == ["consulta uno", "consulta dos", "consulta tres"]


def test_la_ayuda_lista_el_comando_cliente(app_mod, monkeypatch):
    agente = AgenteEspia(respuesta_falsa("ok"))
    salida, _ = preparar_main(monkeypatch, app_mod, agente, ["/ayuda", "/salir"])

    app_mod.main()

    assert "/cliente" in salida.getvalue()


def test_la_confirmacion_muestra_una_raya_para_los_valores_vacios(app_mod, monkeypatch):
    salida, _ = consola_capturada(monkeypatch, app_mod)
    elegir_espia(monkeypatch, app_mod, "si")

    app_mod._confirmar_en_consola(
        "crear_caso", {"cliente": "ACME", "procedimiento": None},
    )

    texto = salida.getvalue()
    # Un campo vacío se lee como raya, no como el `None` de Python.
    assert "procedimiento: —" in texto
    assert "None" not in texto
    # Los valores reales se siguen mostrando tal cual.
    assert "cliente: ACME" in texto
