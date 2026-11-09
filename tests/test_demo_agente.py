"""Demostración de resolución del agente (scripts/demo_agente.py), sin red.

Estos tests son offline: no hay proveedor, ni índice FAISS, ni base de datos
real. La construcción del agente se inyecta (un constructor falso que devuelve
las piezas) y el camino real se ejerce con el mismo doble guionado de
`tests/test_agente.py`, de modo que lo que se verifica es el cableado de la
demo —rutas temporales, confirmador, lectura de vuelta del repositorio y
código de salida— y no la calidad de una respuesta del modelo.
"""
import re
import tempfile
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from scripts import demo_agente
from src import config
from src.agente import PasoAgente, RespuestaAgente
from src.casos import RepositorioCasos

CLIENTE = "Camila Rojas"


# --- dobles ---------------------------------------------------------------
class AgenteFalso:
    """Agente con un guion fijo de pasos: no toca red ni herramientas."""

    def __init__(self, pasos, respuesta="Resolución simulada.", error=None):
        self.pasos = list(pasos)
        self.respuesta = respuesta
        self.error = error
        self.consultas = []

    def consultar(self, pregunta):
        self.consultas.append(pregunta)
        return RespuestaAgente(
            pregunta=pregunta,
            respuesta=self.respuesta,
            abstencion=False,
            pasos=self.pasos,
            iteraciones=len(self.pasos) + 1,
            tokens_total=0,
            latencia_seg=0.0,
            error=self.error,
        )


class MotorFalso:
    """Doble del motor RAG: la demo solo necesita `recuperar`."""

    def recuperar(self, pregunta, k=None):
        return []


class ModeloFalso:
    """Doble de ChatOpenAI: guion de mensajes, sin proveedor."""

    def __init__(self, guion):
        self.guion = list(guion)
        self.herramientas_recibidas = []

    def bind_tools(self, herramientas, **kwargs):
        self.herramientas_recibidas = list(herramientas)
        return self

    def invoke(self, mensajes, config=None):
        if not self.guion:
            raise AssertionError("El guion del modelo falso se agotó")
        return self.guion.pop(0)


def paso(numero, herramienta, ok=True, error=None):
    return PasoAgente(
        numero=numero,
        herramienta=herramienta,
        argumentos={},
        ok=ok,
        error=error,
    )


def llamada(nombre, argumentos, id_llamada):
    return {
        "name": nombre,
        "args": argumentos,
        "id": id_llamada,
        "type": "tool_call",
    }


def mensaje_tool(*llamadas):
    return AIMessage(content="", tool_calls=list(llamadas))


def mensaje_final(texto):
    return AIMessage(content=texto)


def constructor_falso(repositorio, agente, capturado):
    """Constructor inyectable: registra las rutas y devuelve piezas armadas."""

    def construir(**kwargs):
        capturado.update(kwargs)
        return demo_agente.PiezasDemo(
            agente=agente,
            repositorio=repositorio,
            memoria=None,
            motor=None,
            ruta_casos=Path(kwargs["ruta_casos"]),
            ruta_memoria=Path(kwargs["ruta_memoria"]),
            ruta_trazas=Path(kwargs["ruta_trazas"]),
        )

    return construir


def repo_con_caso_resuelto(ruta):
    """Repositorio temporal cuyo último caso es el que la demo debe verificar."""
    repositorio = RepositorioCasos(ruta=ruta)
    caso = repositorio.crear_caso(
        cliente=CLIENTE,
        categoria="recepcion",
        descripcion="Notebook que no enciende.",
    )
    repositorio.agregar_nota(
        caso["id"], "El equipo no enciende ni con el cargador conectado."
    )
    repositorio.escalar(caso["id"], "Requiere soporte de nivel 2.")
    return repositorio, caso["id"]


# --- aislamiento de estado ------------------------------------------------
def test_la_demo_arma_el_repositorio_fuera_de_data_output(tmp_path):
    repositorio = RepositorioCasos(ruta=tmp_path / "sin_uso.db")
    capturado = {}
    agente = AgenteFalso([paso(1, "crear_caso")])

    codigo = demo_agente.main(
        [],
        constructor=constructor_falso(repositorio, agente, capturado),
        confirmador=lambda accion, resumen: True,
    )

    # Sin caso verificado la demo falla, pero las rutas ya quedaron capturadas.
    assert codigo == 1
    rutas = [capturado["ruta_casos"], capturado["ruta_memoria"],
             capturado["ruta_trazas"]]
    base_temporal = Path(tempfile.gettempdir()).resolve()
    for ruta in rutas:
        resuelta = Path(ruta).resolve()
        assert base_temporal in resuelta.parents, f"{resuelta} no está en temp"
        assert config.DIR_OUTPUT.resolve() not in resuelta.parents
        assert resuelta != config.RUTA_CASOS.resolve()
    # Las tres piezas comparten el mismo directorio temporal de la corrida.
    assert len({Path(ruta).parent for ruta in rutas[:2]}) == 1


# --- resolución verificada ------------------------------------------------
def test_main_devuelve_0_cuando_el_repositorio_confirma_la_resolucion(
    tmp_path, capsys
):
    repositorio, id_caso = repo_con_caso_resuelto(tmp_path / "casos.db")
    capturado = {}
    agente = AgenteFalso([
        paso(1, "crear_caso"),
        paso(2, "agregar_nota"),
        paso(3, "escalar_a_humano"),
    ])

    codigo = demo_agente.main(
        [],
        constructor=constructor_falso(repositorio, agente, capturado),
        confirmador=lambda accion, resumen: True,
    )

    salida = capsys.readouterr().out
    assert codigo == 0
    assert "crear_caso" in salida
    assert "agregar_nota" in salida
    assert "escalar_a_humano" in salida
    assert "ok=sí" in salida
    assert "Resolución simulada." in salida
    assert CLIENTE in salida
    assert id_caso in salida
    assert "escalado" in salida
    assert "Casos antes: 11" in salida
    assert "Casos después: 11" in salida
    assert "VERIFICADA" in salida


def test_main_devuelve_1_y_explica_si_no_hay_caso_resuelto(tmp_path, capsys):
    repositorio = RepositorioCasos(ruta=tmp_path / "casos.db")
    capturado = {}
    agente = AgenteFalso(
        [paso(1, "crear_caso", ok=False, error="Sin confirmación.")],
        respuesta="No pude escribir nada.",
    )

    codigo = demo_agente.main(
        [],
        constructor=constructor_falso(repositorio, agente, capturado),
        confirmador=lambda accion, resumen: False,
    )

    salida = capsys.readouterr().out
    assert codigo == 1
    assert "ok=no" in salida
    assert "Sin confirmación." in salida
    assert "No se pudo verificar la resolución" in salida
    assert CLIENTE in salida
    assert "escalado" in salida


@pytest.mark.parametrize("modo", ["planificado", "reactivo"])
def test_main_usa_el_modo_configurado_si_no_se_pasa_flag(
    tmp_path, monkeypatch, modo
):
    monkeypatch.setattr(config, "AGENTE_MODO", modo)
    repositorio = RepositorioCasos(ruta=tmp_path / "casos.db")
    capturado = {}
    agente = AgenteFalso([])

    demo_agente.main(
        [], constructor=constructor_falso(repositorio, agente, capturado),
        confirmador=lambda accion, resumen: False,
    )

    assert capturado["modo"] == modo


def test_main_reenvia_los_flags_al_constructor(tmp_path, capsys):
    repositorio = RepositorioCasos(ruta=tmp_path / "casos.db")
    capturado = {}
    agente = AgenteFalso([])

    codigo = demo_agente.main(
        ["--modo", "planificado", "--limite-pasos", "3",
         "--pregunta", "Registra algo"],
        constructor=constructor_falso(repositorio, agente, capturado),
        confirmador=lambda accion, resumen: True,
    )

    salida = capsys.readouterr().out
    assert codigo == 1  # el agente falso no escribió nada
    assert capturado["modo"] == "planificado"
    assert capturado["max_pasos"] == 3
    assert agente.consultas == ["Registra algo"]
    assert "consume tokens" in salida


# --- verificación contra el repositorio -----------------------------------
def test_verificar_resolucion_exige_estado_escalado_y_nota(tmp_path):
    repositorio = RepositorioCasos(ruta=tmp_path / "casos.db")
    caso = repositorio.crear_caso(
        cliente=CLIENTE,
        categoria="recepcion",
        descripcion="Notebook que no enciende.",
    )

    # Abierto y sin notas: todavía no hay resolución.
    assert demo_agente.verificar_resolucion(repositorio) is None
    repositorio.agregar_nota(caso["id"], "Nota sin escalar.")
    # Con nota pero sin escalar: sigue sin ser una resolución.
    assert demo_agente.verificar_resolucion(repositorio) is None
    repositorio.escalar(caso["id"], "Requiere soporte de nivel 2.")
    verificado = demo_agente.verificar_resolucion(repositorio)
    assert verificado is not None
    assert verificado["id"] == caso["id"]
    assert verificado["estado"] == "escalado"
    assert verificado["notas"]


# --- confirmador ----------------------------------------------------------
def test_el_confirmador_imprime_la_accion_y_aprueba(capsys):
    confirmador = demo_agente.confirmador_demo()

    aprobado = confirmador("crear_caso", {"cliente": CLIENTE})

    salida = capsys.readouterr().out
    assert aprobado is True
    assert "crear_caso" in salida
    assert CLIENTE in salida
    assert "-> sí" in salida
    assert "automátic" in salida.lower()


# --- construcción real, sin red -------------------------------------------
def test_construir_agente_usa_las_rutas_indicadas_en_tmp(tmp_path):
    rutas = {
        "ruta_casos": tmp_path / "casos.db",
        "ruta_memoria": tmp_path / "memoria.db",
        "ruta_trazas": tmp_path / "trazas",
    }

    piezas = demo_agente.construir_agente(
        **rutas,
        confirmador=demo_agente.confirmador_demo(),
        modo="reactivo",
        max_pasos=4,
        motor=MotorFalso(),
    )

    assert piezas.repositorio.ruta == rutas["ruta_casos"]
    assert piezas.repositorio.contar() == 10  # solo la semilla versionada
    assert piezas.memoria.largo.ruta == rutas["ruta_memoria"]
    assert piezas.agente.modo == "reactivo"
    assert piezas.agente.max_pasos == 4
    assert Path(rutas["ruta_memoria"]).exists()


def test_main_resuelve_de_punta_a_punta_con_modelo_guionado(tmp_path, capsys):
    """Camino real: herramientas reales, repositorio temporal y LLM guionado.

    La semilla tiene 10 casos, así que el primero que crea la demo es
    CASO-0011; el guion lo cita para encadenar nota y escalada.
    """
    id_esperado = "CASO-0011"
    guion = [
        mensaje_tool(llamada("crear_caso", {
            "cliente": CLIENTE,
            "categoria": "recepcion",
            "descripcion": "Notebook que no enciende.",
        }, "call_1")),
        mensaje_tool(llamada("agregar_nota", {
            "id_caso": id_esperado,
            "nota": "El equipo no enciende ni con el cargador conectado.",
        }, "call_2")),
        mensaje_tool(llamada("escalar_a_humano", {
            "id_caso": id_esperado,
            "motivo": "Requiere soporte de nivel 2.",
        }, "call_3")),
        mensaje_final("Caso registrado, con nota y escalado a nivel 2."),
    ]

    def construir_demo(**kwargs):
        return demo_agente.construir_agente(
            **kwargs, motor=MotorFalso(), llm=ModeloFalso(guion)
        )

    # Este guion demuestra el bucle reactivo, con independencia del default.
    codigo = demo_agente.main(["--modo", "reactivo"], constructor=construir_demo)

    salida = capsys.readouterr().out
    assert codigo == 0
    assert id_esperado in salida
    assert "Caso registrado, con nota y escalado a nivel 2." in salida
    assert "VERIFICADA" in salida


# --- S9: el default planificado está documentado ---------------------------
def test_el_default_planificado_esta_documentado():
    """S9: README y la ayuda de los scripts dicen que el default es planificado.

    Se comprueba el valor, no el nombre de la variable: el texto debe nombrar
    `planificado` como default. Y ninguna superficie de documentación puede
    afirmar lo contrario, que el default sea el bucle reactivo.
    """
    raiz = Path(__file__).resolve().parents[1]

    # 1) Ayuda de la demo: el default se nombra, no solo la variable que lo fija.
    ayuda_demo = " ".join(demo_agente._parser().format_help().split())
    assert "por defecto: planificado" in ayuda_demo
    assert "AGENTE_MODO" in ayuda_demo

    # 2) Ayuda de la comparación: el docstring es toda su ayuda de uso.
    from scripts import comparar_modos

    ayuda_comparar = " ".join((comparar_modos.__doc__ or "").split())
    assert "AGENTE_MODO=planificado" in ayuda_comparar
    assert "default" in ayuda_comparar.lower()

    # 3) README: la tabla de variables y la sección de modos lo documentan.
    readme = (raiz / "README.md").read_text(encoding="utf-8")
    assert "| `AGENTE_MODO` | `planificado` |" in readme
    assert "**Planificado** (`AGENTE_MODO=planificado`, por defecto)" in readme

    # 4) Ninguna superficie de documentación afirma un default reactivo.
    patron = re.compile(
        r"(por defecto|default|predeterminad)\w*[^.\n]{0,80}reactivo"
        r"|reactivo[^.\n]{0,80}(por defecto|default|predeterminad)",
        re.IGNORECASE,
    )
    superficies = [raiz / "README.md", raiz / "whitebox.md"]
    superficies += sorted((raiz / "docs").rglob("*.md"))
    superficies += sorted((raiz / "scripts").glob("*.py"))
    afirmaciones = [
        f"{ruta.relative_to(raiz)}:{numero}: {linea.strip()}"
        for ruta in superficies
        for numero, linea in enumerate(
            ruta.read_text(encoding="utf-8").splitlines(), 1
        )
        if patron.search(linea)
    ]
    assert afirmaciones == []
