"""Lectura del plan validado desde la consola: JSON y panel.

`src.app` se importa tarde porque construye el motor y exige la clave; los
helpers no tocan red ni estado, así que se prueban con una consola capturada.
"""
import io
import json

import pytest
from rich.console import Console

from src.planificador import PasoPlan, Plan


@pytest.fixture
def app_mod():
    import src.app

    return src.app


def consola_de_prueba():
    salida = io.StringIO()
    consola = Console(file=salida, width=100, force_terminal=False)
    return salida, consola


def plan_de_prueba():
    return Plan(
        objetivo="Informar el plazo del CASO-0007",
        reflexion="El caso ya está identificado y el plazo depende de su fecha de creación.",
        version=1,
        pasos=[
            PasoPlan(
                id="p1",
                herramienta="consultar_caso",
                argumentos={"id_caso": "CASO-0007"},
                depende_de=[],
                criterio_exito="Se obtuvo el caso.",
            ),
        ],
    )


def test_sin_plan_no_hay_json(app_mod):
    assert app_mod.plan_como_json(None) is None


def test_el_plan_se_serializa_como_json_legible(app_mod):
    texto = app_mod.plan_como_json(plan_de_prueba())

    assert json.loads(texto)["pasos"][0]["id"] == "p1"
    assert "\n" in texto  # indentado, no una línea ilegible


def test_el_plan_se_dibuja_como_card_sin_sintaxis_json(app_mod):
    salida, consola = consola_de_prueba()

    assert app_mod.mostrar_plan(plan_de_prueba(), consola=consola) is True

    visible = salida.getvalue()
    assert "Plan · 1 paso" in visible
    assert "consultar_caso" in visible
    assert "id_caso = CASO-0007" in visible
    assert "criterio de éxito · Se obtuvo el caso." in visible
    assert "por qué · El caso ya está identificado" in visible
    assert '"pasos"' not in visible
    assert "{" not in visible


def test_el_plan_en_json_se_conserva_para_copiar(app_mod):
    salida, consola = consola_de_prueba()

    assert app_mod.mostrar_plan(
        plan_de_prueba(), consola=consola, como_json=True
    ) is True

    visible = salida.getvalue()
    assert "Plan · JSON" in visible
    assert '"pasos"' in visible
    assert "CASO-0007" in visible


def test_un_plan_vacio_lo_dice(app_mod):
    plan = Plan(objetivo="Saludar", version=1, pasos=[])
    salida, consola = consola_de_prueba()

    assert app_mod.mostrar_plan(plan, consola=consola) is True

    visible = salida.getvalue()
    assert "Plan · vacío" in visible
    assert "no necesita herramientas" in visible
    assert "por qué" not in visible


def test_mostrar_plan_sin_plan_no_imprime_nada(app_mod):
    salida, consola = consola_de_prueba()

    assert app_mod.mostrar_plan(None, consola=consola) is False
    assert salida.getvalue() == ""


# --- S4: la card no presenta el criterio como un hecho consumado ---------
def paso_ejecutado(herramienta, argumentos, ok):
    """Traza mínima de un paso ya ejecutado (lo mismo que publica el agente)."""
    from types import SimpleNamespace

    return SimpleNamespace(herramienta=herramienta, argumentos=argumentos, ok=ok)


def plan_con_identificacion_fallida():
    return Plan(
        objetivo="Identificar al cliente",
        reflexion="La pregunta pide el estado de un caso.",
        version=1,
        pasos=[
            PasoPlan(
                id="p1",
                herramienta="identificar_cliente",
                argumentos={"nombre": "Ricardo"},
                depende_de=[],
                criterio_exito="Se identificó al cliente en la sesión.",
            ),
        ],
    )


def test_el_criterio_se_etiqueta_como_criterio_de_exito(app_mod):
    salida, consola = consola_de_prueba()

    app_mod.mostrar_plan(plan_de_prueba(), consola=consola)

    visible = salida.getvalue()
    assert "criterio de éxito · Se obtuvo el caso." in visible
    assert "criterio · Se obtuvo el caso." not in visible


def test_un_paso_con_ok_no_se_muestra_como_fallido(app_mod):
    salida, consola = consola_de_prueba()

    app_mod.mostrar_plan(
        plan_con_identificacion_fallida(), consola=consola,
        pasos_ejecutados=[paso_ejecutado("identificar_cliente", {"nombre": "Ricardo"}, True)],
    )

    visible = salida.getvalue()
    assert "falló" not in visible
    assert "✗" not in visible
    assert "criterio de éxito · Se identificó al cliente en la sesión." in visible


def test_un_paso_fallido_se_muestra_como_fallido(app_mod):
    salida, consola = consola_de_prueba()

    app_mod.mostrar_plan(
        plan_con_identificacion_fallida(), consola=consola,
        pasos_ejecutados=[paso_ejecutado("identificar_cliente", {"nombre": "Ricardo"}, False)],
    )

    visible = salida.getvalue()
    assert "✗" in visible
    assert "falló" in visible
    # La estructura y el "por qué" siguen ahí: solo cambia el veredicto.
    assert "Plan · 1 paso" in visible
    assert "por qué · La pregunta pide el estado de un caso." in visible
    assert "nombre = Ricardo" in visible


def test_sin_traza_la_card_no_inventa_veredictos(app_mod):
    # `/plan` redibuja el plan sin la traza: sin dato de ejecución no hay ✗.
    salida, consola = consola_de_prueba()

    app_mod.mostrar_plan(plan_con_identificacion_fallida(), consola=consola)

    visible = salida.getvalue()
    assert "✗" not in visible
    assert "falló" not in visible
