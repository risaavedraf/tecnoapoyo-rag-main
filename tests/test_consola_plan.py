"""Lectura del plan validado desde la consola: JSON y panel.

`src.app` se importa tarde porque construye el motor y exige la clave; los
helpers no tocan red ni estado, así que se prueban con una consola capturada.
"""
import io
import json

import pytest
from rich.console import Console

from src.planificador import ATRIBUTO_INTENTO_PLAN, PasoPlan, Plan


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
def paso_ejecutado(herramienta, argumentos, ok, paso_id=None, intento=None):
    """Traza mínima de un paso ya ejecutado (lo mismo que publica el agente).

    `intento` es el plan que ejecutó el paso: un turno con re-plan acumula en la
    traza pasos de varios planes y los ids (`p1`, `p2`, ...) se repiten.
    """
    from types import SimpleNamespace

    paso = SimpleNamespace(
        herramienta=herramienta, argumentos=argumentos, ok=ok, paso_id=paso_id,
    )
    if intento is not None:
        setattr(paso, ATRIBUTO_INTENTO_PLAN, intento)
    return paso


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


def plan_con_placeholder():
    return Plan(
        objetivo="Anotar la fecha de creación del caso",
        reflexion="La nota cita la fecha que devuelve la búsqueda.",
        version=1,
        pasos=[
            PasoPlan(
                id="p1",
                herramienta="buscar_documentacion",
                argumentos={"pregunta": "fecha de creación del caso"},
                depende_de=[],
                criterio_exito="Se obtuvo la fecha.",
            ),
            PasoPlan(
                id="p2",
                herramienta="agregar_nota",
                argumentos={
                    "id_caso": "CASO-0007",
                    "nota": "{{p1.datos.fecha_creacion}}",
                },
                depende_de=["p1"],
                criterio_exito="La nota quedó registrada.",
            ),
        ],
    )


def test_un_paso_con_placeholder_fallido_se_marca_en_la_card(app_mod):
    # S2: la traza trae los argumentos ya resueltos, así que el veredicto no
    # puede depender de comparar los del plan (que traen el placeholder).
    salida, consola = consola_de_prueba()

    app_mod.mostrar_plan(
        plan_con_placeholder(), consola=consola,
        pasos_ejecutados=[
            paso_ejecutado(
                "buscar_documentacion", {"pregunta": "fecha de creación del caso"},
                True, paso_id="p1",
            ),
            paso_ejecutado(
                "agregar_nota", {"id_caso": "CASO-0007", "nota": "2026-01-05"},
                False, paso_id="p2",
            ),
        ],
    )

    lineas = salida.getvalue().splitlines()
    fallida = next(linea for linea in lineas if "agregar_nota" in linea)
    exitosa = next(linea for linea in lineas if "buscar_documentacion" in linea)
    assert "✗" in fallida
    assert "falló" in fallida
    # El paso que se ejecutó bien no se marca, y el plan se sigue leyendo.
    assert "✗" not in exitosa
    assert "falló" not in exitosa
    assert "Plan · 2 pasos" in salida.getvalue()
    assert "nota = {{p1.datos.fecha_creacion}}" in salida.getvalue()

    # Sin traza la card no inventa veredictos: el placeholder no da un veredicto.
    sin_traza, consola_sin_traza = consola_de_prueba()
    app_mod.mostrar_plan(plan_con_placeholder(), consola=consola_sin_traza)
    assert "✗" not in sin_traza.getvalue()
    assert "falló" not in sin_traza.getvalue()


def plan_con_dos_pasos():
    """Plan final de un turno que re-planificó: `p1` lista y `p2` consulta."""
    return Plan(
        objetivo="Informar el estado de un caso",
        reflexion="Primero se listan los casos y luego se consulta el indicado.",
        version=1,
        pasos=[
            PasoPlan(
                id="p1",
                herramienta="listar_casos",
                argumentos={"limite": 1},
                depende_de=[],
                criterio_exito="Se listaron los casos.",
            ),
            PasoPlan(
                id="p2",
                herramienta="consultar_caso",
                argumentos={"id_caso": "CASO-0007"},
                depende_de=[],
                criterio_exito="Se obtuvo el caso.",
            ),
        ],
    )


def linea_de(salida, herramienta):
    return next(linea for linea in salida.splitlines() if herramienta in linea)


# --- R4-002: el veredicto no se hereda de un plan anterior ---------------
def test_un_veredicto_de_un_plan_anterior_no_se_hereda(app_mod):
    # El turno re-planificó y el plan final repite el id `p1`. El `p1` fallido
    # del plan descartado no puede marcar como fallido el `p1` que el plan que
    # se muestra sí ejecutó bien: eso inventa un veredicto sobre el plan visible.
    salida, consola = consola_de_prueba()

    app_mod.mostrar_plan(
        plan_con_identificacion_fallida(), consola=consola,
        pasos_ejecutados=[
            paso_ejecutado("identificar_cliente", {"nombre": "Ricardo"}, False,
                           paso_id="p1", intento=1),
            paso_ejecutado("identificar_cliente", {"nombre": "Ricardo"}, True,
                           paso_id="p1", intento=2),
        ],
    )

    visible = salida.getvalue()
    assert "✗" not in visible
    assert "falló" not in visible


def test_un_paso_del_plan_final_sin_ejecutar_no_hereda_veredicto(app_mod):
    # El plan final tiene `p1` y `p2`; el intento que falló ejecutó los dos y el
    # plan final solo llegó a `p1`. `p2` no se ejecutó en el plan que se muestra,
    # así que su traza anterior no puede marcarlo como fallido.
    salida, consola = consola_de_prueba()

    app_mod.mostrar_plan(
        plan_con_dos_pasos(), consola=consola,
        pasos_ejecutados=[
            paso_ejecutado("listar_casos", {"limite": 1}, True,
                           paso_id="p1", intento=1),
            paso_ejecutado("consultar_caso", {"id_caso": "CASO-0007"}, False,
                           paso_id="p2", intento=1),
            paso_ejecutado("listar_casos", {"limite": 1}, True,
                           paso_id="p1", intento=2),
        ],
    )

    visible = salida.getvalue()
    assert "falló" not in visible
    assert "✗" not in linea_de(visible, "consultar_caso")
    assert "criterio de éxito · Se obtuvo el caso." in visible


def test_un_fallo_del_plan_final_si_se_marca(app_mod):
    # El fallo que sí ocurrió en el plan que se muestra se sigue marcando, y el
    # paso homónimo que había fallado en el intento anterior no contamina a `p1`.
    salida, consola = consola_de_prueba()

    app_mod.mostrar_plan(
        plan_con_dos_pasos(), consola=consola,
        pasos_ejecutados=[
            paso_ejecutado("listar_casos", {"limite": 1}, False,
                           paso_id="p1", intento=1),
            paso_ejecutado("listar_casos", {"limite": 1}, True,
                           paso_id="p1", intento=2),
            paso_ejecutado("consultar_caso", {"id_caso": "CASO-0007"}, False,
                           paso_id="p2", intento=2),
        ],
    )

    visible = salida.getvalue()
    assert "✗" in linea_de(visible, "consultar_caso")
    assert "falló" in linea_de(visible, "consultar_caso")
    assert "✗" not in linea_de(visible, "listar_casos")
    assert "falló" not in linea_de(visible, "listar_casos")


def test_sin_traza_la_card_no_inventa_veredictos(app_mod):
    # `/plan` redibuja el plan sin la traza: sin dato de ejecución no hay ✗.
    salida, consola = consola_de_prueba()

    app_mod.mostrar_plan(plan_con_identificacion_fallida(), consola=consola)

    visible = salida.getvalue()
    assert "✗" not in visible
    assert "falló" not in visible
