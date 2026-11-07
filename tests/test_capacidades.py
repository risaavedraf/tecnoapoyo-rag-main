"""Respuesta determinista de capacidades: texto canónico y detector.

La pregunta «¿qué puedes hacer?» no debe gastar tokens ni una llamada al
modelo. El detector es la única puerta: si no reconoce la frase, el turno va
al camino normal.
"""
import pytest

from src.capacidades import (
    CAPACIDADES,
    TEXTO_CAPACIDADES,
    es_consulta_de_capacidades,
)
from src.herramientas import CLAVES_SALIDA


def test_las_capacidades_cubren_todas_las_tools_declaradas():
    # Fuente única: si se agrega una tool y no se documenta su capacidad, este
    # test lo señala en vez de dejar la respuesta canónica desactualizada.
    herramientas = {
        nombre for _, _, _, tools in CAPACIDADES for nombre in tools
    }
    assert herramientas == set(CLAVES_SALIDA)


@pytest.mark.parametrize("texto", [
    "¿Qué puedes hacer?",
    "HOLA, ¿para qué sirves?",
    "quiero saber qué puedes hacer",
    "cuáles son tus capacidades",
    "en qué me puedes ayudar",
    "¿qué hace el agente?",
    "a qué te dedicas",
    "¿qué sabes hacer exactamente?",
    "¿en qué me puedes ayudar hoy?",
    "¿cuáles son todas tus funciones?",
    "no sé qué puedes hacer",
])
def test_detecta_la_pregunta_de_capacidades(texto):
    assert es_consulta_de_capacidades(texto)


@pytest.mark.parametrize("texto", [
    "¿puedes hacer un caso nuevo?",
    "¿qué puedes hacer con mi caso?",
    "hola",
    "¿cuánto demora el diagnóstico?",
    "lista mis casos",
    "no me digas qué puedes hacer",
    "no quiero saber qué puedes hacer",
])
def test_no_confunde_consultas_operativas_con_capacidades(texto):
    assert not es_consulta_de_capacidades(texto)


def test_el_texto_explica_las_capacidades_y_los_limites():
    assert TEXTO_CAPACIDADES.startswith("Puedo ayudarte con esto:")
    for familia in ("Documentación", "Casos", "Escrituras", "Plazos",
                    "Datos faltantes", "Memoria"):
        assert familia in TEXTO_CAPACIDADES
    assert "No puedo ver casos de otros clientes" in TEXTO_CAPACIDADES
    assert "sin respaldo documental" in TEXTO_CAPACIDADES
