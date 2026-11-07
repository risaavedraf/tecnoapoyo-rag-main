"""El canal de pensamiento del reactivo se imprime en dim italic.

No se usa con streaming: ahí el texto del modelo ya se transmite como parte
de la respuesta y repetirlo sería duplicarlo.
"""
import io

import pytest
from rich.console import Console


@pytest.fixture
def app_mod():
    import src.app

    return src.app


def test_mostrar_pensamiento_imprime_en_dim(app_mod):
    salida = io.StringIO()
    consola = Console(file=salida, width=100, force_terminal=False)

    app_mod.mostrar_pensamiento("Primero reviso el caso.", consola=consola)

    assert "· Primero reviso el caso." in salida.getvalue()
