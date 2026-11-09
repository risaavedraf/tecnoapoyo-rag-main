"""Arranque compartido de la consola y los scripts (S5-S7).

Dos pasos estaban copiados en cinco archivos: reconectar los flujos en UTF-8 y
cortar el arranque sin clave con la instrucción que la configura. `src.arranque`
los define una sola vez. Aquí se fija su contrato observable: el mensaje exacto,
que la guarda no construya ningún cliente y que la reconexión tolere flujos que
no la implementan.

El último test es de punta a punta: un script documentado sin `LLM_API_KEY`
tiene que imprimir la instrucción y salir distinto de cero, no el traceback del
proveedor.
"""
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from src import arranque, config, credenciales

RAIZ_PROYECTO = Path(__file__).resolve().parents[1]


# --- exigir_clave ---------------------------------------------------------
def test_exigir_clave_sin_clave_corta_con_el_mensaje_accionable(monkeypatch):
    """Sin credencial el arranque se corta con la instrucción, no con un error."""
    monkeypatch.setattr(config, "LLM_API_KEY", "")

    with pytest.raises(SystemExit) as error:
        arranque.exigir_clave()

    # El texto es el de `credenciales`, no una copia: la consola y los scripts
    # no pueden separarse con el tiempo.
    assert str(error.value) == credenciales.MENSAJE_FALTA_CLAVE


def test_exigir_clave_con_clave_no_interrumpe(monkeypatch):
    monkeypatch.setattr(config, "LLM_API_KEY", "clave-de-prueba")

    assert arranque.exigir_clave() is None


def test_exigir_clave_no_construye_ningun_cliente(monkeypatch):
    """La guarda mira la credencial: no arma el cliente del proveedor.

    Se espían las dos clases que construyen clientes en el proyecto. La guarda
    se ejercita en sus dos ramas (sin clave y con clave) y ninguna puede
    instanciarlas: pedir la clave no es empezar a hablar con el proveedor.
    """
    import langchain_openai

    construidos = []

    class ClienteEspia:
        def __init__(self, *args, **kwargs):
            construidos.append((args, kwargs))

    monkeypatch.setattr(langchain_openai, "ChatOpenAI", ClienteEspia)
    monkeypatch.setattr(langchain_openai, "OpenAIEmbeddings", ClienteEspia)
    monkeypatch.setattr(config, "LLM_API_KEY", "")

    with pytest.raises(SystemExit):
        arranque.exigir_clave()

    monkeypatch.setattr(config, "LLM_API_KEY", "clave-de-prueba")
    assert arranque.exigir_clave() is None

    assert construidos == []
    # Tampoco las arrastra a su espacio de nombres: el módulo no importa el
    # stack del proveedor.
    assert not {"ChatOpenAI", "OpenAIEmbeddings"} & set(vars(arranque))


# --- reconectar_flujos ----------------------------------------------------
class FlujoFalso:
    """Flujo con `reconfigure` que registra el encoding recibido."""

    def __init__(self, falla=None):
        self.falla = falla
        self.encodings = []

    def reconfigure(self, encoding=None):
        if self.falla is not None:
            raise self.falla
        self.encodings.append(encoding)


def test_reconectar_flujos_reconecta_los_tres_flujos_en_utf8(monkeypatch):
    entrada, salida, errores = FlujoFalso(), FlujoFalso(), FlujoFalso()
    monkeypatch.setattr(sys, "stdin", entrada)
    monkeypatch.setattr(sys, "stdout", salida)
    monkeypatch.setattr(sys, "stderr", errores)

    arranque.reconectar_flujos()

    for flujo in (entrada, salida, errores):
        assert flujo.encodings == ["utf-8"]


def test_reconectar_flujos_es_idempotente(monkeypatch):
    """Llamarla dos veces es seguro: mismo efecto, sin error."""
    entrada, salida, errores = FlujoFalso(), FlujoFalso(), FlujoFalso()
    monkeypatch.setattr(sys, "stdin", entrada)
    monkeypatch.setattr(sys, "stdout", salida)
    monkeypatch.setattr(sys, "stderr", errores)

    arranque.reconectar_flujos()
    arranque.reconectar_flujos()

    for flujo in (entrada, salida, errores):
        assert flujo.encodings == ["utf-8", "utf-8"]


def test_reconectar_flujos_tolera_flujos_sin_reconfigure_y_con_valueerror(
    monkeypatch,
):
    """Un flujo sin `reconfigure` y uno que se niega no rompen el arranque.

    Es el caso de los flujos que entrega el entorno de test y de una consola no
    reconectable: la reconexión es una mejora, no un requisito para arrancar.
    """
    sin_reconfigure = SimpleNamespace()  # ni siquiera tiene el atributo
    se_niega = FlujoFalso(falla=ValueError("flujo no reconectable"))
    normal = FlujoFalso()
    monkeypatch.setattr(sys, "stdin", sin_reconfigure)
    monkeypatch.setattr(sys, "stdout", se_niega)
    monkeypatch.setattr(sys, "stderr", normal)

    arranque.reconectar_flujos()

    # Ninguna de las dos excepciones se propagó y el flujo reconectable sí se
    # reconectó.
    assert normal.encodings == ["utf-8"]


# --- S6: un script documentado sin clave ----------------------------------
def test_build_index_sin_clave_imprime_la_instruccion_y_sale_distinto_de_cero():
    """Sin `LLM_API_KEY` el script no llega al traceback del proveedor.

    `scripts/build_index.py` construye el índice de punta a punta, así que es el
    peor caso: sin la guarda, el cliente de OpenAI falla con
    `Missing credentials` y el mensaje no dice qué comando configura la clave.
    """
    # El hijo importa numpy/faiss: con todos los núcleos OpenBLAS puede fallar
    # antes del import bajo la huella de la suite, como en test_credenciales.
    entorno = {
        **os.environ,
        "LLM_API_KEY": "",
        "OPENBLAS_NUM_THREADS": "1",
    }

    # El hijo reconecta sus flujos en UTF-8 (`reconectar_flujos`), así que su
    # salida se lee en UTF-8: con el encoding de la locale, el mensaje con tilde
    # llegaría mal decodificado en Windows.
    proceso = subprocess.run(
        [sys.executable, "scripts/build_index.py"],
        cwd=RAIZ_PROYECTO,
        env=entorno,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=180,
    )

    assert proceso.returncode != 0
    assert credenciales.MENSAJE_FALTA_CLAVE in proceso.stderr
    assert "Traceback" not in proceso.stderr
    assert "Missing credentials" not in proceso.stderr
    # La guarda corre antes de construir nada: ni corpus ni índice.
    assert "[OK]" not in proceso.stdout


def test_evaluar_agente_sin_clave_sigue_validando_el_dataset():
    """Preserva el camino offline: validar el contrato no exige credenciales.

    `--verificar-dataset` no construye motor, repositorio ni agente. El arranque
    del script no puede convertirlo en un comando que pide la clave: sería un
    requisito inventado sobre una ruta que no habla con el proveedor.
    """
    entorno = {
        **os.environ,
        "LLM_API_KEY": "",
        "OPENBLAS_NUM_THREADS": "1",
    }

    proceso = subprocess.run(
        [sys.executable, "scripts/evaluar_agente.py", "--verificar-dataset"],
        cwd=RAIZ_PROYECTO,
        env=entorno,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=180,
    )

    assert proceso.returncode == 0, proceso.stderr
    assert "esquema válido" in proceso.stdout
    assert credenciales.MENSAJE_FALTA_CLAVE not in proceso.stderr
