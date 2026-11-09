"""Persistencia y validación de la clave del proveedor en `.env`.

Cubren las reglas del comando de configuración: escribir solo la variable de la
clave sin tocar el resto del archivo, no dejar el archivo a medias si la
escritura falla, rechazar entradas que no son una clave y clasificar la
validación remota sin bloquear el guardado cuando no hay red.

Ningún test hace peticiones HTTP reales: `urllib.request.urlopen` se sustituye.
"""
import os
import stat
import subprocess
import sys
import urllib.error
from email.message import Message

import pytest

from src import credenciales


class Respuesta:
    """Respuesta mínima del endpoint para la validación."""

    def __init__(self, status):
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_guardar_clave_crea_el_env_desde_la_plantilla(tmp_path):
    plantilla = tmp_path / "plantilla.env"
    plantilla.write_text(
        "LLM_BASE_URL=\nLLM_API_KEY=\nLLM_MODEL=ministral-8b-latest\n",
        encoding="utf-8",
    )
    ruta = tmp_path / ".env"

    huella = credenciales.guardar_clave(
        "clave-secreta-1234", ruta=ruta, plantilla=plantilla
    )

    texto = ruta.read_text(encoding="utf-8")
    assert "LLM_API_KEY=clave-secreta-1234" in texto
    assert "LLM_BASE_URL=" in texto
    assert "LLM_MODEL=ministral-8b-latest" in texto
    assert texto.endswith("\n")
    # La huella es para mostrar, nunca la clave completa.
    assert huella == "clav…1234"


def test_guardar_clave_conserva_el_resto_del_archivo(tmp_path):
    ruta = tmp_path / ".env"
    ruta.write_text(
        "# cabecera\n"
        "LLM_BASE_URL=https://api.mistral.ai/v1\n"
        "LLM_API_KEY=vieja\n"
        "LLM_MODEL=ministral-8b-latest\n",
        encoding="utf-8",
    )

    credenciales.guardar_clave("nueva-clave-9999", ruta=ruta)

    assert ruta.read_text(encoding="utf-8").splitlines() == [
        "# cabecera",
        "LLM_BASE_URL=https://api.mistral.ai/v1",
        "LLM_API_KEY=nueva-clave-9999",
        "LLM_MODEL=ministral-8b-latest",
    ]


def test_guardar_clave_agrega_la_variable_si_falta(tmp_path):
    ruta = tmp_path / ".env"
    ruta.write_text("LLM_BASE_URL=https://api.mistral.ai/v1\n", encoding="utf-8")

    credenciales.guardar_clave("clave-nueva-0001", ruta=ruta)

    assert "LLM_API_KEY=clave-nueva-0001" in ruta.read_text(
        encoding="utf-8"
    ).splitlines()


def test_guardar_clave_no_duplica_la_variable(tmp_path):
    ruta = tmp_path / ".env"
    ruta.write_text("LLM_API_KEY=vieja\n", encoding="utf-8")

    credenciales.guardar_clave("clave-unica-0002", ruta=ruta)
    credenciales.guardar_clave("clave-unica-0002", ruta=ruta)

    assert ruta.read_text(encoding="utf-8").count("LLM_API_KEY=") == 1


def test_si_falla_el_reemplazo_el_env_original_queda_intacto(tmp_path, monkeypatch):
    ruta = tmp_path / ".env"
    ruta.write_text("LLM_API_KEY=vieja\n", encoding="utf-8")

    def romper(*args, **kwargs):
        raise OSError("disco lleno")

    monkeypatch.setattr(os, "replace", romper)

    with pytest.raises(OSError):
        credenciales.guardar_clave("clave-nueva-1234", ruta=ruta)

    assert ruta.read_text(encoding="utf-8") == "LLM_API_KEY=vieja\n"
    # El temporal no queda tirado al lado del .env.
    assert [p.name for p in tmp_path.iterdir() if p.name != ".env"] == []


@pytest.mark.skipif(os.name != "posix", reason="permisos POSIX")
def test_el_env_queda_legible_solo_por_el_dueno(tmp_path):
    ruta = tmp_path / ".env"

    credenciales.guardar_clave("clave-permisos-1234", ruta=ruta)

    assert stat.S_IMODE(ruta.stat().st_mode) == 0o600


@pytest.mark.parametrize(
    "entrada",
    ["", "   ", "abc def", "clave\t420", "a\nb", 'clave"1234', "clave'1234"],
)
def test_normalizar_rechaza_entradas_que_no_son_una_clave(entrada):
    with pytest.raises(ValueError):
        credenciales.normalizar(entrada)


def test_normalizar_quita_el_salto_de_linea_del_pegado():
    assert credenciales.normalizar("  clave-pegada-1234\n") == "clave-pegada-1234"


def test_validar_clave_ok(monkeypatch):
    monkeypatch.setattr(
        credenciales.urllib.request, "urlopen", lambda *a, **k: Respuesta(200)
    )

    assert credenciales.validar_clave("clave", "https://api.mistral.ai/v1") == "ok"


def test_validar_clave_envia_la_clave_en_la_cabecera(monkeypatch):
    capturada = {}

    def urlopen(peticion, timeout=None):
        capturada["url"] = peticion.full_url
        capturada["auth"] = peticion.get_header("Authorization")
        return Respuesta(200)

    monkeypatch.setattr(credenciales.urllib.request, "urlopen", urlopen)

    credenciales.validar_clave("clave-abc", "https://api.mistral.ai/v1")

    assert capturada["url"] == "https://api.mistral.ai/v1/models"
    assert capturada["auth"] == "Bearer clave-abc"


def test_validar_clave_rechazada(monkeypatch):
    def urlopen(*args, **kwargs):
        raise urllib.error.HTTPError("url", 401, "Unauthorized", Message(), None)

    monkeypatch.setattr(credenciales.urllib.request, "urlopen", urlopen)

    assert (
        credenciales.validar_clave("clave", "https://api.mistral.ai/v1")
        == "rechazada"
    )


def test_validar_clave_sin_red(monkeypatch):
    def urlopen(*args, **kwargs):
        raise urllib.error.URLError("sin red")

    monkeypatch.setattr(credenciales.urllib.request, "urlopen", urlopen)

    assert credenciales.validar_clave("clave", "https://api.mistral.ai/v1") == "sin_red"


def test_validar_clave_rechaza_un_endpoint_que_no_es_http(monkeypatch):
    """Un LLM_BASE_URL con otro esquema no debe llegar a `urlopen`."""
    def urlopen(*args, **kwargs):
        raise AssertionError("no debe abrirse la URL")

    monkeypatch.setattr(credenciales.urllib.request, "urlopen", urlopen)

    with pytest.raises(ValueError):
        credenciales.validar_clave("clave", "file:///etc/passwd")


class TerminalFalsa:
    """Sustituto de `sys.stdin` con la única interfaz que usa el comando."""

    def __init__(self, es_terminal=True):
        self._es_terminal = es_terminal

    def isatty(self):
        return self._es_terminal


def test_configurar_clave_exige_terminal(monkeypatch, capsys):
    from src import configurar_clave

    monkeypatch.setattr(sys, "stdin", TerminalFalsa(es_terminal=False))

    assert configurar_clave.main() == 2
    assert "terminal" in capsys.readouterr().err


def test_configurar_clave_no_guarda_si_el_endpoint_rechaza(
    tmp_path, monkeypatch, capsys
):
    from src import configurar_clave

    ruta = tmp_path / ".env"
    monkeypatch.setattr(sys, "stdin", TerminalFalsa())
    monkeypatch.setattr(
        configurar_clave.getpass, "getpass", lambda *a, **k: "clave-rechazada-1234"
    )
    monkeypatch.setattr(credenciales, "RUTA_ENV", ruta)

    def urlopen(*args, **kwargs):
        raise urllib.error.HTTPError("url", 401, "Unauthorized", Message(), None)

    monkeypatch.setattr(credenciales.urllib.request, "urlopen", urlopen)

    assert configurar_clave.main() == 1
    assert not ruta.exists()
    assert "rechazó" in capsys.readouterr().err


def test_configurar_clave_guarda_la_clave_sin_mostrarla(
    tmp_path, monkeypatch, capsys
):
    from src import configurar_clave

    ruta = tmp_path / ".env"
    monkeypatch.setattr(sys, "stdin", TerminalFalsa())
    monkeypatch.setattr(
        configurar_clave.getpass, "getpass", lambda *a, **k: "clave-valida-1234"
    )
    monkeypatch.setattr(credenciales, "RUTA_ENV", ruta)
    monkeypatch.setattr(credenciales, "RUTA_PLANTILLA", tmp_path / "sin-plantilla")
    monkeypatch.setattr(
        credenciales.urllib.request, "urlopen", lambda *a, **k: Respuesta(200)
    )

    assert configurar_clave.main() == 0

    assert ruta.read_text(encoding="utf-8") == "LLM_API_KEY=clave-valida-1234\n"
    salida = capsys.readouterr()
    assert "clave-valida-1234" not in salida.out + salida.err
    assert "clav…1234" in salida.out


def test_configurar_clave_guarda_sin_red_con_aviso(tmp_path, monkeypatch, capsys):
    from src import configurar_clave

    ruta = tmp_path / ".env"
    monkeypatch.setattr(sys, "stdin", TerminalFalsa())
    monkeypatch.setattr(
        configurar_clave.getpass, "getpass", lambda *a, **k: "clave-offline-5678"
    )
    monkeypatch.setattr(credenciales, "RUTA_ENV", ruta)
    monkeypatch.setattr(credenciales, "RUTA_PLANTILLA", tmp_path / "sin-plantilla")

    def urlopen(*args, **kwargs):
        raise urllib.error.URLError("sin red")

    monkeypatch.setattr(credenciales.urllib.request, "urlopen", urlopen)

    assert configurar_clave.main() == 0
    assert ruta.read_text(encoding="utf-8") == "LLM_API_KEY=clave-offline-5678\n"
    assert "Aviso" in capsys.readouterr().err


def test_app_sin_clave_instruye_como_configurarla():
    """La app arranca sin clave y falla con la instrucción, no con un traceback."""
    # El hijo importa numpy/faiss antes de mirar la clave: con 12 núcleos
    # OpenBLAS reserva buffers por hilo y, bajo la huella de la suite completa,
    # falla con "Memory allocation still failed" antes de llegar al chequeo.
    # Un hilo alcanza para este subproceso y deja el test determinista.
    entorno = {
        **os.environ,
        "LLM_API_KEY": "",
        "OPENBLAS_NUM_THREADS": "1",
    }

    proceso = subprocess.run(
        [sys.executable, "-m", "src.app"],
        cwd=credenciales.RAIZ,
        env=entorno,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert proceso.returncode != 0
    assert "configurar_clave" in proceso.stderr
    assert "LLM_API_KEY" in proceso.stderr
