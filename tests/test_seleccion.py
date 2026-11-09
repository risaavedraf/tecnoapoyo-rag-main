"""Contratos del menú de selección por flechas y de su respaldo numerado.

Sin terminal real y sin red: las teclas se inyectan con un lector falso y la
entrada del respaldo se sustituye con monkeypatch. La `Console` escribe en un
`StringIO` no interactivo, así que el texto renderizado se puede afirmar tal
cual (Rich no emite códigos de control sin terminal).

El respaldo se dispara cuando el lector devuelve `"sin_soporte"`, que es lo que
hace `leer_tecla` sin TTY. Un lector inyectado permite probar ese camino sin
tocar `sys.stdin`.
"""
import io
import os
import sys
import types
from dataclasses import FrozenInstanceError

import pytest
from rich.console import Console

from src import seleccion
from src.seleccion import Opcion, elegir, leer_tecla

PREGUNTA = "¿Guardar la nota?"


class _Bomba(BaseException):
    """Falla el test si un camino que no debía ejecutarse lo hace.

    Hereda de `BaseException` a propósito: el adaptador de teclado degrada a
    `"sin_soporte"` ante cualquier `Exception`, así que una bomba normal
    quedaría silenciada.
    """


class _StdinFalso:
    """Stdin mínimo: solo lo que `leer_tecla` consulta antes de leer teclas."""

    def __init__(self, es_tty):
        self.es_tty = es_tty

    def isatty(self):
        return self.es_tty


class _StdinConDescriptor(_StdinFalso):
    """Stdin falso con el descriptor que usará el camino POSIX."""

    def fileno(self):
        return 7


def _instalar_terminal_posix(monkeypatch, hay_mas):
    """Instala `termios`, `tty` y `select` falsos y devuelve los eventos.

    Permite ejercitar el camino POSIX desde Windows sin un terminal real: el
    orden de los eventos prueba que el modo crudo se restaura siempre.
    """
    eventos = []

    def tcgetattr(descriptor):
        eventos.append(("get", descriptor))
        return "atributos"

    def tcsetattr(descriptor, accion, atributos):
        eventos.append(("set", descriptor, accion, atributos))

    def setraw(descriptor):
        eventos.append(("raw", descriptor))

    def select(lectura, escritura, errores, plazo):
        eventos.append(("select", plazo))
        if hay_mas:
            return (lectura, [], [])
        return ([], [], [])

    termios = types.SimpleNamespace(
        TCSADRAIN=1, error=OSError,
        tcgetattr=tcgetattr, tcsetattr=tcsetattr,
    )
    monkeypatch.setitem(sys.modules, "termios", termios)
    monkeypatch.setitem(sys.modules, "tty", types.SimpleNamespace(setraw=setraw))
    monkeypatch.setitem(
        sys.modules, "select", types.SimpleNamespace(select=select)
    )
    return eventos


@pytest.fixture
def consola():
    return Console(file=io.StringIO(), force_terminal=False, width=100)


@pytest.fixture
def opciones():
    return (
        Opcion("Sí, guardar", "si", "Escribe la nota en el caso."),
        Opcion("No, cancelar", "no", "No se toca el caso.", predeterminada=True),
        Opcion("Volver", "volver", "Regresa al menú anterior."),
    )


def _texto(consola):
    return consola.file.getvalue()


def _lector(*teclas):
    pendientes = iter(teclas)
    return lambda: next(pendientes)


def _entrada_falsa(consola, monkeypatch, respuestas):
    """Sustituye `console.input` y devuelve la lista de prompts mostrados."""
    pendientes = iter(respuestas)
    prompts = []

    def leer(prompt=""):
        prompts.append(prompt)
        return next(pendientes)

    monkeypatch.setattr(consola, "input", leer)
    return prompts


def test_opcion_es_inmutable_y_tiene_valores_por_defecto():
    opcion = Opcion("Sí", "si")
    assert opcion.descripcion == ""
    assert opcion.predeterminada is False
    atributo = "etiqueta"
    with pytest.raises(FrozenInstanceError):
        setattr(opcion, atributo, "No")


# --- Camino de flechas -----------------------------------------------------


def test_abajo_abajo_enter_selecciona_la_tercera(consola, opciones):
    elegida = elegir(consola, PREGUNTA, opciones,
                     lector_teclas=_lector("abajo", "abajo", "enter"))

    assert isinstance(elegida, Opcion)
    assert elegida.valor == "volver"


def test_abajo_abajo_enter_con_dos_opciones_envuelve(consola):
    opciones = (Opcion("Sí", "si"), Opcion("No", "no"))

    elegida = elegir(consola, PREGUNTA, opciones,
                     lector_teclas=_lector("abajo", "abajo", "enter"))

    assert elegida.valor == "si"


def test_el_menu_detiene_el_live_al_terminar(consola, opciones):
    elegir(consola, PREGUNTA, opciones, lector_teclas=_lector("enter"))

    assert getattr(consola, "_live", None) is None


def test_el_menu_muestra_pregunta_cursor_y_descripcion_de_la_actual(consola, opciones):
    elegir(consola, PREGUNTA, opciones,
           lector_teclas=_lector("abajo", "abajo", "enter"))
    texto = _texto(consola)

    assert PREGUNTA in texto
    assert "❯ 3. Volver" in texto
    assert "1. Sí, guardar" in texto
    assert "2. No, cancelar" in texto
    assert "Regresa al menú anterior." in texto
    # La descripción visible es la de la opción actual, no la de la anterior.
    assert "Escribe la nota en el caso." not in texto
    # Un solo cursor: el resaltado es exclusivo.
    assert texto.count("❯") == 1


def test_arriba_desde_el_primero_envuelve_al_ultimo(consola, opciones):
    elegida = elegir(consola, PREGUNTA, opciones, indice_inicial=0,
                     lector_teclas=_lector("arriba", "enter"))

    assert elegida.valor == "volver"
    assert "❯ 3. Volver" in _texto(consola)


def test_abajo_desde_el_ultimo_envuelve_al_primero(consola, opciones):
    elegida = elegir(consola, PREGUNTA, opciones, indice_inicial=2,
                     lector_teclas=_lector("abajo", "enter"))

    assert elegida.valor == "si"
    assert "❯ 1. Sí, guardar" in _texto(consola)


def test_indice_inicial_desplaza_el_cursor(consola, opciones):
    elegida = elegir(consola, PREGUNTA, opciones, indice_inicial=1,
                     lector_teclas=_lector("enter"))

    assert elegida.valor == "no"
    assert "❯ 2. No, cancelar" in _texto(consola)


def test_otra_tecla_no_mueve_el_cursor(consola, opciones):
    elegida = elegir(consola, PREGUNTA, opciones,
                     lector_teclas=_lector("otra", "abajo", "enter"))

    assert elegida.valor == "no"


def test_escape_devuelve_la_opcion_predeterminada(consola, opciones):
    elegida = elegir(consola, PREGUNTA, opciones, indice_inicial=2,
                     lector_teclas=_lector("escape"))

    assert isinstance(elegida, Opcion)
    assert elegida.valor == "no"
    assert elegida.predeterminada is True


def test_escape_dos_veces_es_estable(consola, opciones):
    primera = elegir(consola, PREGUNTA, opciones, lector_teclas=_lector("escape"))
    segunda = elegir(consola, PREGUNTA, opciones, lector_teclas=_lector("escape"))

    assert primera == segunda


def test_escape_sin_predeterminada_devuelve_la_ultima(consola):
    opciones = (Opcion("Sí", "si"), Opcion("No", "no"))

    elegida = elegir(consola, PREGUNTA, opciones, lector_teclas=_lector("escape"))

    assert isinstance(elegida, Opcion)
    assert elegida.valor == "no"


# --- Respaldo numerado -----------------------------------------------------


def test_sin_soporte_usa_el_respaldo_y_acepta_el_numero(consola, opciones, monkeypatch):
    prompts = _entrada_falsa(consola, monkeypatch, ["2"])

    elegida = elegir(consola, PREGUNTA, opciones,
                     lector_teclas=_lector("sin_soporte"))

    assert isinstance(elegida, Opcion)
    assert elegida.valor == "no"
    assert len(prompts) == 1
    texto = _texto(consola)
    assert "  1. Sí, guardar" in texto
    assert "  2. No, cancelar" in texto
    assert "  3. Volver" in texto
    assert "Pulsa Enter para «No, cancelar»." in texto
    # El listado numerado se imprime una sola vez.
    assert texto.count("  1. Sí, guardar") == 1


def test_sin_soporte_enter_vacio_elige_la_predeterminada(consola, opciones, monkeypatch):
    prompts = _entrada_falsa(consola, monkeypatch, [""])

    elegida = elegir(consola, PREGUNTA, opciones,
                     lector_teclas=_lector("sin_soporte"))

    assert elegida.valor == "no"
    assert len(prompts) == 1


def test_sin_soporte_enter_vacio_sin_predeterminada_elige_la_ultima(consola, monkeypatch):
    opciones = (Opcion("Sí", "si"), Opcion("No", "no"))
    _entrada_falsa(consola, monkeypatch, [""])

    elegida = elegir(consola, PREGUNTA, opciones,
                     lector_teclas=_lector("sin_soporte"))

    assert elegida.valor == "no"


def test_sin_soporte_acepta_la_inicial_de_la_opcion(consola, opciones, monkeypatch):
    _entrada_falsa(consola, monkeypatch, ["s"])

    elegida = elegir(consola, PREGUNTA, opciones,
                     lector_teclas=_lector("sin_soporte"))

    assert elegida.valor == "si"


def test_respuesta_invalida_reintenta_una_sola_vez(consola, opciones, monkeypatch):
    prompts = _entrada_falsa(consola, monkeypatch, ["9", "2"])

    elegida = elegir(consola, PREGUNTA, opciones,
                     lector_teclas=_lector("sin_soporte"))

    assert elegida.valor == "no"
    assert len(prompts) == 2
    texto = _texto(consola)
    assert texto.count("Escribe 1, 2 o 3, o Enter para «No, cancelar».") == 1
    # El listado no se repite al reintentar: las opciones ya están visibles.
    assert texto.count("  1. Sí, guardar") == 1


def test_respuesta_vacia_tras_un_error_tambien_es_valida(consola, opciones, monkeypatch):
    prompts = _entrada_falsa(consola, monkeypatch, ["9", ""])

    elegida = elegir(consola, PREGUNTA, opciones,
                     lector_teclas=_lector("sin_soporte"))

    assert elegida.valor == "no"
    assert len(prompts) == 2


def test_sin_soporte_con_eof_devuelve_la_predeterminada(consola, opciones, monkeypatch):
    def entrada_cerrada(prompt=""):
        raise EOFError

    monkeypatch.setattr(consola, "input", entrada_cerrada)

    elegida = elegir(consola, PREGUNTA, opciones,
                     lector_teclas=_lector("sin_soporte"))

    assert isinstance(elegida, Opcion)
    assert elegida.valor == "no"


def test_el_respaldo_lee_sin_un_live_abierto(consola, opciones, monkeypatch):
    """El `Live` debe estar detenido antes de leer una línea libre."""
    en_vivo = []

    def leer(prompt=""):
        en_vivo.append(getattr(consola, "_live", None))
        return "2"

    monkeypatch.setattr(consola, "input", leer)

    elegir(consola, PREGUNTA, opciones, lector_teclas=_lector("sin_soporte"))

    assert en_vivo == [None]


# --- Validación ------------------------------------------------------------


def test_elegir_exige_al_menos_dos_opciones(consola):
    with pytest.raises(ValueError, match="dos opciones"):
        elegir(consola, PREGUNTA, (Opcion("Sí", "si"),))


def test_elegir_exige_valores_unicos(consola):
    opciones = (Opcion("Sí", "si"), Opcion("No", "si"))

    with pytest.raises(ValueError, match="repetid"):
        elegir(consola, PREGUNTA, opciones)


@pytest.mark.parametrize("indice", (-1, 3))
def test_elegir_valida_el_indice_inicial(consola, opciones, indice):
    with pytest.raises(ValueError, match="índice"):
        elegir(consola, PREGUNTA, opciones, indice_inicial=indice)


def test_la_validacion_ocurre_antes_de_leer_teclas(consola, opciones):
    def bomba():
        raise _Bomba("no debe leer teclas si los argumentos son inválidos")

    with pytest.raises(ValueError):
        elegir(consola, PREGUNTA, opciones, indice_inicial=9,
               lector_teclas=bomba)


# --- Clasificación de bytes (pura, sin terminal) ---------------------------


@pytest.mark.parametrize(
    "secuencia, esperada",
    (
        # Windows: prefijo extendido 0xE0 o 0x00 + código de flecha.
        (b"\xe0H", "arriba"),
        ("\xe0H", "arriba"),
        (b"\x00H", "arriba"),
        (b"\xe0P", "abajo"),
        (b"\x00P", "abajo"),
        # K/M son flechas izquierda/derecha: el menú es vertical.
        (b"\xe0K", "otra"),
        (b"\x00K", "otra"),
        (b"\xe0M", "otra"),
        (b"\x00M", "otra"),
        # POSIX: ESC [ A/B (y la variante ESC O A/B de algunos terminales).
        (b"\x1b[A", "arriba"),
        (b"\x1b[B", "abajo"),
        (b"\x1bOA", "arriba"),
        (b"\x1bOB", "abajo"),
        (b"\x1b[C", "otra"),
        (b"\x1b[D", "otra"),
        # Enter y Escape.
        ("\r", "enter"),
        ("\n", "enter"),
        (b"\r", "enter"),
        (b"\n", "enter"),
        ("\x1b", "escape"),
        (b"\x1b", "escape"),
        # Ctrl+C en modo crudo tambi\u00e9n cancela, igual que Escape.
        (b"\x03", "escape"),
        ("\x03", "escape"),
        # Una tecla cualquiera no mueve ni confirma.
        (b"x", "otra"),
        ("", "otra"),
    ),
)
def test_clasificar_secuencia(secuencia, esperada):
    assert seleccion._clasificar_secuencia(secuencia) == esperada


# --- `leer_tecla`: degradación sin terminal --------------------------------


def test_sin_tty_no_intenta_leer_teclas_crudas(monkeypatch):
    monkeypatch.setattr(sys, "stdin", _StdinFalso(es_tty=False))

    def bomba():
        raise _Bomba("no debe intentar modo crudo sin TTY")

    monkeypatch.setattr(seleccion, "_leer_tecla_cruda", bomba)

    assert leer_tecla() == "sin_soporte"


@pytest.mark.parametrize("error", (OSError("consola rota"), ImportError("sin módulo")))
def test_los_fallos_del_terminal_degradan_a_sin_soporte(monkeypatch, error):
    monkeypatch.setattr(sys, "stdin", _StdinFalso(es_tty=True))

    def falla():
        raise error

    monkeypatch.setattr(seleccion, "_leer_tecla_cruda", falla)

    assert leer_tecla() == "sin_soporte"


def test_un_tty_real_usa_el_lector_crudo(monkeypatch):
    monkeypatch.setattr(sys, "stdin", _StdinFalso(es_tty=True))
    monkeypatch.setattr(seleccion, "_leer_tecla_cruda", lambda: "enter")

    assert leer_tecla() == "enter"


def test_elegir_cae_al_respaldo_cuando_stdin_no_es_tty(consola, opciones, monkeypatch):
    monkeypatch.setattr(sys, "stdin", _StdinFalso(es_tty=False))
    monkeypatch.setattr(
        seleccion, "_leer_tecla_cruda",
        lambda: (_ for _ in ()).throw(_Bomba("modo crudo sin TTY")),
    )
    _entrada_falsa(consola, monkeypatch, ["2"])

    # Sin lector inyectado: el elegir real consulta leer_tecla().
    elegida = elegir(consola, PREGUNTA, opciones)

    assert isinstance(elegida, Opcion)
    assert elegida.valor == "no"
    assert "  2. No, cancelar" in _texto(consola)


# --- Camino POSIX: escape suelto, flecha y restauración -------------------


def test_posix_escape_suelto_no_bloquea_y_restaura(monkeypatch):
    eventos = _instalar_terminal_posix(monkeypatch, hay_mas=False)
    monkeypatch.setattr(sys, "stdin", _StdinConDescriptor(es_tty=True))
    monkeypatch.setattr(os, "read", lambda descriptor, cantidad: b"\x1b")

    assert seleccion._leer_tecla_posix() == "escape"

    # `select` con plazo corto: no se espera el segundo byte de una flecha.
    assert ("select", 0.05) in eventos
    # Pase lo que pase, el terminal vuelve a su estado anterior.
    assert eventos[-1] == ("set", 7, 1, "atributos")


def test_posix_lee_una_flecha_como_secuencia(monkeypatch):
    eventos = _instalar_terminal_posix(monkeypatch, hay_mas=True)
    monkeypatch.setattr(sys, "stdin", _StdinConDescriptor(es_tty=True))
    partes = iter((b"\x1b", b"[", b"B"))
    monkeypatch.setattr(os, "read", lambda descriptor, cantidad: next(partes))

    assert seleccion._leer_tecla_posix() == "abajo"
    assert eventos[-1] == ("set", 7, 1, "atributos")


def test_posix_restaura_el_terminal_si_falla_la_lectura(monkeypatch):
    eventos = _instalar_terminal_posix(monkeypatch, hay_mas=False)
    monkeypatch.setattr(sys, "stdin", _StdinConDescriptor(es_tty=True))

    def falla(descriptor, cantidad):
        raise OSError("consola rota")

    monkeypatch.setattr(os, "read", falla)

    with pytest.raises(OSError):
        seleccion._leer_tecla_posix()

    assert ("raw", 7) in eventos
    assert eventos[-1] == ("set", 7, 1, "atributos")
