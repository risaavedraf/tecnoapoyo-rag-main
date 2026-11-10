"""Menú de selección con flechas para la consola del asistente.

El asistente pide confirmaciones de escritura y el mensaje visible habla de
«s/n». `rich.prompt.Confirm` asume «y/n», así que una «s» válida se rechaza y
el bucle de reintentos parece un cuelgue. Este módulo reemplaza esa pregunta
por un menú de flechas con cursor visible y un respaldo numerado.

Dos caminos de entrada:

  1. Teclas crudas (`leer_tecla`): flechas, Enter y Escape, sin escribir. En
     Windows se lee con `msvcrt`; en POSIX se pone `sys.stdin` en modo crudo
     con `termios`/`tty` y se restaura en un `finally`.
  2. Respaldo numerado (`_elegir_numerado`): se usa cuando el lector devuelve
     `sin_soporte`. Las consolas de Windows y las entradas redirigidas no
     siempre entregan caracteres de control; bloquearse esperándolos dejaría
     al usuario sin forma de responder. El respaldo imprime las opciones una
     sola vez y lee una línea con `console.input`, la misma ruta que ya
     funciona con cualquier terminal.

El respaldo no usa `rich.live.Live`: mientras se lee una línea libre no hay
que repintar nada y un `Live` abierto competiría con el eco de la entrada.
"""
import os
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from rich.console import Console
from rich.live import Live
from rich.text import Text

from src.texto import sin_tildes

ARRIBA = "arriba"
ABAJO = "abajo"
ENTER = "enter"
ESCAPE = "escape"
OTRA = "otra"
SIN_SOPORTE = "sin_soporte"

# Prefijos de tecla extendida que entrega `msvcrt.getwch` en Windows.
_PREFIJOS_EXTENDIDOS = ("\xe0", "\x00")


@dataclass(frozen=True)
class Opcion:
    """Una opción del menú.

    `predeterminada` marca la opción segura: la eligen Escape y el Enter vacío
    del respaldo numerado. Si ninguna la marca, se usa la última del menú.
    """

    etiqueta: str
    valor: str
    descripcion: str = ""
    predeterminada: bool = False


def _clasificar_secuencia(secuencia: bytes | str) -> str:
    """Traduce una secuencia de bytes (o su equivalente str) a una tecla.

    Es pura y no toca el terminal, así que se puede probar con las secuencias
    documentadas de Windows y de POSIX sin abrir una consola real.
    """
    if isinstance(secuencia, str):
        crudo = secuencia.encode("latin-1", errors="replace")
    else:
        crudo = secuencia

    if crudo in (b"\r", b"\n", b"\r\n"):
        return ENTER
    if crudo in (b"\x1b", b"\x03"):
        # Ctrl+C en modo crudo no genera KeyboardInterrupt: se trata como
        # Escape para que el menú se pueda cancelar siempre.
        return ESCAPE
    if crudo in (b"\x1b[A", b"\x1bOA", b"\xe0H", b"\x00H"):
        return ARRIBA
    if crudo in (b"\x1b[B", b"\x1bOB", b"\xe0P", b"\x00P"):
        return ABAJO
    # K/M (izquierda/derecha) y cualquier otra tecla no mueven el cursor.
    return OTRA


def _leer_tecla_windows() -> str:
    """Lee una tecla con `msvcrt` y traduce las flechas extendidas."""
    import msvcrt

    primera = msvcrt.getwch()
    if primera in _PREFIJOS_EXTENDIDOS:
        return _clasificar_secuencia(primera + msvcrt.getwch())
    return _clasificar_secuencia(primera)


def _leer_tecla_posix() -> str:
    """Lee una tecla en modo crudo y restaura el terminal en `finally`."""
    import select
    import termios
    import tty

    descriptor = sys.stdin.fileno()
    # Los módulos `termios`/`tty` no existen en Windows: el analizador de tipos
    # los marca como desconocidos, aunque este camino solo corre en POSIX.
    anteriores = termios.tcgetattr(descriptor)  # type: ignore[attr-defined]
    try:
        tty.setraw(descriptor)  # type: ignore[attr-defined]
        primero = os.read(descriptor, 1)
        if not primero:
            # stdin cerrado: no hay teclas que leer ni forma de reintentar.
            return SIN_SOPORTE
        if primero != b"\x1b":
            return _clasificar_secuencia(primero)
        # Un Escape suelto y el inicio de una flecha comparten el primer byte.
        # Se espera un instante por los bytes siguientes en vez de bloquear a
        # quien solo quiso cancelar.
        resto = b""
        while len(resto) < 2 and select.select([descriptor], [], [], 0.05)[0]:
            resto += os.read(descriptor, 1)
        return _clasificar_secuencia(primero + resto)
    finally:
        termios.tcsetattr(  # type: ignore[attr-defined]
            descriptor, termios.TCSADRAIN, anteriores  # type: ignore[attr-defined]
        )


def _leer_tecla_cruda() -> str:
    """Despacha al lector de la plataforma actual."""
    if os.name == "nt":
        return _leer_tecla_windows()
    return _leer_tecla_posix()


def leer_tecla() -> str:
    """Devuelve la tecla pulsada o `"sin_soporte"` si no hay teclas crudas.

    `"sin_soporte"` cubre los tres casos en que el llamador debe usar el
    respaldo numerado: stdin no es un terminal, la plataforma no tiene módulo
    de teclas o la consola falló al leer.
    """
    try:
        if not sys.stdin.isatty():
            return SIN_SOPORTE
    except (AttributeError, OSError, ValueError):
        return SIN_SOPORTE

    try:
        return _leer_tecla_cruda()
    except Exception:
        # Cualquier fallo del terminal (OSError, ImportError, termios.error)
        # debe terminar en el respaldo numerado, nunca en un traceback ni en
        # una espera infinita. KeyboardInterrupt y SystemExit no son Exception
        # y siguen propagándose.
        return SIN_SOPORTE


def _validar(opciones: tuple[Opcion, ...], indice_inicial: int) -> None:
    """Rechaza menús ambiguos antes de tocar el terminal."""
    if len(opciones) < 2:
        raise ValueError("El menú necesita al menos dos opciones.")
    valores = [opcion.valor for opcion in opciones]
    repetidos = sorted({valor for valor in valores if valores.count(valor) > 1})
    if repetidos:
        raise ValueError(
            "Hay valores repetidos en el menú: " + ", ".join(repetidos) + "."
        )
    if not 0 <= indice_inicial < len(opciones):
        raise ValueError(
            f"El índice inicial {indice_inicial} está fuera de rango "
            f"(0..{len(opciones) - 1})."
        )


def _indice_predeterminado(opciones: tuple[Opcion, ...]) -> int:
    """Devuelve la opción segura: la marcada o, si no hay, la última."""
    for posicion, opcion in enumerate(opciones):
        if opcion.predeterminada:
            return posicion
    return len(opciones) - 1


def _opcion_predeterminada(opciones: tuple[Opcion, ...]) -> Opcion:
    """Igual que `_indice_predeterminado`, pero devuelve la opción."""
    return opciones[_indice_predeterminado(opciones)]


def _renderizar(
    pregunta: str, opciones: tuple[Opcion, ...], indice: int
) -> Text:
    """Arma el cuadro del menú: pregunta, lista con cursor y descripción."""
    predeterminada = _opcion_predeterminada(opciones)
    lineas: list[tuple[str, str]] = [(pregunta, "bold")]
    for posicion, opcion in enumerate(opciones):
        elegida = posicion == indice
        cursor = "❯" if elegida else " "
        estilo = "bold reverse cyan" if elegida else ""
        lineas.append((f"{cursor} {posicion + 1}. {opcion.etiqueta}", estilo))
    if opciones[indice].descripcion:
        lineas.append(("", ""))
        lineas.append((f"  {opciones[indice].descripcion}", "dim"))
    lineas.append(
        (f"  ↑/↓ mover · Enter elegir · Esc «{predeterminada.etiqueta}»", "dim")
    )
    return Text("\n").join(Text(texto, style=estilo) for texto, estilo in lineas)


def _normalizar(texto: str) -> str:
    """Minúsculas sin tildes, para comparar iniciales tecleadas."""
    return sin_tildes(texto).casefold()


def _iniciales(opcion: Opcion) -> set[str]:
    """Iniciales aceptadas para una opción: etiqueta y valor."""
    iniciales = set()
    for texto in (opcion.etiqueta, opcion.valor):
        normalizado = _normalizar(texto).strip()
        if normalizado:
            iniciales.add(normalizado[0])
    return iniciales


def _interpretar(
    respuesta: str,
    opciones: tuple[Opcion, ...],
    predeterminada: Opcion,
) -> Opcion | None:
    """Convierte la línea tecleada en una opción, o `None` si no es válida."""
    limpio = (respuesta or "").strip()
    if not limpio:
        return predeterminada
    if limpio.isdigit():
        try:
            numero = int(limpio)
        except ValueError:
            return None
        if 1 <= numero <= len(opciones):
            return opciones[numero - 1]
        return None
    inicial = _normalizar(limpio)
    if len(inicial) == 1:
        # Una inicial compartida es ambigua: elegir la primera resolvería por
        # orden de la lista una decisión que la persona no tomó.
        coincidencias = [
            opcion for opcion in opciones if inicial in _iniciales(opcion)
        ]
        if len(coincidencias) == 1:
            return coincidencias[0]
    return None


def _enumeracion(cantidad: int) -> str:
    """Enumera los números disponibles: «1», «1 o 2», «1, 2 o 3»."""
    numeros = [str(numero) for numero in range(1, cantidad + 1)]
    if len(numeros) == 1:
        return numeros[0]
    return ", ".join(numeros[:-1]) + " o " + numeros[-1]


def _elegir_numerado(
    consola: Console,
    pregunta: str,
    opciones: tuple[Opcion, ...],
) -> Opcion:
    """Respaldo sin teclas crudas: lista numerada y una línea de entrada.

    Las opciones se imprimen una sola vez; un reintento solo repite la
    instrucción, porque la lista sigue visible arriba. Enter vacío elige la
    opción predeterminada, la misma que elige Escape en el menú de flechas.
    """
    predeterminada = _opcion_predeterminada(opciones)
    consola.print(Text(pregunta, style="bold"))
    for posicion, opcion in enumerate(opciones, start=1):
        linea = Text()
        linea.append(f"  {posicion}. ", style="bold")
        linea.append(opcion.etiqueta)
        consola.print(linea)
    consola.print(
        Text(f"  Pulsa Enter para «{predeterminada.etiqueta}».", style="dim")
    )

    while True:
        try:
            respuesta = consola.input("> ")
        except (EOFError, OSError):
            # Sin entrada disponible no se puede reintentar: se devuelve la
            # opción segura en vez de dejar el proceso esperando.
            return predeterminada
        elegida = _interpretar(respuesta, opciones, predeterminada)
        if elegida is not None:
            return elegida
        consola.print(
            Text(
                f"Escribe {_enumeracion(len(opciones))}, o Enter para "
                f"«{predeterminada.etiqueta}».",
                style="yellow",
            )
        )


def _recorrer(
    consola: Console,
    pregunta: str,
    opciones: tuple[Opcion, ...],
    indice_inicial: int,
    lector: Callable[[], str],
) -> Opcion:
    """Bucle del menú de flechas; usa `Live` solo mientras hay teclas."""
    indice = indice_inicial
    vivo = Live(
        _renderizar(pregunta, opciones, indice),
        console=consola,
        auto_refresh=False,
        transient=False,
    )
    vivo.start(refresh=True)
    try:
        while True:
            tecla = lector()
            if tecla == SIN_SOPORTE:
                # Se corta el `Live` en el `finally` antes de leer una línea
                # libre. El marco a medias se borra (`transient`) para que el
                # respaldo imprima su lista numerada sin texto duplicado.
                vivo.transient = True
                break
            if tecla == ARRIBA:
                indice = (indice - 1) % len(opciones)
            elif tecla == ABAJO:
                indice = (indice + 1) % len(opciones)
            elif tecla == ENTER:
                return opciones[indice]
            elif tecla == ESCAPE:
                return _opcion_predeterminada(opciones)
            vivo.update(_renderizar(pregunta, opciones, indice), refresh=True)
    finally:
        vivo.stop()
    return _elegir_numerado(consola, pregunta, opciones)


def elegir(
    console: Console,
    pregunta: str,
    opciones: Iterable[Opcion],
    indice_inicial: int = 0,
    lector_teclas: Callable[[], str] | None = None,
) -> Opcion:
    """Muestra el menú y devuelve la opción elegida, siempre un `Opcion`.

    Las flechas mueven con vuelta al principio; Enter confirma; Escape elige
    la opción marcada como predeterminada (o la última si no hay ninguna).
    Si no hay teclas crudas disponibles, cae al respaldo numerado sin dejar
    un `Live` abierto mientras lee la línea.
    """
    lista = tuple(opciones)
    _validar(lista, indice_inicial)
    lector = lector_teclas or leer_tecla
    return _recorrer(console, pregunta, lista, indice_inicial, lector)
