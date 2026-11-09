"""Infraestructura SQLite compartida por los repositorios de TecnoApoyo.

`src/casos.py` (casos y notas) y `src/memoria.py` (hechos, sesiones y turnos)
guardan sus datos con las mismas decisiones: una conexión por operación, un
esquema idempotente, commit al salir del bloque y un identificador secuencial
legible (`PREFIJO-0001`). Este módulo es la única definición de esas piezas: un
arreglo en la política de conexión, en el commit o en el generador de ids llega
a los dos repositorios a la vez, y no quedan dos copias que puedan discrepar.

Decisiones de diseño:
  - El error base es uno solo por una razón de contrato: cada repositorio
    expone su propio tipo (`ErrorCaso`, `ErrorMemoria`) como subclase, así el
    llamador sigue capturando el error de su dominio, mientras el texto y el
    `codigo` (`validacion`, `no_encontrado`, …) tienen una sola definición.
  - La preparación de la conexión es un gancho, no una regla fija. Casos
    necesita registrar una colación de nombres en cada conexión y memoria no la
    necesita: el módulo compartido no impone ninguna, la recibe quien la usa.
  - El generador de ids recibe tabla, prefijo y columna como parámetros. Un
    prefijo o una tabla fijos habrían obligado a mantener dos copias del mismo
    algoritmo, que es justo lo que este módulo elimina.
"""
import re
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

# Un nombre de tabla o de columna interpolado en el SQL tiene que ser un
# identificador simple: los repositorios pasan constantes del módulo, nunca
# texto del usuario, y esta comprobación lo deja dicho de forma ejecutable.
IDENTIFICADOR_VALIDO = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class ErrorPersistencia(Exception):
    """Fallo de dominio con un código estable, compartido por los repositorios.

    El texto se muestra tal cual, pero el código permite distinguir un
    argumento inválido (`validacion`) de una fila inexistente (`no_encontrado`)
    sin analizar mensajes. Los repositorios lo especializan para que cada capa
    capture su propio tipo y no pueda confundir un error con el del otro.
    """

    def __init__(self, mensaje: str, codigo: str = "validacion"):
        super().__init__(mensaje)
        self.codigo = codigo


def texto_obligatorio(
    valor: str,
    campo: str,
    error: type[ErrorPersistencia] = ErrorPersistencia,
) -> str:
    """Devuelve el texto sin espacios en los extremos o rechaza el vacío.

    `error` es la clase a lanzar: el mensaje es el mismo para todos los
    repositorios, pero el tipo es el de cada dominio (`ErrorCaso`,
    `ErrorMemoria`), así una herramienta puede capturar solo el suyo.
    """
    texto = (valor or "").strip()
    if not texto:
        raise error(f"El campo '{campo}' no puede venir vacío.", "validacion")
    return texto


@contextmanager
def conexion(
    ruta: str | Path,
    preparar: Callable[[sqlite3.Connection], None] | None = None,
) -> Iterator[sqlite3.Connection]:
    """Conexión por operación: abre, prepara, entrega y confirma al terminar.

    Una conexión por operación y no una compartida a nivel de módulo: los tests
    abren y cierran el repositorio para simular reinicios del proceso, y una
    conexión viva a nivel de módulo dejaría el archivo bloqueado en Windows. El
    `commit` llega cuando el bloque del llamador termina sin excepción; el
    `close` ocurre siempre.

    `preparar` es el gancho de cada repositorio (por ejemplo, registrar la
    colación de nombres de cliente) y se ejecuta antes de entregar la conexión:
    ninguna sentencia puede nombrar una función que esa conexión no registró.
    """
    abierta = sqlite3.connect(ruta)
    abierta.row_factory = sqlite3.Row
    try:
        if preparar is not None:
            preparar(abierta)
        yield abierta
        abierta.commit()
    finally:
        abierta.close()


def crear_esquema(
    ruta: str | Path,
    esquema: str,
    preparar: Callable[[sqlite3.Connection], None] | None = None,
) -> None:
    """Crea el esquema si falta y confirma el cambio.

    `esquema` es idempotente (`CREATE TABLE IF NOT EXISTS`), así que abrir un
    repositorio sobre una base existente no toca los datos ya guardados.
    """
    with conexion(ruta, preparar=preparar) as abierta:
        abierta.executescript(esquema)


def registrar_colacion(
    abierta: sqlite3.Connection,
    nombre: str,
    clave: Callable[[str], str],
) -> None:
    """Registra en `abierta` una colación que compara dos textos por `clave`.

    SQLite no persiste las colaciones definidas por la aplicación: viven en la
    conexión y hay que registrarlas en cada una antes de la primera sentencia
    que las nombre. `clave` normaliza el texto (tildes, mayúsculas, espacios de
    más) y la colación compara esas claves; así el orden de SQLite y la
    igualdad del código usan un único criterio.
    """
    def comparar(primero: str, segundo: str) -> int:
        izquierda, derecha = clave(primero), clave(segundo)
        return (izquierda > derecha) - (izquierda < derecha)

    abierta.create_collation(nombre, comparar)


def siguiente_id(abierta: sqlite3.Connection, tabla: str, prefijo: str,
                 columna: str = "id") -> str:
    """Siguiente identificador secuencial de `tabla`, con formato `PREFIJO-0001`.

    Se toma el último valor por orden descendente de la columna y se le suma
    uno. Si esa fila trae un id con otro formato (una base manipulada a mano),
    se continúa desde el conteo existente en vez de fallar: un id ajeno no debe
    bloquear la creación de filas nuevas.

    `tabla` y `columna` entran en la sentencia por interpolación, así que se
    validan como identificadores simples; sus valores son constantes de cada
    repositorio, nunca texto del usuario.
    """
    tabla = _identificador(tabla)
    columna = _identificador(columna)
    fila = abierta.execute(
        f"SELECT {columna} FROM {tabla} ORDER BY {columna} DESC LIMIT 1"
    ).fetchone()
    ultimo = 0
    if fila is not None:
        try:
            ultimo = int(str(fila[columna]).split("-")[1])
        except (IndexError, ValueError):
            ultimo = abierta.execute(f"SELECT COUNT(*) FROM {tabla}").fetchone()[0]
    return f"{prefijo}-{ultimo + 1:04d}"


def _identificador(nombre: str) -> str:
    """Acepta solo nombres de tabla o de columna simples, sin SQL inyectable."""
    if not IDENTIFICADOR_VALIDO.match(nombre or ""):
        raise ValueError(f"Identificador SQL no permitido: {nombre!r}.")
    return nombre
