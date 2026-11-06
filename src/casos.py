"""Repositorio SQLite de casos de soporte de TecnoApoyo (RA2 / IL2.1).

Los casos son el dominio de escritura del agente: se crean en la recepción,
reciben notas, cambian de estado y se escalan. Este módulo es la única puerta
de acceso a esos datos, y lo hace con sentencias parametrizadas: no existe
ningún camino para ejecutar SQL construido a partir de texto del usuario.

Decisiones de diseño:
  - Una conexión por operación. Los tests abren y cierran el repositorio varias
    veces para simular reinicios del proceso, y una conexión compartida a nivel
    de módulo dejaría el archivo bloqueado en Windows.
  - Los estados y las prioridades vienen de una allowlist. Un estado libre
    rompería los filtros y los informes sin que nadie lo note.
  - Los ids son secuenciales y estables (CASO-0001). El agente cita casos por
    id, así que un id reciclado apuntaría a otro caso en un turno posterior.
  - Las notas viven en su propia tabla y no como JSON dentro de la fila del
    caso: una nota nueva es un INSERT, no una lectura-modificación-escritura
    del campo completo, y no existe un "JSON de notas" que pueda quedar
    corrupto.
"""
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date
from pathlib import Path

from src import config

ESTADOS_VALIDOS = ("abierto", "en_progreso", "escalado", "cerrado")
PRIORIDADES_VALIDAS = ("baja", "media", "alta")

ESQUEMA = """
CREATE TABLE IF NOT EXISTS casos (
    id              TEXT PRIMARY KEY,
    cliente         TEXT NOT NULL,
    categoria       TEXT NOT NULL,
    descripcion     TEXT NOT NULL,
    prioridad       TEXT NOT NULL,
    estado          TEXT NOT NULL,
    procedimiento   TEXT,
    fecha_creacion  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS casos_notas (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    id_caso  TEXT NOT NULL,
    nota     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_casos_estado ON casos(estado);
CREATE INDEX IF NOT EXISTS idx_notas_caso ON casos_notas(id_caso);
"""


class ErrorCaso(Exception):
    """Fallo de dominio con un código estable para la capa de herramientas.

    La herramienta devuelve `error` como texto, pero el código permite
    distinguir un argumento inválido (`validacion`) de un caso inexistente
    (`no_encontrado`) sin analizar mensajes.
    """

    def __init__(self, mensaje: str, codigo: str = "validacion"):
        super().__init__(mensaje)
        self.codigo = codigo


def _texto_obligatorio(valor: str, campo: str) -> str:
    texto = (valor or "").strip()
    if not texto:
        raise ErrorCaso(f"El campo '{campo}' no puede venir vacío.", "validacion")
    return texto


def _validar_estado(estado: str) -> str:
    normalizado = (estado or "").strip().lower()
    if normalizado not in ESTADOS_VALIDOS:
        raise ErrorCaso(
            f"Estado '{estado}' no permitido. Valores válidos: "
            f"{', '.join(ESTADOS_VALIDOS)}.",
            "estado_invalido",
        )
    return normalizado


def _validar_prioridad(prioridad: str) -> str:
    normalizado = (prioridad or "").strip().lower()
    if normalizado not in PRIORIDADES_VALIDAS:
        raise ErrorCaso(
            f"Prioridad '{prioridad}' no permitida. Valores válidos: "
            f"{', '.join(PRIORIDADES_VALIDAS)}.",
            "prioridad_invalida",
        )
    return normalizado


class RepositorioCasos:
    """Acceso a los casos persistidos en SQLite."""

    def __init__(self, ruta=None, semilla=None, sembrar: bool = True):
        self.ruta = Path(ruta) if ruta else config.RUTA_CASOS
        self.ruta.parent.mkdir(parents=True, exist_ok=True)
        self._crear_esquema()
        self.sembrados = 0
        if sembrar:
            ruta_semilla = Path(semilla) if semilla else config.RUTA_CASOS_SEMILLA
            self.sembrados = self.sembrar_desde_archivo(ruta_semilla)

    # -- Infraestructura --------------------------------------------------
    @contextmanager
    def _conexion(self) -> Iterator[sqlite3.Connection]:
        conexion = sqlite3.connect(self.ruta)
        conexion.row_factory = sqlite3.Row
        try:
            yield conexion
            conexion.commit()
        finally:
            conexion.close()

    def _crear_esquema(self) -> None:
        with self._conexion() as conexion:
            conexion.executescript(ESQUEMA)

    @staticmethod
    def _notas_por_caso(conexion: sqlite3.Connection) -> dict[str, list[str]]:
        """Todas las notas agrupadas por caso, en orden de inserción."""
        agrupadas: dict[str, list[str]] = {}
        for fila in conexion.execute(
            "SELECT id_caso, nota FROM casos_notas ORDER BY id"
        ).fetchall():
            agrupadas.setdefault(fila["id_caso"], []).append(fila["nota"])
        return agrupadas

    def _serializar(self, fila: sqlite3.Row,
                    notas: dict[str, list[str]]) -> dict:
        caso = dict(fila)
        caso["notas"] = notas.get(caso["id"], [])
        return caso

    def _siguiente_id(self, conexion: sqlite3.Connection) -> str:
        fila = conexion.execute(
            "SELECT id FROM casos ORDER BY id DESC LIMIT 1"
        ).fetchone()
        ultimo = 0
        if fila is not None:
            try:
                ultimo = int(str(fila["id"]).split("-")[1])
            except (IndexError, ValueError):
                # Un id con formato ajeno no debe bloquear la creación de casos
                # nuevos: se continúa desde el conteo existente.
                ultimo = conexion.execute("SELECT COUNT(*) FROM casos").fetchone()[0]
        return f"CASO-{ultimo + 1:04d}"

    # -- Semilla ----------------------------------------------------------
    def sembrar_desde_archivo(self, ruta: Path) -> int:
        """Carga la semilla versionada. Idempotente: no duplica casos.

        Devuelve cuántas filas se insertaron realmente (0 en un segundo
        arranque sobre la misma base). Un estado inválido en la semilla aborta
        la carga completa: la semilla es código versionado, no entrada de
        usuario, así que un valor raro es un defecto que debe verse.
        """
        import json

        ruta = Path(ruta)
        if not ruta.exists():
            return 0
        try:
            contenido = json.loads(ruta.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ErrorCaso(
                f"La semilla {ruta.name} no es JSON válido: {exc}", "semilla_invalida"
            ) from exc

        insertados = 0
        with self._conexion() as conexion:
            for caso in contenido.get("casos", []):
                estado = _validar_estado(caso.get("estado", "abierto"))
                prioridad = _validar_prioridad(caso.get("prioridad", "media"))
                cursor = conexion.execute(
                    """INSERT OR IGNORE INTO casos
                       (id, cliente, categoria, descripcion, prioridad, estado,
                        procedimiento, fecha_creacion)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        _texto_obligatorio(caso.get("id"), "id"),
                        _texto_obligatorio(caso.get("cliente"), "cliente"),
                        _texto_obligatorio(caso.get("categoria"), "categoria"),
                        _texto_obligatorio(caso.get("descripcion"), "descripcion"),
                        prioridad,
                        estado,
                        caso.get("procedimiento"),
                        _texto_obligatorio(caso.get("fecha_creacion"), "fecha_creacion"),
                    ),
                )
                if cursor.rowcount == 0:
                    continue
                insertados += 1
                for nota in caso.get("notas", []):
                    conexion.execute(
                        "INSERT INTO casos_notas (id_caso, nota) VALUES (?, ?)",
                        (caso["id"], _texto_obligatorio(nota, "nota")),
                    )
        return insertados

    # -- Lectura ----------------------------------------------------------
    def contar(self) -> int:
        with self._conexion() as conexion:
            return conexion.execute("SELECT COUNT(*) FROM casos").fetchone()[0]

    def obtener_caso(self, id_caso: str) -> dict:
        id_normalizado = _texto_obligatorio(id_caso, "id_caso").upper()
        with self._conexion() as conexion:
            fila = conexion.execute(
                "SELECT * FROM casos WHERE id = ?", (id_normalizado,)
            ).fetchone()
            if fila is None:
                raise ErrorCaso(
                    f"No existe el caso '{id_normalizado}'.", "no_encontrado"
                )
            notas = self._notas_por_caso(conexion)
        return self._serializar(fila, notas)

    def listar_casos(self, estado: str | None = None,
                     limite: int = 20) -> list[dict]:
        limite = max(1, min(int(limite), 100))
        # Dos sentencias literales en vez de una construida por concatenación:
        # así ningún valor de entrada forma parte del texto SQL.
        with self._conexion() as conexion:
            if estado is None:
                filas = conexion.execute(
                    "SELECT * FROM casos ORDER BY id LIMIT ?", (limite,)
                ).fetchall()
            else:
                filas = conexion.execute(
                    "SELECT * FROM casos WHERE estado = ? ORDER BY id LIMIT ?",
                    (_validar_estado(estado), limite),
                ).fetchall()
            notas = self._notas_por_caso(conexion)
        return [self._serializar(fila, notas) for fila in filas]

    # -- Escritura --------------------------------------------------------
    def crear_caso(self, cliente: str, categoria: str, descripcion: str,
                   prioridad: str = "media", estado: str = "abierto",
                   procedimiento: str | None = None,
                   fecha_creacion: str = "") -> dict:
        datos = (
            _texto_obligatorio(cliente, "cliente"),
            _texto_obligatorio(categoria, "categoria"),
            _texto_obligatorio(descripcion, "descripcion"),
            _validar_prioridad(prioridad),
            _validar_estado(estado),
            (procedimiento or "").strip() or None,
            fecha_creacion or date.today().isoformat(),
        )
        with self._conexion() as conexion:
            id_caso = self._siguiente_id(conexion)
            conexion.execute(
                """INSERT INTO casos
                   (id, cliente, categoria, descripcion, prioridad, estado,
                    procedimiento, fecha_creacion)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (id_caso, *datos),
            )
        return self.obtener_caso(id_caso)

    def agregar_nota(self, id_caso: str, nota: str) -> dict:
        nota_limpia = _texto_obligatorio(nota, "nota")
        caso = self.obtener_caso(id_caso)
        with self._conexion() as conexion:
            conexion.execute(
                "INSERT INTO casos_notas (id_caso, nota) VALUES (?, ?)",
                (caso["id"], nota_limpia),
            )
        return self.obtener_caso(caso["id"])

    def actualizar_estado(self, id_caso: str, estado: str) -> dict:
        estado_validado = _validar_estado(estado)
        caso = self.obtener_caso(id_caso)
        with self._conexion() as conexion:
            conexion.execute(
                "UPDATE casos SET estado = ? WHERE id = ?",
                (estado_validado, caso["id"]),
            )
        return self.obtener_caso(caso["id"])

    def escalar(self, id_caso: str, motivo: str) -> dict:
        motivo_limpio = _texto_obligatorio(motivo, "motivo")
        caso = self.obtener_caso(id_caso)
        with self._conexion() as conexion:
            conexion.execute(
                "UPDATE casos SET estado = ? WHERE id = ?", ("escalado", caso["id"])
            )
            # El motivo queda como nota: la escalada sin justificación no sirve
            # en la auditoría y quien retome el caso necesita el contexto.
            conexion.execute(
                "INSERT INTO casos_notas (id_caso, nota) VALUES (?, ?)",
                (caso["id"], f"Escalado: {motivo_limpio}"),
            )
        return self.obtener_caso(caso["id"])
