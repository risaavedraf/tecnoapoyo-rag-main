"""Memoria del agente de TecnoApoyo: corto y largo plazo (RA2 / IL2.2).

La memoria de corto plazo es el historial que viaja dentro del prompt; la de
largo plazo son hechos del usuario y sesiones persistidas en SQLite. Este
módulo no habla con el proveedor del LLM: la compresión del historial se
inyecta como función, así que la estrategia `resumen` se puede verificar sin
red y el motor RAG sigue funcionando con el modelo apagado.

Decisiones de diseño:
  - Una conexión por operación, igual que en `src/casos.py`. Los tests abren y
    cierran el repositorio para simular reinicios del proceso, y una conexión
    compartida a nivel de módulo dejaría el archivo bloqueado en Windows.
  - Solo sentencias literales parametrizadas: ningún texto del usuario forma
    parte del SQL.
  - Las estrategias de corto plazo vienen de una allowlist. Una estrategia
    desconocida dejaría el historial sin recorte o sin resumen sin que nadie lo
    note, así que se rechaza con un código de dominio.
  - Los ids de sesión son secuenciales y estables (SES-0001): el usuario retoma
    una conversación citándola por id, y un id reciclado apuntaría a otra
    sesión.
  - El resumen por defecto no llama al LLM: concatena "rol: contenido" y
    conserva la cola más reciente por debajo de un tope de caracteres. Un
    resumen que crece sin límite terminaría desplazando al contexto recuperado.
  - Los turnos no llevan columna de id propia: se ordenan por el rowid
    implícito de SQLite, que preserva el orden de inserción sin agregar una
    columna que el contrato de datos no declara.
"""
import re
import sqlite3
import unicodedata
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from src import config

# Espejo de config.ESTRATEGIAS_MEMORIA, que se usa para resolver el default del
# entorno. Se repite aquí porque el dominio del corto plazo no debe depender de
# que el .env haya definido una estrategia válida.
ESTRATEGIAS_CORTO_PLAZO = ("buffer", "ventana", "resumen")

# Tope de caracteres del resumen que genera el resumidor por defecto.
LIMITE_RESUMEN = 600

# Palabras sin contenido de búsqueda: si se puntuaran, cualquier hecho que
# contenga "de" o "mi" parecería relevante para cualquier consulta.
PALABRAS_VACIAS = frozenset({
    "con", "cual", "cuando", "como", "del", "donde", "esta", "este", "esto",
    "las", "los", "mas", "mis", "para", "pero", "por", "que", "quien", "sin",
    "sobre", "son", "sus", "todo", "una", "uno", "unos", "unas", "y",
})

ESQUEMA = """
CREATE TABLE IF NOT EXISTS hechos (
    usuario        TEXT NOT NULL,
    clave          TEXT NOT NULL,
    valor          TEXT NOT NULL,
    actualizado_en TEXT NOT NULL,
    PRIMARY KEY (usuario, clave)
);

CREATE TABLE IF NOT EXISTS sesiones (
    id        TEXT PRIMARY KEY,
    creada_en TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS turnos (
    id_sesion  TEXT NOT NULL,
    rol        TEXT NOT NULL,
    contenido  TEXT NOT NULL,
    creado_en  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_turnos_sesion ON turnos(id_sesion);
"""


class ErrorMemoria(Exception):
    """Fallo de dominio con un código estable para la capa de herramientas.

    El código permite distinguir un argumento inválido (`validacion`), una
    estrategia fuera de la allowlist (`estrategia_invalida`) y una sesión
    inexistente (`no_encontrado`) sin analizar el mensaje.
    """

    def __init__(self, mensaje: str, codigo: str = "validacion"):
        super().__init__(mensaje)
        self.codigo = codigo


def _texto_obligatorio(valor: str, campo: str) -> str:
    texto = (valor or "").strip()
    if not texto:
        raise ErrorMemoria(f"El campo '{campo}' no puede venir vacío.", "validacion")
    return texto


def _validar_estrategia(estrategia: str) -> str:
    normalizada = (estrategia or "").strip().lower()
    if normalizada not in ESTRATEGIAS_CORTO_PLAZO:
        raise ErrorMemoria(
            f"Estrategia de memoria '{estrategia}' no permitida. Valores "
            f"válidos: {', '.join(ESTRATEGIAS_CORTO_PLAZO)}.",
            "estrategia_invalida",
        )
    return normalizada


def _ahora() -> str:
    """Marca de tiempo ordenable y legible, con microsegundos para desempatar."""
    return datetime.now().isoformat()


def _tokens(texto: str) -> list[str]:
    """Palabras en minúsculas y sin tildes, para comparar consulta y hechos.

    Se normaliza con NFKD y se descartan los diacríticos: así "Ñuñoa" y "nunoa"
    se consideran la misma palabra en ambos lados de la comparación.
    """
    descompuesto = unicodedata.normalize("NFKD", (texto or "").lower())
    sin_tildes = "".join(c for c in descompuesto if not unicodedata.combining(c))
    return [parte for parte in re.split(r"[^0-9a-z]+", sin_tildes) if parte]


def _tokens_consulta(consulta: str) -> list[str]:
    """Palabras útiles de la consulta, sin repetir y sin palabras vacías."""
    utiles: list[str] = []
    for token in _tokens(consulta):
        if len(token) < 3 or token in PALABRAS_VACIAS or token in utiles:
            continue
        utiles.append(token)
    return utiles


def resumidor_por_defecto(resumen_previo: str,
                          turnos: list[dict[str, object]]) -> str:
    """Resumidor local: concatena los turnos y conserva la cola reciente.

    No toca la red ni el proveedor del LLM. El recorte conserva el final del
    texto porque los turnos más nuevos son los que el usuario acaba de decir.
    """
    partes: list[str] = []
    previo = (resumen_previo or "").strip()
    if previo:
        partes.append(previo)
    partes.extend(f"{turno['rol']}: {turno['contenido']}" for turno in turnos)
    texto = " | ".join(partes).strip()
    if len(texto) > LIMITE_RESUMEN:
        texto = "…" + texto[-(LIMITE_RESUMEN - 1):]
    return texto


# Firma del resumidor inyectable: recibe el resumen acumulado y los turnos que
# salieron de la ventana, y devuelve el resumen nuevo.
Resumidor = Callable[[str, list[dict[str, object]]], str]


class MemoriaCortoPlazo:
    """Historial de la conversación en curso, con tres políticas de recorte."""

    def __init__(self, estrategia: str = "ventana", ventana: int = 6,
                 resumidor: Resumidor | None = None) -> None:
        self.estrategia = _validar_estrategia(estrategia)
        # Un valor accidental (0 o negativo) dejaría el historial vacío: el
        # mismo criterio que aplica config.MEMORIA_VENTANA.
        self.ventana = max(1, int(ventana))
        self._resumidor: Resumidor = resumidor or resumidor_por_defecto
        self._turnos: list[dict[str, str]] = []
        self._resumen = ""

    # -- Lectura ----------------------------------------------------------
    @property
    def resumen(self) -> str:
        return self._resumen

    def turnos(self) -> list[dict[str, object]]:
        """Copia de los turnos vigentes: el llamador no muta el historial."""
        return [dict(turno) for turno in self._turnos]

    def texto(self) -> str:
        """Bloque listo para el prompt: resumen (si hay) y turnos vigentes."""
        lineas: list[str] = []
        if self._resumen:
            lineas.append("Resumen de turnos anteriores:")
            lineas.append(self._resumen)
        lineas.extend(
            f"{turno['rol']}: {turno['contenido']}" for turno in self._turnos
        )
        return "\n".join(lineas)

    # -- Escritura --------------------------------------------------------
    def agregar(self, rol: str, contenido: str) -> None:
        self._turnos.append({
            "rol": _texto_obligatorio(rol, "rol"),
            "contenido": _texto_obligatorio(contenido, "contenido"),
        })
        self._recortar()

    def limpiar(self) -> None:
        """Vacía el historial y el resumen: una sesión nueva no arrastra nada."""
        self._turnos = []
        self._resumen = ""

    # -- Recorte ----------------------------------------------------------
    def _recortar(self) -> None:
        if self.estrategia == "buffer":
            return
        excedente = len(self._turnos) - self.ventana
        if excedente <= 0:
            return
        salientes = self._turnos[:excedente]
        self._turnos = self._turnos[excedente:]
        if self.estrategia == "resumen":
            # Solo la estrategia de resumen pliega lo olvidado; la ventana pura
            # descarta, que es justamente su diferencia verificable.
            self._resumen = (
                self._resumidor(self._resumen, [dict(t) for t in salientes]) or ""
            ).strip()


class MemoriaLargoPlazo:
    """Hechos del usuario y sesiones persistidas en SQLite."""

    def __init__(self, ruta: str | Path | None = None,
                 usuario: str = "por_defecto") -> None:
        self.ruta = Path(ruta) if ruta else config.RUTA_MEMORIA
        self.ruta.parent.mkdir(parents=True, exist_ok=True)
        self.usuario = _texto_obligatorio(usuario, "usuario")
        self._crear_esquema()

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
    def _siguiente_id_sesion(conexion: sqlite3.Connection) -> str:
        fila = conexion.execute(
            "SELECT id FROM sesiones ORDER BY id DESC LIMIT 1"
        ).fetchone()
        ultimo = 0
        if fila is not None:
            try:
                ultimo = int(str(fila["id"]).split("-")[1])
            except (IndexError, ValueError):
                # Un id con formato ajeno no debe bloquear la creación de
                # sesiones nuevas: se continúa desde el conteo existente.
                ultimo = conexion.execute("SELECT COUNT(*) FROM sesiones").fetchone()[0]
        return f"SES-{ultimo + 1:04d}"

    # -- Hechos -----------------------------------------------------------
    def guardar_hecho(self, clave: str, valor: str) -> dict[str, object]:
        """Crea o actualiza un hecho. Un hecho por (usuario, clave)."""
        clave_limpia = _texto_obligatorio(clave, "clave")
        valor_limpio = _texto_obligatorio(valor, "valor")
        momento = _ahora()
        with self._conexion() as conexion:
            conexion.execute(
                """INSERT INTO hechos (usuario, clave, valor, actualizado_en)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(usuario, clave) DO UPDATE SET
                       valor = excluded.valor,
                       actualizado_en = excluded.actualizado_en""",
                (self.usuario, clave_limpia, valor_limpio, momento),
            )
        return {
            "usuario": self.usuario,
            "clave": clave_limpia,
            "valor": valor_limpio,
            "actualizado_en": momento,
        }

    def listar_hechos(self) -> list[dict[str, object]]:
        """Hechos del usuario, del más reciente al más antiguo."""
        with self._conexion() as conexion:
            filas = conexion.execute(
                """SELECT usuario, clave, valor, actualizado_en
                   FROM hechos WHERE usuario = ?
                   ORDER BY actualizado_en DESC, clave ASC""",
                (self.usuario,),
            ).fetchall()
        return [dict(fila) for fila in filas]

    def borrar_hecho(self, clave: str) -> bool:
        clave_limpia = _texto_obligatorio(clave, "clave")
        with self._conexion() as conexion:
            cursor = conexion.execute(
                "DELETE FROM hechos WHERE usuario = ? AND clave = ?",
                (self.usuario, clave_limpia),
            )
            borrados = cursor.rowcount
        # Un DELETE que no tocó filas devuelve False: el llamador no debe creer
        # que borró algo que nunca existió.
        return borrados > 0

    def hechos_relevantes(self, consulta: str,
                          limite_caracteres: int = 400) -> list[dict[str, object]]:
        """Hechos que comparten palabras con la consulta, con tope de tamaño.

        El puntaje es la cantidad de palabras de la consulta presentes en la
        clave o el valor; los empates se resuelven por recencia. Si ninguna
        coincide se cae a la recencia pura, porque un bloque de memoria sin
        relación todavía es preferible a un prompt sin memoria del usuario.
        """
        limite = max(1, int(limite_caracteres))
        palabras = _tokens_consulta(consulta)
        # listar_hechos ya viene ordenado por recencia: el índice es el
        # desempate, sin necesidad de ordenar por fecha otra vez.
        puntuados: list[tuple[int, int, dict[str, object]]] = []
        for indice, hecho in enumerate(self.listar_hechos()):
            vocabulario = set(_tokens(f"{hecho['clave']} {hecho['valor']}"))
            puntaje = sum(1 for palabra in palabras if palabra in vocabulario)
            puntuados.append((puntaje, indice, hecho))
        con_coincidencia = [item for item in puntuados if item[0] > 0]
        candidatos = con_coincidencia or puntuados
        candidatos.sort(key=lambda item: (-item[0], item[1]))

        seleccionados: list[dict[str, object]] = []
        acumulado = 0
        for _, _, hecho in candidatos:
            tamano = len(str(hecho["valor"]))
            if seleccionados and acumulado + tamano > limite:
                # Se salta el hecho que no cabe, no se corta la lista: uno más
                # breve y relevante todavía puede entrar en el presupuesto.
                continue
            seleccionados.append(hecho)
            acumulado += tamano
        return seleccionados

    # -- Sesiones ---------------------------------------------------------
    def crear_sesion(self) -> str:
        with self._conexion() as conexion:
            id_sesion = self._siguiente_id_sesion(conexion)
            conexion.execute(
                "INSERT INTO sesiones (id, creada_en) VALUES (?, ?)",
                (id_sesion, _ahora()),
            )
        return id_sesion

    def guardar_turno(self, id_sesion: str, rol: str,
                      contenido: str) -> None:
        """Persiste un turno. Una sesión inexistente se rechaza en vez de
        acumular turnos que ninguna sesión va a poder recuperar."""
        id_limpio = _texto_obligatorio(id_sesion, "id_sesion")
        with self._conexion() as conexion:
            existe = conexion.execute(
                "SELECT 1 FROM sesiones WHERE id = ?", (id_limpio,)
            ).fetchone()
            if existe is None:
                raise ErrorMemoria(
                    f"No existe la sesión '{id_limpio}'.", "no_encontrado"
                )
            conexion.execute(
                """INSERT INTO turnos (id_sesion, rol, contenido, creado_en)
                   VALUES (?, ?, ?, ?)""",
                (
                    id_limpio,
                    _texto_obligatorio(rol, "rol"),
                    _texto_obligatorio(contenido, "contenido"),
                    _ahora(),
                ),
            )

    def listar_sesiones(self, limite: int = 20) -> list[dict[str, object]]:
        limite_limpio = max(1, min(int(limite), 100))
        with self._conexion() as conexion:
            filas = conexion.execute(
                "SELECT id, creada_en FROM sesiones ORDER BY id DESC LIMIT ?",
                (limite_limpio,),
            ).fetchall()
        return [dict(fila) for fila in filas]

    def cargar_turnos(self, id_sesion: str) -> list[dict[str, object]]:
        """Turnos de una sesión en orden de inserción; [] si no existe."""
        id_limpio = _texto_obligatorio(id_sesion, "id_sesion")
        with self._conexion() as conexion:
            filas = conexion.execute(
                """SELECT rol, contenido, creado_en FROM turnos
                   WHERE id_sesion = ? ORDER BY rowid""",
                (id_limpio,),
            ).fetchall()
        return [dict(fila) for fila in filas]

    def existe_sesion(self, id_sesion: str) -> bool:
        """True si la sesión está registrada, aunque no tenga turnos."""
        id_limpio = _texto_obligatorio(id_sesion, "id_sesion")
        with self._conexion() as conexion:
            fila = conexion.execute(
                "SELECT 1 FROM sesiones WHERE id = ?", (id_limpio,)
            ).fetchone()
        return fila is not None

    def borrar_sesion(self, id_sesion: str) -> bool:
        id_limpio = _texto_obligatorio(id_sesion, "id_sesion")
        with self._conexion() as conexion:
            cursor = conexion.execute(
                "DELETE FROM sesiones WHERE id = ?", (id_limpio,)
            )
            borrada = cursor.rowcount > 0
            if borrada:
                # Los turnos se borran explícitamente: las claves foráneas de
                # SQLite están desactivadas por defecto, así que confiar en un
                # ON DELETE CASCADE dejaría turnos huérfanos.
                conexion.execute(
                    "DELETE FROM turnos WHERE id_sesion = ?", (id_limpio,)
                )
        return borrada


class Memoria:
    """Fachada que combina corto y largo plazo para el agente."""

    def __init__(self, corto: MemoriaCortoPlazo | None = None,
                 largo: MemoriaLargoPlazo | None = None,
                 usuario: str = "por_defecto",
                 resumidor: Resumidor | None = None,
                 estrategia: str | None = None,
                 ventana: int | None = None) -> None:
        self.corto = corto if corto is not None else MemoriaCortoPlazo(
            estrategia=estrategia or config.MEMORIA_ESTRATEGIA,
            ventana=ventana if ventana is not None else config.MEMORIA_VENTANA,
            resumidor=resumidor,
        )
        self.largo = largo if largo is not None else MemoriaLargoPlazo(
            usuario=usuario
        )
        self.usuario = usuario
        self._sesion_id: str | None = None

    # -- Prompt -----------------------------------------------------------
    def bloque_para(self, consulta: str) -> str:
        """Bloque de hechos relevantes; "" cuando no hay nada que inyectar."""
        hechos = self.largo.hechos_relevantes(consulta)
        if not hechos:
            return ""
        lineas = ["Memoria relevante del usuario:"]
        lineas.extend(
            f"- {hecho['clave']}: {hecho['valor']}" for hecho in hechos
        )
        return "\n".join(lineas)

    # -- Hechos -----------------------------------------------------------
    def recordar(self, clave: str, valor: str) -> dict[str, object]:
        return self.largo.guardar_hecho(clave, valor)

    def listar_hechos(self) -> list[dict[str, object]]:
        return self.largo.listar_hechos()

    def borrar_hecho(self, clave: str) -> bool:
        return self.largo.borrar_hecho(clave)

    # -- Sesiones ---------------------------------------------------------
    @property
    def sesion_id(self) -> str | None:
        return self._sesion_id

    def abrir_sesion(self) -> str:
        self._sesion_id = self.largo.crear_sesion()
        return self._sesion_id

    def registrar(self, rol: str, contenido: str) -> None:
        """Agrega al corto plazo y, si hay sesión abierta, la persiste."""
        self.corto.agregar(rol, contenido)
        if self._sesion_id is not None:
            self.largo.guardar_turno(self._sesion_id, rol, contenido)

    def cargar_sesion(self, id_sesion: str) -> int:
        """Restaura una sesión en el corto plazo y la deja activa.

        Falla cerrada: si la sesión no existe lanza `ErrorMemoria` ANTES de
        tocar el corto plazo. Antes se limpiaba el historial y se dejaba activo
        un id inexistente, con lo que el siguiente `registrar` reventaba dentro
        del agente (que lo traga) y el turno se perdía en silencio.
        """
        id_limpio = _texto_obligatorio(id_sesion, "id_sesion")
        if not self.largo.existe_sesion(id_limpio):
            raise ErrorMemoria(
                f"No existe la sesión '{id_limpio}'.", "no_encontrado"
            )
        turnos = self.largo.cargar_turnos(id_limpio)
        self.corto.limpiar()
        for turno in turnos:
            self.corto.agregar(str(turno["rol"]), str(turno["contenido"]))
        self._sesion_id = id_limpio
        return len(turnos)


# --- Resumidor respaldado por LLM (IL2.2) --------------------------------
# `resumidor_por_defecto` (local) NO es la estrategia de resumen: es el respaldo.
# La estrategia `resumen` existe para ahorrar tokens conservando hechos, y eso
# requiere un modelo que condense; concatenar y recortar solo es una ventana con
# prefijo. El resumidor local queda para cuando no hay LLM o cuando el proveedor
# falla, y `ResumidorConFallback` cuenta esos fallos para que la traza lo diga.

PROMPT_RESUMEN = """Resume la conversación de soporte de TecnoApoyo en {max_palabras} palabras o menos.

Conserva hechos del usuario, preferencias, identificadores de caso (CASO-NNNN) y decisiones tomadas. No inventes información que no esté en los turnos.

Resumen previo:
{resumen_previo}

Turnos nuevos:
{turnos}
"""


def resumir_con_llm(llm: Any, resumen_previo: str,
                    turnos: list[dict[str, object]],
                    max_palabras: int = 120) -> str:
    """Condensa los turnos salientes con el proveedor.

    Salida acotada a `max_palabras`: un resumen que crece sin límite deja de
    ahorrar tokens y vuelve a ser un historial completo con otro nombre.
    """
    detalle = "\n".join(
        f"{turno.get('rol', '?')}: {turno.get('contenido', '')}" for turno in turnos
    ) or "(sin turnos)"
    prompt = PROMPT_RESUMEN.format(
        max_palabras=max(1, int(max_palabras)),
        resumen_previo=(resumen_previo or "(sin resumen previo)").strip(),
        turnos=detalle,
    )
    respuesta = llm.invoke(prompt)
    contenido = getattr(respuesta, "content", "") or ""
    if isinstance(contenido, list):
        contenido = " ".join(str(parte) for parte in contenido)
    return " ".join(str(contenido).split()[:max(1, int(max_palabras))]).strip()


class ResumidorConFallback:
    """Resumidor con respaldo local: una falla del LLM no tumba la sesión.

    `fallos` cuenta cuántas veces se usó el respaldo para que la interfaz pueda
    advertir que el resumen es degradado en vez de fingir que fue el modelo.
    """

    def __init__(self, principal: Resumidor,
                 respaldo: Resumidor | None = None) -> None:
        self.principal = principal
        self.respaldo = respaldo or resumidor_por_defecto
        self.fallos = 0

    def __call__(self, resumen_previo: str,
                 turnos: list[dict[str, object]]) -> str:
        try:
            return self.principal(resumen_previo, turnos)
        except Exception:  # noqa: BLE001 - el resumen es accesorio, no bloqueante
            self.fallos += 1
            return self.respaldo(resumen_previo, turnos)


def crear_resumidor(nombre: str | None = None, llm: Any | None = None) -> Resumidor:
    """Devuelve el resumidor configurado, con fallback local garantizado.

    `llm`: resume con el proveedor y cae al local si falla.
    `local`: usa el respaldo directamente (sin red ni cuota).
    Si se pide `llm` sin modelo inyectado se devuelve el local: el arranque no
    debe depender de que el proveedor esté disponible.
    """
    elegido = (nombre or config.MEMORIA_RESUMIDOR).strip().lower()
    if elegido == "llm" and llm is not None:
        return ResumidorConFallback(
            principal=lambda previo, salientes: resumir_con_llm(llm, previo, salientes)
        )
    return resumidor_por_defecto
