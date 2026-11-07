"""Herramientas del agente de TecnoApoyo (RA2 / IL2.1).

Familias de tools, con un contrato de retorno común (`ResultadoHerramienta`):

  - consulta: `buscar_documentacion`, `consultar_caso`, `listar_casos`
  - escritura: `crear_caso`, `agregar_nota`, `actualizar_estado`,
    `escalar_a_humano`
  - razonamiento: `calcular_plazo` (determinista, sin red y sin `eval`)
  - conversación: `pedir_datos`, que pregunta al usuario el dato que falta y
    no lee ni escribe casos

Dos capas, con propósitos distintos:
  1. Funciones puras. Reciben `repositorio` (y `motor` para el RAG) como
     primer argumento y devuelven `ResultadoHerramienta`. Son lo que prueban
     los tests de contrato y lo que puede reutilizar cualquier ejecutor.
  2. Tools registradas (`crear_herramientas`). Exponen al modelo solo los
     argumentos del negocio y devuelven JSON serializable, porque el bucle del
     agente debe poder meter ese texto en un `ToolMessage`.

Cómo se impide una escritura sin humano:

  - El esquema de argumentos que ve el modelo no tiene campo `confirmado` y
    declara `extra="forbid"`: si el modelo lo envía igual, la validación
    rechaza la llamada completa y no se ejecuta nada.
  - La confirmación no es un dato, es una función (`Confirmador`) que el
    ejecutor inyecta al construir las herramientas. Esa función es la que
    muestra el `s/n` en consola. Sin confirmador, o si devuelve False, la
    escritura no ocurre y la tool responde `ok=False`.
  - Por eso el modelo no puede autoconfirmarse: no tiene forma de alcanzar el
    canal de confirmación, solo puede proponer la acción.

Cómo se decide la identidad de la sesión:

  - El agente publica el mensaje del turno en curso con `mensaje_del_turno`
    (un `ContextVar`) antes de ejecutar cualquier tool, en los dos modos, y lo
    restablece al terminar el turno.
  - `identificar_cliente` solo acepta un nombre que aparezca en ese mensaje. Un
    correo, o un nombre que venga de la memoria o del historial, responde
    `ok=False` sin repetir el valor y sin cambiar el alcance: la memoria no
    prueba quién está escribiendo.

Qué devuelve cada tool en `datos` se declara una sola vez en `CLAVES_SALIDA`;
`validar_plan` usa esa tabla para rechazar un camino `{{pN.datos...}}` antes de
ejecutar nada.
"""
import unicodedata
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import date, timedelta
from typing import Any

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src import config
from src.casos import (
    ESTADOS_VALIDOS,
    PRIORIDADES_VALIDAS,
    ErrorCaso,
    RepositorioCasos,
    clave_de_cliente,
)

# Plazo estándar en días hábiles por tipo de servicio. Es una tabla, no una
# fórmula inventada por el modelo: el agente solo puede consultar valores
# definidos aquí, y el cálculo de fechas lo hace `sumar_dias_habiles`.
PLAZOS_HABILES: dict[str, int] = {
    "diagnostico": 2,
    "revision": 1,
    "reparacion": 5,
    "repuesto": 7,
    "formateo": 1,
    "respaldo_datos": 2,
    "mantenimiento_preventivo": 3,
    "actualizacion_software": 2,
}

MENSAJE_SIN_CONFIRMACION = (
    "La escritura '{accion}' requiere confirmación explícita del usuario (s/n). "
    "No se modificó ningún caso."
)

# Alcances con los que se pueden construir las tools.
#   - 'global': sin filtro por cliente. Es el default y lo que esperan los
#     scripts de evaluación y demostración, que trabajan sobre el dominio
#     completo.
#   - 'sesion': `listar_casos` queda atado al cliente de la sesión. El modelo no
#     puede cambiarlo por argumentos, y sin cliente la tool se niega a listar.
ALCANCES_HERRAMIENTAS = ("global", "sesion")

# Mensaje del alcance de sesión cuando todavía no hay cliente identificado. Es
# accionable a propósito: dice el comando exacto que destraba la consulta, en
# vez de responder "no encontré casos", que sonaba a dato del dominio.
MENSAJE_CLIENTE_SIN_IDENTIFICAR = (
    "No hay un cliente identificado en esta sesión, así que no puedo listar "
    "casos: solo puedo mostrar los casos de un cliente identificado, nunca los "
    "de otros clientes. Pide al operador que identifique al cliente con "
    "/cliente <nombre> y vuelve a intentarlo."
)

# Respuesta única para un id que la sesión no puede usar: no existe, o existe y
# es de otro cliente. Es el mismo texto para los dos casos a propósito: si
# difiriera, el agente podría sondear ids ajenos y deducir cuáles existen. No
# incluye ningún dato del caso, ni siquiera el nombre del cliente del id.
MENSAJE_CASO_NO_DISPONIBLE = (
    "El caso '{id_caso}' no está disponible en esta sesión: puede que no exista "
    "o que no pertenezca al cliente identificado, y por privacidad no se "
    "muestra ningún dato. Revisa el id, o pide al operador que identifique al "
    "cliente con /cliente <nombre>."
)

# (accion, resumen de lo que se escribiría) -> True solo si el humano aprobó.
Confirmador = Callable[[str, dict[str, Any]], bool]

# Mensaje del turno en curso, publicado por el agente antes de ejecutar
# cualquier herramienta (en los dos modos). Una tool puede contrastar lo que el
# modelo propone con lo que la persona escribió en ese mensaje, que es el único
# texto del que se puede afirmar que la persona lo dijo. Es un `ContextVar` y no
# un global: dos turnos concurrentes no comparten mensaje, y al salir del turno
# el valor anterior vuelve solo.
_MENSAJE_TURNO: ContextVar[str] = ContextVar("mensaje_turno", default="")


@contextmanager
def mensaje_del_turno(texto: str) -> Iterator[None]:
    """Publica el mensaje del turno para las tools y lo restablece al salir."""
    token = _MENSAJE_TURNO.set(texto or "")
    try:
        yield
    finally:
        _MENSAJE_TURNO.reset(token)


def mensaje_turno_actual() -> str:
    """Mensaje del turno en curso (cadena vacía fuera de un turno)."""
    return _MENSAJE_TURNO.get()


def normalizar_para_comparar(valor: str) -> str:
    """Minúsculas y sin tildes, para comparar nombres tal como se escriben.

    Quien declara su nombre escribe "María", y el modelo puede repetirlo como
    "maria": la comparación no puede depender del acento ni de la mayúscula.
    """
    descompuesto = unicodedata.normalize("NFKD", valor or "")
    sin_tildes = "".join(
        letra for letra in descompuesto if not unicodedata.combining(letra)
    )
    return sin_tildes.lower()


def _palabras_de(texto: str) -> list[str]:
    """Palabras del texto, sin partir correos ni nombres: `a@b.cl` es una."""
    return [
        normalizar_para_comparar(palabra.strip(".,;:¡!¿?()[]\"'"))
        for palabra in (texto or "").split()
    ]


def _menciona_el_nombre(mensaje: str, nombre: str) -> bool:
    """El nombre aparece como palabra propia, no dentro de un correo ni de otra.

    «ricardo» no cuenta en «escríbeme a ricardo@testeo.com»: el correo es una
    sola palabra y no coincide. Sí cuenta en «soy ricardo» o «habla Ricardo».
    """
    objetivo = [parte for parte in _palabras_de(nombre) if parte]
    if not objetivo:
        return False
    palabras = [parte for parte in _palabras_de(mensaje) if parte]
    ancho = len(objetivo)
    for inicio in range(len(palabras) - ancho + 1):
        if palabras[inicio:inicio + ancho] == objetivo:
            return True
    return False


# Mensaje cuando el valor propuesto para identificar es un correo. No repite el
# valor: repetirlo lo devuelve al modelo y al historial, que es justo lo que la
# regla evita.
MENSAJE_IDENTIFICACION_POR_CORREO = (
    "No puedo identificar a un cliente con una dirección de correo: el dominio "
    "de casos no guarda correo y la identidad solo se toma del nombre que la "
    "persona declara. Pide el nombre completo tal como figura en el caso o el "
    "identificador CASO-NNNN."
)

# Mensaje cuando el nombre no aparece en el mensaje del turno actual: puede
# venir de la memoria, del historial o de una suposición del modelo. Tampoco
# repite el valor.
MENSAJE_IDENTIFICACION_FUERA_DEL_TURNO = (
    "No puedo identificar a un cliente con un nombre que no aparezca en el "
    "mensaje de este turno: los datos de memoria, el correo y el historial no "
    "identifican a nadie. Pide a la persona que declare su nombre completo y "
    "vuelve a intentarlo."
)

# Mensaje cuando el nombre es válido pero no tiene casos. Tampoco repite el
# valor: puede haberse escrito en un contexto que no era una identificación.
MENSAJE_IDENTIFICACION_SIN_CASOS = (
    "No hay casos registrados a ese nombre. Pide el nombre completo tal como "
    "figura en el caso o el identificador CASO-NNNN."
)


class ResultadoHerramienta(BaseModel):
    """Contrato común de toda herramienta del agente."""

    ok: bool
    datos: Any = None
    error: str | None = None

    def como_json(self) -> str:
        """Serializa el contrato para el `ToolMessage` del bucle del agente."""
        return self.model_dump_json()


# Declaración única de lo que cada tool devuelve dentro de `datos`. Es el
# contrato que consume `validar_plan`: un placeholder `{{pN.datos.clave}}` se
# contrasta con las claves de la tool del paso pN antes de ejecutar nada, y el
# mensaje de error enumera esta misma lista. Si la salida real cambia y esta
# declaración no, el test de contrato lo señala.
#
# Solo se declara el primer nivel: lo que hay dentro de una lista de casos o de
# fragmentos se resuelve recién con el dato real, en tiempo de ejecución.
CAMPOS_DE_CASO = (
    "id", "cliente", "categoria", "descripcion", "prioridad", "estado",
    "procedimiento", "fecha_creacion", "notas",
)

CLAVES_SALIDA: dict[str, tuple[str, ...]] = {
    "buscar_documentacion": (
        "fragmentos", "mejor_score", "umbral", "evidencia_suficiente",
    ),
    "consultar_caso": CAMPOS_DE_CASO,
    "listar_casos": ("total", "casos"),
    "identificar_cliente": ("cliente", "casos"),
    "crear_caso": CAMPOS_DE_CASO,
    "agregar_nota": CAMPOS_DE_CASO,
    "actualizar_estado": CAMPOS_DE_CASO,
    "escalar_a_humano": CAMPOS_DE_CASO,
    "calcular_plazo": (
        "tipo_servicio", "fecha_inicio", "dias_habiles_base",
        "dias_habiles_extra", "dias_habiles_totales", "fecha_estimada",
    ),
    "recordar": ("usuario", "clave", "valor", "actualizado_en"),
    "pedir_datos": ("pregunta",),
}


def _sin_confirmacion(accion: str) -> ResultadoHerramienta:
    return ResultadoHerramienta(
        ok=False, error=MENSAJE_SIN_CONFIRMACION.format(accion=accion)
    )


def _desde_error(exc: Exception) -> ResultadoHerramienta:
    return ResultadoHerramienta(ok=False, error=str(exc))


def _confirmar(confirmador: Confirmador | None, accion: str,
               resumen: dict[str, Any]) -> bool:
    """Pregunta al humano antes de escribir. Falla cerrado por diseño."""
    if confirmador is None:
        return False
    try:
        return bool(confirmador(accion, resumen))
    except Exception:  # noqa: BLE001 - una consola rota no debe autorizar nada
        return False


# --- Contratos de argumentos (validación Pydantic antes de ejecutar) ------
class _ArgsBase(BaseModel):
    """Base de los esquemas visibles al modelo.

    `extra="forbid"` es la pieza que cierra el bypass: un campo que no esté
    declarado aquí (por ejemplo un `confirmado` inventado) invalida la llamada
    entera en vez de colarse hasta la escritura.
    """

    model_config = ConfigDict(extra="forbid")


class ArgsBuscarDocumentacion(_ArgsBase):
    """Argumentos de la búsqueda documental."""

    pregunta: str = Field(min_length=3, max_length=500,
                          description="Pregunta del cliente en lenguaje natural.")
    k: int = Field(default=3, ge=1, le=10,
                   description="Cantidad de fragmentos a recuperar.")


class ArgsPedirDatos(_ArgsBase):
    """Argumentos de la pregunta que el agente hace al usuario."""

    pregunta: str = Field(min_length=3, max_length=500,
                          description="Pregunta breve y concreta para el usuario.")

    @field_validator("pregunta")
    @classmethod
    def _pregunta_con_contenido(cls, valor: str) -> str:
        # `min_length` cuenta espacios: una pregunta en blanco no es una
        # pregunta. El valor se conserva tal cual, sin recortarlo.
        if not (valor or "").strip():
            raise ValueError("La pregunta no puede estar vacía.")
        return valor


class ArgsConsultarCaso(_ArgsBase):
    """Argumentos de la consulta de un caso."""

    id_caso: str = Field(description="Identificador del caso, por ejemplo CASO-0004.")

    @field_validator("id_caso")
    @classmethod
    def _normalizar(cls, valor: str) -> str:
        normalizado = (valor or "").strip().upper()
        if not normalizado.startswith("CASO-") or not normalizado[5:].isdigit() \
                or len(normalizado[5:]) != 4:
            raise ValueError("El id debe tener el formato CASO-NNNN.")
        return normalizado


class ArgsListarCasos(_ArgsBase):
    """Argumentos del listado de casos."""

    estado: str | None = Field(
        default=None,
        description=f"Filtro opcional. Valores válidos: {', '.join(ESTADOS_VALIDOS)}.",
    )
    limite: int = Field(default=10, ge=1, le=50)


class ArgsIdentificarCliente(_ArgsBase):
    """Argumentos de la identificación del cliente de la sesión."""

    nombre: str = Field(
        min_length=2,
        max_length=120,
        description="Nombre del cliente tal como lo declara el usuario.",
    )


class ArgsCrearCaso(_ArgsBase):
    """Argumentos de la creación de un caso."""

    cliente: str = Field(min_length=3, max_length=120)
    categoria: str = Field(min_length=3, max_length=60,
                           description="Categoría de soporte, por ejemplo recepcion.")
    descripcion: str = Field(min_length=5, max_length=1000)
    prioridad: str = Field(default="media", description="baja, media o alta.")
    procedimiento: str | None = Field(
        default=None, description="Procedimiento asociado, por ejemplo PROC-001."
    )

    @field_validator("prioridad")
    @classmethod
    def _prioridad_valida(cls, valor: str) -> str:
        normalizado = (valor or "").strip().lower()
        if normalizado not in PRIORIDADES_VALIDAS:
            raise ValueError(
                f"Prioridad inválida. Valores válidos: {', '.join(PRIORIDADES_VALIDAS)}."
            )
        return normalizado

    @field_validator("procedimiento")
    @classmethod
    def _procedimiento_valido(cls, valor: str | None) -> str | None:
        if valor is None or not str(valor).strip():
            return None
        normalizado = str(valor).strip().upper()
        if not normalizado.startswith("PROC-") or not normalizado[5:].isdigit() \
                or len(normalizado[5:]) != 3:
            raise ValueError("El procedimiento debe tener el formato PROC-NNN.")
        return normalizado


class ArgsAgregarNota(_ArgsBase):
    """Argumentos para agregar una nota a un caso."""

    id_caso: str = Field(description="Identificador del caso, por ejemplo CASO-0004.")
    nota: str = Field(min_length=3, max_length=1000)


class ArgsActualizarEstado(_ArgsBase):
    """Argumentos para cambiar el estado de un caso."""

    id_caso: str = Field(description="Identificador del caso, por ejemplo CASO-0004.")
    estado: str = Field(description=f"Uno de: {', '.join(ESTADOS_VALIDOS)}.")

    @field_validator("estado")
    @classmethod
    def _estado_valido(cls, valor: str) -> str:
        normalizado = (valor or "").strip().lower()
        if normalizado not in ESTADOS_VALIDOS:
            raise ValueError(
                f"Estado inválido. Valores válidos: {', '.join(ESTADOS_VALIDOS)}."
            )
        return normalizado


class ArgsEscalarHumano(_ArgsBase):
    """Argumentos para escalar un caso a una persona."""

    id_caso: str = Field(description="Identificador del caso, por ejemplo CASO-0003.")
    motivo: str = Field(min_length=3, max_length=500)


class ArgsCalcularPlazo(_ArgsBase):
    """Argumentos del cálculo de plazo."""
    tipo_servicio: str = Field(
        description=f"Uno de: {', '.join(sorted(PLAZOS_HABILES))}."
    )
    fecha_inicio: str | None = Field(
        default=None, description="Fecha ISO de inicio (AAAA-MM-DD). Por defecto, hoy."
    )
    dias_habiles_extra: int = Field(
        default=0, ge=0, le=30, description="Días hábiles adicionales por complejidad."
    )


class ArgsRecordar(_ArgsBase):
    """Argumentos para guardar un hecho duradero del usuario."""

    clave: str = Field(
        min_length=2, max_length=60,
        description="Etiqueta breve del hecho, por ejemplo correo o equipo.",
    )
    valor: str = Field(
        min_length=2, max_length=500,
        description="Valor literal del hecho, tal como lo expresó el usuario.",
    )


# --- Funciones puras: consulta -------------------------------------------
def buscar_documentacion(motor: Any, pregunta: str,
                         k: int = 3) -> ResultadoHerramienta:
    """Recupera fragmentos y decide si la evidencia alcanza el umbral del RAG.

    No genera respuesta: el veredicto `evidencia_suficiente` usa el mismo
    `UMBRAL_ABSTENCION` del motor, así el agente no puede responder por encima
    de una abstención.
    """
    try:
        fragmentos = motor.recuperar(pregunta, k=k)
    except Exception as exc:  # el índice puede faltar o estar corrupto
        return _desde_error(exc)
    mejor = fragmentos[0]["score"] if fragmentos else 0.0
    return ResultadoHerramienta(ok=True, datos={
        "fragmentos": fragmentos,
        "mejor_score": mejor,
        "umbral": config.UMBRAL_ABSTENCION,
        "evidencia_suficiente": mejor >= config.UMBRAL_ABSTENCION,
    })


def consultar_caso(repositorio: RepositorioCasos, id_caso: str) -> ResultadoHerramienta:
    try:
        return ResultadoHerramienta(ok=True, datos=repositorio.obtener_caso(id_caso))
    except ErrorCaso as exc:
        return _desde_error(exc)


def listar_casos(repositorio: RepositorioCasos, estado: str | None = None,
                 limite: int = 10,
                 cliente: str | None = None) -> ResultadoHerramienta:
    """Lista casos, del más reciente al más antiguo, con su total real.

    `total` sale de `contar()` con los mismos filtros, no de `len(casos)`:
    `len(casos)` es el tamaño de la página y reportarlo como total fue un
    defecto real (el agente decía "hay 10 casos" mirando una página de 10).

    `cliente`, cuando se informa, acota el listado a ese cliente. La tool de la
    consola lo fija desde la sesión; el modelo no lo ve como argumento.
    """
    try:
        casos = repositorio.listar_casos(estado=estado, limite=limite,
                                         cliente=cliente)
        total = repositorio.contar(estado=estado, cliente=cliente)
    except ErrorCaso as exc:
        return _desde_error(exc)
    return ResultadoHerramienta(ok=True, datos={"total": total, "casos": casos})


# --- Funciones puras: escritura (siempre tras confirmación) --------------
def crear_caso(repositorio: RepositorioCasos, confirmador: Confirmador | None,
               cliente: str, categoria: str, descripcion: str,
               prioridad: str = "media",
               procedimiento: str | None = None) -> ResultadoHerramienta:
    resumen = {
        "cliente": cliente, "categoria": categoria, "descripcion": descripcion,
        "prioridad": prioridad, "procedimiento": procedimiento,
    }
    if not _confirmar(confirmador, "crear_caso", resumen):
        return _sin_confirmacion("crear_caso")
    try:
        caso = repositorio.crear_caso(
            cliente=cliente, categoria=categoria, descripcion=descripcion,
            prioridad=prioridad, procedimiento=procedimiento,
        )
    except ErrorCaso as exc:
        return _desde_error(exc)
    return ResultadoHerramienta(ok=True, datos=caso)


def agregar_nota(repositorio: RepositorioCasos, confirmador: Confirmador | None,
                 id_caso: str, nota: str) -> ResultadoHerramienta:
    resumen = {"id_caso": id_caso, "nota": nota}
    if not _confirmar(confirmador, "agregar_nota", resumen):
        return _sin_confirmacion("agregar_nota")
    try:
        caso = repositorio.agregar_nota(id_caso, nota)
    except ErrorCaso as exc:
        return _desde_error(exc)
    return ResultadoHerramienta(ok=True, datos=caso)


def actualizar_estado(repositorio: RepositorioCasos, confirmador: Confirmador | None,
                      id_caso: str, estado: str) -> ResultadoHerramienta:
    resumen = {"id_caso": id_caso, "estado_anterior": None, "estado_nuevo": estado}
    # El estado actual se lee antes de confirmar para que el humano vea de qué
    # a qué se cambia; si el caso no existe, el error aparece antes del s/n.
    try:
        resumen["estado_anterior"] = repositorio.obtener_caso(id_caso)["estado"]
    except ErrorCaso as exc:
        return _desde_error(exc)
    if not _confirmar(confirmador, "actualizar_estado", resumen):
        return _sin_confirmacion("actualizar_estado")
    try:
        caso = repositorio.actualizar_estado(id_caso, estado)
    except ErrorCaso as exc:
        return _desde_error(exc)
    return ResultadoHerramienta(ok=True, datos=caso)


def escalar_a_humano(repositorio: RepositorioCasos, confirmador: Confirmador | None,
                     id_caso: str, motivo: str) -> ResultadoHerramienta:
    resumen = {"id_caso": id_caso, "motivo": motivo}
    if not _confirmar(confirmador, "escalar_a_humano", resumen):
        return _sin_confirmacion("escalar_a_humano")
    try:
        caso = repositorio.escalar(id_caso, motivo)
    except ErrorCaso as exc:
        return _desde_error(exc)
    return ResultadoHerramienta(ok=True, datos=caso)


# --- Razonamiento determinista -------------------------------------------
def sumar_dias_habiles(inicio: date, dias: int) -> date:
    """Suma días hábiles salteando sábados y domingos.

    Determinista y sin feriados: la tabla de plazos y este cálculo son la
    fuente de verdad, no el modelo. Los feriados quedan fuera a propósito para
    no fijar un calendario que cambia por año y por país.
    """
    actual = inicio
    restantes = max(0, dias)
    while restantes > 0:
        actual += timedelta(days=1)
        if actual.weekday() < 5:
            restantes -= 1
    return actual


def calcular_plazo(tipo_servicio: str, fecha_inicio: str | None = None,
                   dias_habiles_extra: int = 0) -> ResultadoHerramienta:
    """Calcula el plazo comprometido de un servicio sin usar el modelo."""
    tipo = (tipo_servicio or "").strip().lower()
    if tipo not in PLAZOS_HABILES:
        return ResultadoHerramienta(
            ok=False,
            error=(
                f"Tipo de servicio '{tipo_servicio}' no reconocido. Valores válidos: "
                f"{', '.join(sorted(PLAZOS_HABILES))}."
            ),
        )
    if fecha_inicio is None or not str(fecha_inicio).strip():
        inicio = date.today()
    else:
        try:
            inicio = date.fromisoformat(str(fecha_inicio).strip())
        except ValueError:
            return ResultadoHerramienta(
                ok=False,
                error=f"Fecha '{fecha_inicio}' inválida. Se espera AAAA-MM-DD.",
            )
    base = PLAZOS_HABILES[tipo]
    total = base + max(0, int(dias_habiles_extra))
    estimada = sumar_dias_habiles(inicio, total)
    return ResultadoHerramienta(ok=True, datos={
        "tipo_servicio": tipo,
        "fecha_inicio": inicio.isoformat(),
        "dias_habiles_base": base,
        "dias_habiles_extra": max(0, int(dias_habiles_extra)),
        "dias_habiles_totales": total,
        "fecha_estimada": estimada.isoformat(),
    })


# --- Tools registradas para el modelo ------------------------------------
def crear_herramienta_recordar(memoria: Any) -> BaseTool:
    """Tool de memoria de largo plazo.

    No pide confirmación: es una nota del asistente sobre el propio usuario
    (preferencia, contacto, equipo), no una escritura del dominio de casos. Se
    puede inspeccionar y borrar desde `/memoria`.
    """

    @tool("recordar", args_schema=ArgsRecordar)
    def recordar_tool(clave: str, valor: str) -> str:
        """Guarda un hecho duradero del usuario para futuras conversaciones.

        Devuelve en `datos`: `usuario`, `clave`, `valor` y `actualizado_en`.
        Es una nota del asistente, no una prueba de identidad: un hecho guardado
        (por ejemplo un correo) no identifica a nadie en un turno posterior.
        """
        try:
            hecho = memoria.recordar(clave, valor)
        except Exception as exc:  # noqa: BLE001 - la memoria no debe abortar el bucle
            return _desde_error(exc).como_json()
        return ResultadoHerramienta(ok=True, datos=hecho).como_json()

    return recordar_tool


def crear_herramientas(repositorio: RepositorioCasos, motor: Any,
                       confirmador: Confirmador | None = None,
                       alcance: str = "global",
                       cliente_actual: str | Callable[[], str | None] | None = None,
                       al_identificar: Callable[[str], None] | None = None,
                       ) -> list[BaseTool]:
    """Construye las tools con sus dependencias ya inyectadas.

    `motor` solo necesita exponer `recuperar(pregunta, k=...)`; los tests
    inyectan un doble y la aplicación pasa el `RAGEngine` real.

    `confirmador` es la única vía hacia una escritura. Si se omite, las tools
    de escritura responden `ok=False`: el default seguro no es "preguntar
    después", es "no escribir".

    `alcance` decide qué casos puede ver y tocar la sesión:

      - `global` (default): sin filtro por cliente. Es el comportamiento
        histórico y el que usan `scripts/evaluar_agente.py`,
        `scripts/demo_agente.py` y `scripts/comparar_modos.py`, que trabajan
        sobre el dominio completo.
      - `sesion`: las tools que operan sobre un caso (`listar_casos`,
        `consultar_caso`, `agregar_nota`, `actualizar_estado` y
        `escalar_a_humano`) quedan atadas a `cliente_actual`. El cliente no es
        un argumento del esquema, así que el modelo no puede cambiarlo; si no
        hay cliente identificado, responden `ok=False` con el comando que lo
        resuelve y no devuelven ni modifican ningún caso ajeno.

    El cliente de la sesión es un filtro declarado por el operador, no una
    autenticación: no verifica identidad, solo acota qué casos se muestran.

    Cualquier otro alcance es un error de programación y se rechaza: un valor
    no reconocido no puede caer en silencio al alcance global, que es el que
    expone los casos de todos los clientes.
    """
    alcance_normalizado = (alcance or "").strip().lower()
    if alcance_normalizado not in ALCANCES_HERRAMIENTAS:
        raise ValueError(
            f"Alcance '{alcance}' no permitido. Valores válidos: "
            f"{', '.join(ALCANCES_HERRAMIENTAS)}."
        )
    if alcance_normalizado == "global":
        pass

    def cliente_de_sesion() -> str | None:
        """Cliente actual, resuelto en cada llamada.

        Acepta un nombre fijo (evaluación y demo) o un `callable` (la consola).
        Con el `callable`, `identificar_cliente` cambia el alcance de las tools
        ya construidas: el resto del mismo plan ve el cliente nuevo en vez de
        quedarse con el que había al armar el agente.
        """
        if alcance_normalizado != "sesion":
            return None
        valor = cliente_actual() if callable(cliente_actual) else cliente_actual
        return (valor or "").strip() or None

    def caso_no_disponible(id_caso: str) -> ResultadoHerramienta:
        """Resultado genérico para un id que la sesión no puede usar."""
        return ResultadoHerramienta(
            ok=False, error=MENSAJE_CASO_NO_DISPONIBLE.format(id_caso=id_caso)
        )

    def bloqueo_por_alcance(id_caso: str) -> ResultadoHerramienta | None:
        """Por qué la sesión no puede operar sobre este id, o `None` si puede.

        Se evalúa antes de devolver datos, antes de mostrar la confirmación y
        antes de escribir: el caso ajeno ni se muestra ni se modifica, y el
        humano no llega a ver un resumen de un caso que no le corresponde.

        En alcance global no hay bloqueo: la evaluación y la demo trabajan
        sobre el dominio completo y conservan su comportamiento exacto.
        """
        if alcance_normalizado != "sesion":
            return None
        cliente = cliente_de_sesion()
        if cliente is None:
            return ResultadoHerramienta(
                ok=False, error=MENSAJE_CLIENTE_SIN_IDENTIFICAR
            )
        try:
            caso = repositorio.obtener_caso(id_caso)
        except ErrorCaso:
            return caso_no_disponible(id_caso)
        # El cliente es el nombre declarado por el operador con `/cliente` o
        # por el agente con `identificar_cliente`: es un filtro, no una
        # autenticación. La comparación usa `clave_de_cliente`, la misma de la
        # colación del filtro SQL, para que el listado y el bloqueo por id no
        # discrepen: `MARIA FERNANDA ROJAS` es dueña de los casos de
        # `María Fernanda Rojas`, y de nadie más.
        dueno = clave_de_cliente(caso.get("cliente") or "")
        if dueno != clave_de_cliente(cliente):
            return caso_no_disponible(id_caso)
        return None

    def consultar_caso_del_alcance(id_caso: str) -> ResultadoHerramienta:
        """Lectura por id sujeta al alcance de la sesión."""
        bloqueo = bloqueo_por_alcance(id_caso)
        if bloqueo is not None:
            return bloqueo
        return consultar_caso(repositorio, id_caso)

    def listar_casos_del_alcance(estado: str | None,
                                 limite: int) -> ResultadoHerramienta:
        """Listado según el alcance, con el cliente fijado por la sesión."""
        cliente = cliente_de_sesion()
        if alcance_normalizado == "sesion" and cliente is None:
            return ResultadoHerramienta(
                ok=False, error=MENSAJE_CLIENTE_SIN_IDENTIFICAR
            )
        return listar_casos(repositorio, estado, limite, cliente)

    @tool("buscar_documentacion", args_schema=ArgsBuscarDocumentacion)
    def buscar_documentacion_tool(pregunta: str, k: int = 3) -> str:
        """Busca en la documentación oficial de TecnoApoyo (FAQ y procedimientos).

        Devuelve los fragmentos recuperados y si la evidencia alcanza el umbral
        del motor RAG. Si `evidencia_suficiente` es False, el agente debe
        abstenerse: no hay respaldo documental para responder.

        Devuelve en `datos`: `fragmentos` (lista), `mejor_score`, `umbral` y
        `evidencia_suficiente`. Los placeholders válidos son por ejemplo
        {{p1.datos.fragmentos.0.score}} o {{p1.datos.mejor_score}}.
        """
        return buscar_documentacion(motor, pregunta, k).como_json()

    @tool("consultar_caso", args_schema=ArgsConsultarCaso)
    def consultar_caso_tool(id_caso: str) -> str:
        """Devuelve un caso de soporte por su identificador.

        En una sesión con cliente identificado solo devuelve casos de ese
        cliente; cualquier otro id responde que no está disponible.

        Devuelve en `datos` los campos del caso: `id`, `cliente`, `categoria`,
        `descripcion`, `prioridad`, `estado`, `procedimiento`, `fecha_creacion`
        y `notas`. No existe `datos.id_caso`: el campo se llama `id`.
        """
        return consultar_caso_del_alcance(id_caso).como_json()

    @tool("listar_casos", args_schema=ArgsListarCasos)
    def listar_casos_tool(estado: str | None = None, limite: int = 10) -> str:
        """Lista casos de soporte, del más reciente al más antiguo.

        Devuelve el total real de casos que cumplen el filtro y la página
        pedida. En una sesión con cliente identificado, solo devuelve los casos
        de ese cliente: el alcance lo fija el operador, no los argumentos.

        Devuelve en `datos`: `casos` (lista de casos) y `total` (cuántos
        cumplen el filtro). Los placeholders válidos son por ejemplo
        {{p1.datos.casos.0.id}} o {{p1.datos.total}}.
        """
        return listar_casos_del_alcance(estado, limite).como_json()

    @tool("crear_caso", args_schema=ArgsCrearCaso)
    def crear_caso_tool(cliente: str, categoria: str, descripcion: str,
                        prioridad: str = "media",
                        procedimiento: str | None = None) -> str:
        """Propone crear un caso de soporte; la escritura la autoriza el humano.

        Devuelve en `datos` el caso creado: `id`, `cliente`, `categoria`,
        `descripcion`, `prioridad`, `estado`, `procedimiento`, `fecha_creacion`
        y `notas`.
        """
        return crear_caso(repositorio, confirmador, cliente, categoria,
                          descripcion, prioridad, procedimiento).como_json()

    @tool("agregar_nota", args_schema=ArgsAgregarNota)
    def agregar_nota_tool(id_caso: str, nota: str) -> str:
        """Propone agregar una nota a un caso; la escritura la autoriza el humano.

        Devuelve en `datos` el caso completo, con la nota en `notas`: `id`,
        `cliente`, `categoria`, `descripcion`, `prioridad`, `estado`,
        `procedimiento`, `fecha_creacion` y `notas`.
        """
        bloqueo = bloqueo_por_alcance(id_caso)
        if bloqueo is not None:
            return bloqueo.como_json()
        return agregar_nota(repositorio, confirmador, id_caso, nota).como_json()

    @tool("actualizar_estado", args_schema=ArgsActualizarEstado)
    def actualizar_estado_tool(id_caso: str, estado: str) -> str:
        """Propone cambiar el estado de un caso; la autoriza el humano.

        Devuelve en `datos` el caso completo con el `estado` nuevo: `id`,
        `cliente`, `categoria`, `descripcion`, `prioridad`, `estado`,
        `procedimiento`, `fecha_creacion` y `notas`.
        """
        bloqueo = bloqueo_por_alcance(id_caso)
        if bloqueo is not None:
            return bloqueo.como_json()
        return actualizar_estado(repositorio, confirmador, id_caso,
                                 estado).como_json()

    @tool("escalar_a_humano", args_schema=ArgsEscalarHumano)
    def escalar_a_humano_tool(id_caso: str, motivo: str) -> str:
        """Propone escalar un caso a una persona; la autoriza el humano.

        Devuelve en `datos` el caso completo, con `estado` en `escalado` y el
        motivo como última nota: `id`, `cliente`, `categoria`, `descripcion`,
        `prioridad`, `estado`, `procedimiento`, `fecha_creacion` y `notas`.
        """
        bloqueo = bloqueo_por_alcance(id_caso)
        if bloqueo is not None:
            return bloqueo.como_json()
        return escalar_a_humano(repositorio, confirmador, id_caso,
                                motivo).como_json()

    @tool("calcular_plazo", args_schema=ArgsCalcularPlazo)
    def calcular_plazo_tool(tipo_servicio: str, fecha_inicio: str | None = None,
                            dias_habiles_extra: int = 0) -> str:
        """Calcula el plazo comprometido de un servicio en días hábiles.

        Devuelve en `datos`: `tipo_servicio`, `fecha_inicio`,
        `dias_habiles_base`, `dias_habiles_extra`, `dias_habiles_totales` y
        `fecha_estimada` (ISO). El placeholder de la fecha es por ejemplo
        {{p1.datos.fecha_estimada}}.
        """
        return calcular_plazo(tipo_servicio, fecha_inicio,
                              dias_habiles_extra).como_json()

    @tool("identificar_cliente", args_schema=ArgsIdentificarCliente)
    def identificar_cliente_tool(nombre: str) -> str:
        """Identifica al cliente de la sesión con el nombre que declara el usuario.

        Ata la sesión a ese cliente para que `listar_casos`, `consultar_caso` y
        las escrituras operen sobre sus casos en este mismo turno. Si el nombre
        no tiene casos, no cambia el alcance y explica el motivo.

        Solo acepta un nombre que aparezca en el mensaje del turno actual: un
        correo, o un nombre que venga de la memoria o del historial, se rechaza
        con `ok=False` sin repetir el valor y sin cambiar el alcance.

        Devuelve en `datos`: `cliente` (nombre canónico de la base) y `casos`
        (cuántos casos tiene ese nombre).
        """
        if al_identificar is None:  # la tool no se publica sin este callback
            return ResultadoHerramienta(
                ok=False, error="La sesión no permite identificar clientes."
            ).como_json()
        limpio = nombre.strip()
        # S1: la identidad se toma del mensaje del turno, no de la memoria. Un
        # correo no identifica en este dominio y un nombre que la persona no
        # escribió en este turno tampoco: ambos casos fallan cerrado, sin
        # llamar a `al_identificar` y sin devolver el valor rechazado.
        if "@" in limpio:
            return ResultadoHerramienta(
                ok=False, error=MENSAJE_IDENTIFICACION_POR_CORREO
            ).como_json()
        if not _menciona_el_nombre(mensaje_turno_actual(), limpio):
            return ResultadoHerramienta(
                ok=False, error=MENSAJE_IDENTIFICACION_FUERA_DEL_TURNO
            ).como_json()
        try:
            total = repositorio.contar(cliente=limpio)
        except ErrorCaso as exc:
            return _desde_error(exc).como_json()
        if total == 0:
            return ResultadoHerramienta(
                ok=False, error=MENSAJE_IDENTIFICACION_SIN_CASOS
            ).como_json()
        # Nombre canónico de la base: así el filtro del alcance compara el
        # mismo texto que el listado posterior.
        canonico = repositorio.listar_casos(cliente=limpio, limite=1)[0]["cliente"]
        al_identificar(canonico)
        return ResultadoHerramienta(
            ok=True, datos={"cliente": canonico, "casos": total}
        ).como_json()

    @tool("pedir_datos", args_schema=ArgsPedirDatos)
    def pedir_datos_tool(pregunta: str) -> str:
        """Pregunta al usuario el dato que falta para poder continuar.

        Devuelve en `datos` la `pregunta` que el agente debe mostrar tal cual:
        no lee ni escribe casos, y no requiere un cliente identificado. El plan
        que la usa es de un solo paso y el turno termina en esa pregunta.
        """
        return ResultadoHerramienta(
            ok=True, datos={"pregunta": pregunta}
        ).como_json()

    return [
        buscar_documentacion_tool,
        consultar_caso_tool,
        listar_casos_tool,
        crear_caso_tool,
        agregar_nota_tool,
        actualizar_estado_tool,
        escalar_a_humano_tool,
        calcular_plazo_tool,
        pedir_datos_tool,
        *([identificar_cliente_tool]
          if alcance_normalizado == "sesion" and al_identificar is not None
          else []),
    ]


def por_nombre(herramientas: list[BaseTool]) -> dict[str, BaseTool]:
    """Indexa las tools por nombre, que es como las invoca el bucle del agente."""
    return {herramienta.name: herramienta for herramienta in herramientas}
