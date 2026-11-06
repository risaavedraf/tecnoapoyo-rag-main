"""Herramientas del agente de TecnoApoyo (RA2 / IL2.1).

Tres familias, con un contrato de retorno común (`ResultadoHerramienta`):

  - consulta: `buscar_documentacion`, `consultar_caso`, `listar_casos`
  - escritura: `crear_caso`, `agregar_nota`, `actualizar_estado`,
    `escalar_a_humano`
  - razonamiento: `calcular_plazo` (determinista, sin red y sin `eval`)

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
"""
from collections.abc import Callable
from datetime import date, timedelta
from typing import Any

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src import config
from src.casos import ESTADOS_VALIDOS, PRIORIDADES_VALIDAS, ErrorCaso, RepositorioCasos

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

# (accion, resumen de lo que se escribiría) -> True solo si el humano aprobó.
Confirmador = Callable[[str, dict[str, Any]], bool]


class ResultadoHerramienta(BaseModel):
    """Contrato común de toda herramienta del agente."""

    ok: bool
    datos: Any = None
    error: str | None = None

    def como_json(self) -> str:
        """Serializa el contrato para el `ToolMessage` del bucle del agente."""
        return self.model_dump_json()


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
                 limite: int = 10) -> ResultadoHerramienta:
    try:
        casos = repositorio.listar_casos(estado=estado, limite=limite)
    except ErrorCaso as exc:
        return _desde_error(exc)
    return ResultadoHerramienta(ok=True, datos={"total": len(casos), "casos": casos})


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
        """Guarda un hecho duradero del usuario para futuras conversaciones."""
        try:
            hecho = memoria.recordar(clave, valor)
        except Exception as exc:  # noqa: BLE001 - la memoria no debe abortar el bucle
            return _desde_error(exc).como_json()
        return ResultadoHerramienta(ok=True, datos=hecho).como_json()

    return recordar_tool


def crear_herramientas(repositorio: RepositorioCasos, motor: Any,
                       confirmador: Confirmador | None = None) -> list[BaseTool]:
    """Construye las tools con sus dependencias ya inyectadas.

    `motor` solo necesita exponer `recuperar(pregunta, k=...)`; los tests
    inyectan un doble y la aplicación pasa el `RAGEngine` real.

    `confirmador` es la única vía hacia una escritura. Si se omite, las tools
    de escritura responden `ok=False`: el default seguro no es "preguntar
    después", es "no escribir".
    """

    @tool("buscar_documentacion", args_schema=ArgsBuscarDocumentacion)
    def buscar_documentacion_tool(pregunta: str, k: int = 3) -> str:
        """Busca en la documentación oficial de TecnoApoyo (FAQ y procedimientos).

        Devuelve los fragmentos recuperados y si la evidencia alcanza el umbral
        del motor RAG. Si `evidencia_suficiente` es False, el agente debe
        abstenerse: no hay respaldo documental para responder.
        """
        return buscar_documentacion(motor, pregunta, k).como_json()

    @tool("consultar_caso", args_schema=ArgsConsultarCaso)
    def consultar_caso_tool(id_caso: str) -> str:
        """Devuelve un caso de soporte por su identificador."""
        return consultar_caso(repositorio, id_caso).como_json()

    @tool("listar_casos", args_schema=ArgsListarCasos)
    def listar_casos_tool(estado: str | None = None, limite: int = 10) -> str:
        """Lista casos de soporte, opcionalmente filtrados por estado."""
        return listar_casos(repositorio, estado, limite).como_json()

    @tool("crear_caso", args_schema=ArgsCrearCaso)
    def crear_caso_tool(cliente: str, categoria: str, descripcion: str,
                        prioridad: str = "media",
                        procedimiento: str | None = None) -> str:
        """Propone crear un caso de soporte; la escritura la autoriza el humano."""
        return crear_caso(repositorio, confirmador, cliente, categoria,
                          descripcion, prioridad, procedimiento).como_json()

    @tool("agregar_nota", args_schema=ArgsAgregarNota)
    def agregar_nota_tool(id_caso: str, nota: str) -> str:
        """Propone agregar una nota a un caso; la escritura la autoriza el humano."""
        return agregar_nota(repositorio, confirmador, id_caso, nota).como_json()

    @tool("actualizar_estado", args_schema=ArgsActualizarEstado)
    def actualizar_estado_tool(id_caso: str, estado: str) -> str:
        """Propone cambiar el estado de un caso; la autoriza el humano."""
        return actualizar_estado(repositorio, confirmador, id_caso, estado).como_json()

    @tool("escalar_a_humano", args_schema=ArgsEscalarHumano)
    def escalar_a_humano_tool(id_caso: str, motivo: str) -> str:
        """Propone escalar un caso a una persona; la autoriza el humano."""
        return escalar_a_humano(repositorio, confirmador, id_caso, motivo).como_json()

    @tool("calcular_plazo", args_schema=ArgsCalcularPlazo)
    def calcular_plazo_tool(tipo_servicio: str, fecha_inicio: str | None = None,
                            dias_habiles_extra: int = 0) -> str:
        """Calcula el plazo comprometido de un servicio en días hábiles."""
        return calcular_plazo(tipo_servicio, fecha_inicio,
                              dias_habiles_extra).como_json()

    return [
        buscar_documentacion_tool,
        consultar_caso_tool,
        listar_casos_tool,
        crear_caso_tool,
        agregar_nota_tool,
        actualizar_estado_tool,
        escalar_a_humano_tool,
        calcular_plazo_tool,
    ]


def por_nombre(herramientas: list[BaseTool]) -> dict[str, BaseTool]:
    """Indexa las tools por nombre, que es como las invoca el bucle del agente."""
    return {herramienta.name: herramienta for herramienta in herramientas}
