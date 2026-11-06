"""Herramientas del agente de TecnoApoyo (RA2 / IL2.1).

Tres familias, con un contrato de retorno común (`ResultadoHerramienta`):

  - consulta: `buscar_documentacion`, `consultar_caso`, `listar_casos`
  - escritura: `crear_caso`, `agregar_nota`, `actualizar_estado`,
    `escalar_a_humano`
  - razonamiento: `calcular_plazo` (determinista, sin red y sin `eval`)

Reglas que este módulo hace cumplir:
  - El RAG no se reescribe: `buscar_documentacion` delega en
    `RAGEngine.recuperar()` y devuelve el veredicto de evidencia con el mismo
    umbral del motor, de modo que el agente no pueda responder por encima de
    una abstención.
  - Ninguna escritura ocurre sin `confirmado=True`. El parámetro lo entrega la
    consola después de un `s/n` explícito; un modelo que "decida" escribir no
    puede saltarse la confirmación porque el valor por defecto es `False`.
  - Los errores de dominio se devuelven como `ok=False` con el mensaje de la
    allowlist; una excepción nunca llega al bucle del agente.
"""
from datetime import date, timedelta
from typing import Any

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field, field_validator

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


class ResultadoHerramienta(BaseModel):
    """Contrato común de toda herramienta del agente."""

    ok: bool
    datos: Any = None
    error: str | None = None


def _sin_confirmacion(accion: str) -> ResultadoHerramienta:
    return ResultadoHerramienta(
        ok=False, error=MENSAJE_SIN_CONFIRMACION.format(accion=accion)
    )


def _desde_error(exc: Exception) -> ResultadoHerramienta:
    return ResultadoHerramienta(ok=False, error=str(exc))


# --- Contratos de argumentos (validación Pydantic antes de ejecutar) ------
class ArgsBuscarDocumentacion(BaseModel):
    """Argumentos de la búsqueda documental."""

    pregunta: str = Field(min_length=3, max_length=500,
                          description="Pregunta del cliente en lenguaje natural.")
    k: int = Field(default=3, ge=1, le=10,
                   description="Cantidad de fragmentos a recuperar.")


class ArgsConsultarCaso(BaseModel):
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


class ArgsListarCasos(BaseModel):
    """Argumentos del listado de casos."""

    estado: str | None = Field(
        default=None,
        description=f"Filtro opcional. Valores válidos: {', '.join(ESTADOS_VALIDOS)}.",
    )
    limite: int = Field(default=10, ge=1, le=50)


class ArgsCrearCaso(BaseModel):
    """Argumentos de la creación de un caso."""

    cliente: str = Field(min_length=3, max_length=120)
    categoria: str = Field(min_length=3, max_length=60,
                           description="Categoría de soporte, por ejemplo recepcion.")
    descripcion: str = Field(min_length=5, max_length=1000)
    prioridad: str = Field(default="media", description="baja, media o alta.")
    procedimiento: str | None = Field(
        default=None, description="Procedimiento asociado, por ejemplo PROC-001."
    )
    confirmado: bool = Field(
        default=False, description="True solo si el usuario confirmó la escritura."
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


class ArgsAgregarNota(BaseModel):
    """Argumentos para agregar una nota a un caso."""

    id_caso: str = Field(description="Identificador del caso, por ejemplo CASO-0004.")
    nota: str = Field(min_length=3, max_length=1000)
    confirmado: bool = Field(
        default=False, description="True solo si el usuario confirmó la escritura."
    )


class ArgsActualizarEstado(BaseModel):
    """Argumentos para cambiar el estado de un caso."""

    id_caso: str = Field(description="Identificador del caso, por ejemplo CASO-0004.")
    estado: str = Field(description=f"Uno de: {', '.join(ESTADOS_VALIDOS)}.")
    confirmado: bool = Field(
        default=False, description="True solo si el usuario confirmó la escritura."
    )

    @field_validator("estado")
    @classmethod
    def _estado_valido(cls, valor: str) -> str:
        normalizado = (valor or "").strip().lower()
        if normalizado not in ESTADOS_VALIDOS:
            raise ValueError(
                f"Estado inválido. Valores válidos: {', '.join(ESTADOS_VALIDOS)}."
            )
        return normalizado


class ArgsEscalarHumano(BaseModel):
    """Argumentos para escalar un caso a una persona."""

    id_caso: str = Field(description="Identificador del caso, por ejemplo CASO-0003.")
    motivo: str = Field(min_length=3, max_length=500)
    confirmado: bool = Field(
        default=False, description="True solo si el usuario confirmó la escritura."
    )


class ArgsCalcularPlazo(BaseModel):
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


# --- Fábrica de herramientas ligadas al repositorio y al motor -----------
def crear_herramientas(repositorio: RepositorioCasos,
                       motor: Any) -> list[BaseTool]:
    """Construye las tools con sus dependencias ya inyectadas.

    `motor` solo necesita exponer `recuperar(pregunta, k=...)`; los tests
    inyectan un doble y la aplicación pasa el `RAGEngine` real.
    """

    @tool("buscar_documentacion", args_schema=ArgsBuscarDocumentacion)
    def buscar_documentacion(pregunta: str, k: int = 3) -> ResultadoHerramienta:
        """Busca en la documentación oficial de TecnoApoyo (FAQ y procedimientos).

        Devuelve los fragmentos recuperados y si la evidencia alcanza el umbral
        del motor RAG. Si `evidencia_suficiente` es False, el agente debe
        abstenerse: no hay respaldo documental para responder.
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

    @tool("consultar_caso", args_schema=ArgsConsultarCaso)
    def consultar_caso(id_caso: str) -> ResultadoHerramienta:
        """Devuelve un caso de soporte por su identificador."""
        try:
            return ResultadoHerramienta(ok=True, datos=repositorio.obtener_caso(id_caso))
        except ErrorCaso as exc:
            return _desde_error(exc)

    @tool("listar_casos", args_schema=ArgsListarCasos)
    def listar_casos(estado: str | None = None, limite: int = 10) -> ResultadoHerramienta:
        """Lista casos de soporte, opcionalmente filtrados por estado."""
        try:
            casos = repositorio.listar_casos(estado=estado, limite=limite)
        except ErrorCaso as exc:
            return _desde_error(exc)
        return ResultadoHerramienta(ok=True, datos={
            "total": len(casos), "casos": casos,
        })

    @tool("crear_caso", args_schema=ArgsCrearCaso)
    def crear_caso(cliente: str, categoria: str, descripcion: str,
                   prioridad: str = "media", procedimiento: str | None = None,
                   confirmado: bool = False) -> ResultadoHerramienta:
        """Crea un caso de soporte. Requiere confirmación explícita del usuario."""
        if not confirmado:
            return _sin_confirmacion("crear_caso")
        try:
            caso = repositorio.crear_caso(
                cliente=cliente, categoria=categoria, descripcion=descripcion,
                prioridad=prioridad, procedimiento=procedimiento,
            )
        except ErrorCaso as exc:
            return _desde_error(exc)
        return ResultadoHerramienta(ok=True, datos=caso)

    @tool("agregar_nota", args_schema=ArgsAgregarNota)
    def agregar_nota(id_caso: str, nota: str,
                     confirmado: bool = False) -> ResultadoHerramienta:
        """Agrega una nota a un caso. Requiere confirmación explícita del usuario."""
        if not confirmado:
            return _sin_confirmacion("agregar_nota")
        try:
            caso = repositorio.agregar_nota(id_caso, nota)
        except ErrorCaso as exc:
            return _desde_error(exc)
        return ResultadoHerramienta(ok=True, datos=caso)

    @tool("actualizar_estado", args_schema=ArgsActualizarEstado)
    def actualizar_estado(id_caso: str, estado: str,
                          confirmado: bool = False) -> ResultadoHerramienta:
        """Cambia el estado de un caso. Requiere confirmación explícita del usuario."""
        if not confirmado:
            return _sin_confirmacion("actualizar_estado")
        try:
            caso = repositorio.actualizar_estado(id_caso, estado)
        except ErrorCaso as exc:
            return _desde_error(exc)
        return ResultadoHerramienta(ok=True, datos=caso)

    @tool("escalar_a_humano", args_schema=ArgsEscalarHumano)
    def escalar_a_humano(id_caso: str, motivo: str,
                         confirmado: bool = False) -> ResultadoHerramienta:
        """Escala un caso a una persona del equipo. Requiere confirmación."""
        if not confirmado:
            return _sin_confirmacion("escalar_a_humano")
        try:
            caso = repositorio.escalar(id_caso, motivo)
        except ErrorCaso as exc:
            return _desde_error(exc)
        return ResultadoHerramienta(ok=True, datos=caso)

    @tool("calcular_plazo", args_schema=ArgsCalcularPlazo)
    def calcular_plazo_tool(tipo_servicio: str, fecha_inicio: str | None = None,
                            dias_habiles_extra: int = 0) -> ResultadoHerramienta:
        """Calcula el plazo comprometido de un servicio en días hábiles."""
        return calcular_plazo(tipo_servicio, fecha_inicio, dias_habiles_extra)

    return [
        buscar_documentacion,
        consultar_caso,
        listar_casos,
        crear_caso,
        agregar_nota,
        actualizar_estado,
        escalar_a_humano,
        calcular_plazo_tool,
    ]


def por_nombre(herramientas: list[BaseTool]) -> dict[str, BaseTool]:
    """Indexa las tools por nombre, que es como las invoca el bucle del agente."""
    return {herramienta.name: herramienta for herramienta in herramientas}
