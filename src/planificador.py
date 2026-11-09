"""Planificación y ejecución multi-etapa del agente (RA2 / IL2.3).

El bucle reactivo de `src.agente` decide paso a paso: el modelo ve el resultado
de una herramienta y recién ahí elige la siguiente. Este módulo agrega el modo
planificado, donde el modelo **primero** declara un plan explícito en JSON y
**después** el ejecutor lo corre en orden.

Cuatro capas, con responsabilidades separadas:

  1. `PasoPlan` / `Plan` (Pydantic, `extra="forbid"`): el contrato del plan.
  2. `validar_plan`: la allowlist real. Un plan es aceptable solo si todas sus
     herramientas existen, los ids son `pN` únicos, las dependencias existen y
     no forman ciclo, los argumentos validan contra el `args_schema` de la tool
     y cada placeholder `{{pN...}}` referencia un paso declarado en
     `depende_de`. Antes de validar, `normalizar_dependencias` agrega esas
     dependencias implícitas: un placeholder ya obliga a ejecutar antes el paso
     que cita, así que la omisión formal no descarta un plan ejecutable. Un
     plan sin pasos también es aceptable: significa que no hace falta ninguna
     herramienta. Nada de esto se le pide al prompt: el prompt sugiere, el
     validador decide.
  3. `EjecutorPlan`: corre el plan en orden topológico, resuelve los
     placeholders contra las salidas reales e invoca `tool.invoke(...)`, que
     vuelve a validar los argumentos con Pydantic antes de escribir. Un paso
     con `ok=False` corta la ejecución y se reporta en el resultado; no
     propaga la excepción.
  4. `OrquestadorPlanificado`: presupuesto global de pasos, re-planificación
     acotada ante un fallo y veredicto de evidencia. Si el presupuesto o los
     re-plans se agotan, el resultado queda con `error` explícito y sin
     respuesta inventada, igual que el tope del bucle reactivo. Un re-plan que
     no produce un plan válido no descarta los pasos ya ejecutados: se responde
     con ellos y con el fallo de la herramienta que explica el problema.

Códigos estables de `ErrorPlan`. Existen para que el llamador (y los tests)
distingan la causa sin analizar el texto del mensaje:

  - `validacion`: el objeto recibido no es un `Plan` ni se puede convertir en uno.
  - `plan_invalido`: el modelo no produjo un plan aceptable tras agotar los
    intentos de reparación (`Planificador.MAX_INTENTOS`).
  - `herramienta_desconocida`: la herramienta no está en la allowlist real.
  - `tope_de_pasos`: `len(pasos)` supera `PLAN_MAX_PASOS`. Un plan sin pasos
    es válido: es la forma que tiene el modelo de decir que la consulta no
    necesita ninguna herramienta.
  - `id_invalido` / `id_duplicado`: ids que no son `pN` o que se repiten.
  - `dependencia_inexistente`: `depende_de` cita un paso que no existe.
  - `ciclo`: las dependencias no se pueden ordenar.
  - `placeholder_no_declarado`: un `{{pN...}}` cita un paso ausente de `depende_de`.
  - `ruta_desconocida`: el camino del placeholder no empieza por `datos` o la
    clave siguiente no está declarada en `CLAVES_SALIDA` para la tool de ese
    paso. Se detecta antes de ejecutar y entra al ciclo de reparación, no al
    re-plan de ejecución.
  - `pedir_datos_no_solo`: un plan que pregunta con `pedir_datos` debe tener
    ese único paso; el turno termina en la pregunta.
  - `argumentos_invalidos`: los argumentos no validan contra el `args_schema`.
  - `dato_faltante`: al resolver un placeholder, el camino no existe en la salida.

El agotamiento de presupuesto no levanta `ErrorPlan`: queda como
`ResultadoPlanificacion.error` con prefijo `presupuesto:`, porque una consulta
sin pasos disponibles debe devolver traza y no una excepción que el llamador
tenga que capturar para poder informar. Un fallo de ejecución que no sea de
presupuesto se informa en la traza (`pasos`) y no como error del turno: el
resultado de negocio (`ok=False` de la tool) termina el turno respondiendo con
lo ejecutado y sin gastar re-plan; un fallo técnico consume el re-plan
disponible y, agotado ese re-plan, el turno también responde con lo ejecutado.
Una escritura confirmada corta la re-planificación antes de consumirla: el plan
nuevo podría repetir el mismo paso y escribir dos veces el mismo caso (S7), así
que el turno responde con lo que ya se hizo.

Todo es comprobable sin red: el `llm` se inyecta y las tools son las reales
sobre un repositorio temporal y un motor falso.
"""
import json
import re
import time
import types
import typing
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src import capacidades, config
from src.agente import (
    ETIQUETA_CLIENTE_SIN_IDENTIFICAR,
    HERRAMIENTA_BUSQUEDA,
    HERRAMIENTAS_CASO,
    PREGUNTA_CLIENTE,
    REGLAS_DOMINIO,
    PasoAgente,
    _texto_de,
    una_sola_linea,
)
from src.capacidades import normalizar
from src.herramientas import CLAVES_SALIDA, PLAZOS_HABILES, por_nombre

# Id de paso: `p1`, `p2`, ... Sin ceros a la izquierda ni sufijos, para que el
# orden topológico y los mensajes de error hablen el mismo idioma.
PATRON_ID_PASO = re.compile(r"^p[1-9]\d*$")

# Placeholder: `{{p1.datos.id}}`. El camino admite puntos e índices de lista.
PATRON_PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z0-9_]+)\.([^{}]+?)\s*\}\}")

# Índice de lista escrito con corchetes (`casos[0]`), la notación natural del
# modelo junto a la de punto (`casos.0`). `_partes_camino` expande ambos.
PATRON_INDICE = re.compile(r"^(?P<clave>[^\[\]]+)\[(?P<indice>-?\d+)\]$")

# Valor de reemplazo cuando el argumento es un placeholder: se valida contra el
# tipo declarado en el esquema, no contra el valor real (que aún no existe).
SONDA_TEXTO = "sonda"


class ErrorPlan(Exception):
    """Fallo de planificación o de ejecución con un código estable.

    Mismo criterio que `ErrorCaso.codigo`: el texto se muestra, el código se
    programa. Así el orquestador distingue un plan rechazado por validación de
    uno que se quedó sin presupuesto sin parsear mensajes.
    """

    def __init__(self, mensaje: str, codigo: str = "validacion"):
        super().__init__(mensaje)
        self.codigo = codigo
        # Contexto opcional para la poda determinista: la validación lo completa
        # y el planificador lo usa para quitar un argumento opcional cuyo
        # placeholder no tiene contrato.
        self.paso_id: str | None = None
        self.argumento: str | None = None
        self.opcional: bool = False


# --- Contrato del plan ---------------------------------------------------
class PasoPlan(BaseModel):
    """Un paso del plan: qué herramienta, con qué argumentos y de qué depende."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Identificador del paso, con formato pN.")
    herramienta: str = Field(description="Nombre de la tool a invocar.")
    argumentos: dict[str, Any] = Field(
        description="Argumentos de la tool; pueden contener placeholders {{pN...}}."
    )
    depende_de: list[str] = Field(
        default_factory=list,
        description="Ids de pasos cuyas salidas necesita este paso.",
    )
    criterio_exito: str | None = Field(
        default=None,
        description="Qué debe cumplirse para considerar el paso logrado.",
    )


class Plan(BaseModel):
    """Plan completo: objetivo, pasos y versión del contrato."""

    model_config = ConfigDict(extra="forbid")

    objetivo: str = Field(min_length=1, description="Objetivo de la consulta.")
    reflexion: str = Field(
        default="",
        description="Por qué este plan y no otro, en una o dos frases.",
    )
    pasos: list[PasoPlan] = Field(description="Pasos en cualquier orden declarado.")
    version: int = Field(default=1, ge=1)


# --- Resultados ----------------------------------------------------------
@dataclass
class ResultadoPlan:
    """Lo que dejó la ejecución de un plan.

    `pasos` reutiliza `PasoAgente`: una traza de plan y una traza reactiva se
    leen igual y la observabilidad (IL2.4) no necesita dos formatos.
    `salidas` mapea id de paso -> payload JSON ya parseado de la herramienta,
    que es lo que consumen los placeholders. `replanes` queda en 0 en el
    ejecutor: quien re-planifica es el orquestador, que lleva su propio conteo.
    `fallo_negocio` distingue el resultado negativo de la tool (`ok=False` en el
    payload) de un fallo técnico (dato faltante, excepción, argumentos
    inválidos): un plan nuevo no arregla un resultado de negocio.
    """

    pasos: list[PasoAgente] = field(default_factory=list)
    salidas: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    replanes: int = 0
    evidencia_deficiente: bool = False
    presupuesto_usado: int = 0
    fallo_negocio: bool = False


@dataclass
class ResultadoPlanificacion:
    """Resultado de resolver una consulta en modo planificado."""

    pasos: list[PasoAgente] = field(default_factory=list)
    salidas: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    replanes: int = 0
    evidencia_deficiente: bool = False
    presupuesto_usado: int = 0
    # Último plan efectivamente ejecutado: la traza y la comparación de modos
    # necesitan el plan, no solo los pasos sueltos.
    plan: Any = None


# --- Utilidades internas -------------------------------------------------
def _indice_herramientas(herramientas: Any) -> dict[str, BaseTool]:
    """Acepta la lista de tools o un índice ya armado.

    El índice ya construido evita reindexar en cada validación, pero la lista
    es la forma natural de pasarlas desde `crear_herramientas`.
    """
    if isinstance(herramientas, dict):
        return dict(herramientas)
    return por_nombre(list(herramientas))


def _como_plan(plan: Any) -> Plan:
    """Normaliza a `Plan`. Un dict se valida; cualquier otra cosa se rechaza."""
    if isinstance(plan, Plan):
        return plan
    if isinstance(plan, dict):
        try:
            return Plan.model_validate(plan)
        except ValidationError as exc:
            raise ErrorPlan(f"El plan no respeta el contrato: {exc}", "validacion") from exc
    raise ErrorPlan(
        f"Se esperaba un Plan o un dict, se recibió {type(plan).__name__}.",
        "validacion",
    )


def _es_placeholder_completo(valor: str) -> bool:
    coincidencia = PATRON_PLACEHOLDER.fullmatch(valor.strip())
    return coincidencia is not None


def _placeholders(valor: Any) -> list[tuple[str, str]]:
    """Todos los `(paso_id, camino)` de un valor, también en dicts y listas."""
    encontrados: list[tuple[str, str]] = []
    if isinstance(valor, dict):
        for anidado in valor.values():
            encontrados.extend(_placeholders(anidado))
    elif isinstance(valor, list):
        for anidado in valor:
            encontrados.extend(_placeholders(anidado))
    elif isinstance(valor, str):
        for coincidencia in PATRON_PLACEHOLDER.finditer(valor):
            encontrados.append((coincidencia.group(1), coincidencia.group(2).strip()))
    return encontrados


def _sonda(anotacion: Any) -> Any:
    """Valor de relleno del tipo declarado, para validar sin el dato real."""
    origen = typing.get_origin(anotacion)
    if origen is typing.Union or origen is types.UnionType:
        opciones = [arg for arg in typing.get_args(anotacion) if arg is not type(None)]
        return _sonda(opciones[0]) if opciones else None
    if origen in (list, set, tuple, frozenset):
        return []
    if origen is dict:
        return {}
    if anotacion is int:
        return 1
    if anotacion is float:
        return 1.0
    if anotacion is bool:
        return True
    return SONDA_TEXTO


def _anotacion_de(esquema: Any, clave: str) -> Any:
    campos = getattr(esquema, "model_fields", None)
    if not isinstance(campos, dict):
        return str
    campo = campos.get(clave)
    return campo.annotation if campo is not None else str


def _con_sondas(argumentos: dict[str, Any], esquema: Any) -> dict[str, Any]:
    """Reemplaza cada placeholder por una sonda del tipo declarado.

    Sin esto, un `k: int` alimentado por `{{p1.datos.mejor_score}}` fallaría la
    validación Pydantic en `validar_plan` aunque el plan sea correcto.
    """

    def reemplazar(valor: Any, anotacion: Any) -> Any:
        if isinstance(valor, dict):
            return {clave: reemplazar(anidado, str) for clave, anidado in valor.items()}
        if isinstance(valor, list):
            return [reemplazar(anidado, str) for anidado in valor]
        if isinstance(valor, str) and _es_placeholder_completo(valor):
            return _sonda(anotacion)
        return valor

    return {
        clave: reemplazar(valor, _anotacion_de(esquema, clave))
        for clave, valor in argumentos.items()
    }


def _validar_argumentos(paso: PasoPlan, herramienta: BaseTool) -> None:
    """Valida los argumentos con el `args_schema` real de la tool.

    Los errores de los campos que traen un placeholder se descartan: su valor
    definitivo no existe todavía, y quien lo valida de verdad es
    `tool.invoke(...)` durante la ejecución.
    """
    esquema = getattr(herramienta, "args_schema", None)
    if esquema is None or not hasattr(esquema, "model_validate"):
        return
    con_placeholder = {
        clave for clave, valor in paso.argumentos.items() if _placeholders(valor)
    }
    try:
        esquema.model_validate(_con_sondas(paso.argumentos, esquema))
    except ValidationError as exc:
        graves = [
            error
            for error in exc.errors()
            if not (error.get("loc") and error["loc"][0] in con_placeholder)
        ]
        if not graves:
            return
        detalle = "; ".join(
            f"{'.'.join(str(parte) for parte in error['loc']) or 'plan'}: {error['msg']}"
            for error in graves
        )
        raise ErrorPlan(
            f"Argumentos inválidos para '{paso.herramienta}': {detalle}",
            "argumentos_invalidos",
        ) from exc


# --- Orden y resolución de argumentos ------------------------------------
def orden_topologico(plan: Plan | dict) -> list[PasoPlan]:
    """Ordena los pasos por dependencias, con desempate por número de id.

    Recibe un plan ya validado en el uso normal; si las dependencias no se
    pueden satisfacer levanta `ErrorPlan`, nunca devuelve un orden parcial.
    """
    pasos = _como_plan(plan).pasos
    pendientes = {paso.id: paso for paso in pasos}
    orden: list[PasoPlan] = []
    resueltos: set[str] = set()

    for paso in pasos:
        for dependencia in paso.depende_de:
            if dependencia not in pendientes:
                raise ErrorPlan(
                    f"El paso '{paso.id}' depende de '{dependencia}', que no existe.",
                    "dependencia_inexistente",
                )

    while pendientes:
        listos = [
            paso for paso in pendientes.values() if set(paso.depende_de) <= resueltos
        ]
        if not listos:
            faltantes = ", ".join(sorted(pendientes))
            raise ErrorPlan(
                f"El plan tiene un ciclo de dependencias entre: {faltantes}.",
                "ciclo",
            )
        listos.sort(key=lambda paso: int(paso.id[1:]))
        for paso in listos:
            orden.append(paso)
            resueltos.add(paso.id)
            del pendientes[paso.id]
    return orden


def _partes_camino(camino: str) -> list[str]:
    """Parte un camino de placeholder en claves y posiciones de lista.

    Acepta las dos notaciones que un modelo escribe para un índice: con punto
    (`casos.0.id`) y con corchetes (`casos[0].id`). `nombre[1]` se expande a
    `nombre` y `1`; un corchete no numérico queda tal cual, para que tanto la
    validación de claves como la navegación real lo rechacen. Los segmentos
    vacíos se conservan: `datos..casos` no es `datos.casos`.
    """
    partes: list[str] = []
    for bruto in camino.split("."):
        parte = bruto.strip()
        coincidencia = PATRON_INDICE.match(parte)
        if coincidencia is None:
            partes.append(parte)
            continue
        partes.append(coincidencia.group("clave"))
        partes.append(coincidencia.group("indice"))
    return partes


def _resolver_camino(paso_id: str, camino: str, salidas: dict[str, Any]) -> Any:
    """Navega `datos.fragmentos.0.score` o `datos.casos[0].id` en una salida."""
    if paso_id not in salidas:
        raise ErrorPlan(
            f"No hay salida registrada para el paso '{paso_id}'.", "dato_faltante"
        )
    actual = salidas[paso_id]
    for parte in _partes_camino(camino):
        parte = parte.strip()
        if isinstance(actual, dict) and parte in actual:
            actual = actual[parte]
        elif (
            isinstance(actual, list)
            and parte.lstrip("-").isdigit()
            and -len(actual) <= int(parte) < len(actual)
        ):
            actual = actual[int(parte)]
        else:
            raise ErrorPlan(
                f"No existe el camino '{{{{{paso_id}.{camino}}}}}' en las salidas "
                "del plan.",
                "dato_faltante",
            )
    return actual


def resolver_argumentos(argumentos: dict[str, Any],
                        salidas: dict[str, Any]) -> dict[str, Any]:
    """Sustituye los placeholders de `argumentos` por las salidas reales.

    Un valor que es exactamente un placeholder se reemplaza por el dato crudo
    (conserva su tipo: int, dict, lista). Si el placeholder está embebido en un
    texto, la sustitución es textual. Un camino inexistente es
    `ErrorPlan("dato_faltante")`, no un `None` silencioso que la tool aceptaría.
    """

    def resolver(valor: Any) -> Any:
        if isinstance(valor, dict):
            return {clave: resolver(anidado) for clave, anidado in valor.items()}
        if isinstance(valor, list):
            return [resolver(anidado) for anidado in valor]
        if not isinstance(valor, str):
            return valor
        coincidencia = PATRON_PLACEHOLDER.fullmatch(valor.strip())
        if coincidencia is not None:
            return _resolver_camino(coincidencia.group(1),
                                    coincidencia.group(2).strip(), salidas)
        return PATRON_PLACEHOLDER.sub(
            lambda m: str(_resolver_camino(m.group(1), m.group(2).strip(), salidas)),
            valor,
        )

    return {clave: resolver(valor) for clave, valor in argumentos.items()}


def evidencia_deficiente(pasos: list[PasoAgente]) -> bool:
    """Misma regla que el guard de abstención del bucle reactivo.

    Verdadero solo si la última `buscar_documentacion` ejecutada dijo que la
    evidencia no alcanza y ninguna herramienta de caso devolvió datos después.
    """

    def datos_de(paso: PasoAgente) -> dict[str, Any] | None:
        if not paso.salida:
            return None
        try:
            payload = json.loads(paso.salida)
        except (TypeError, ValueError):
            return None
        datos = payload.get("datos") if isinstance(payload, dict) else None
        return datos if isinstance(datos, dict) else None

    deficiente = False
    for paso in pasos:
        if not paso.ok:
            continue
        if paso.herramienta == HERRAMIENTA_BUSQUEDA:
            datos = datos_de(paso)
            if datos is not None and "evidencia_suficiente" in datos:
                deficiente = not bool(datos["evidencia_suficiente"])
        elif paso.herramienta in HERRAMIENTAS_CASO:
            # Un caso recuperado con éxito es evidencia operativa: limpia el
            # veredicto documental previo, igual que en `AgenteTecnoApoyo`.
            deficiente = False
    return deficiente


# --- Validación del plan -------------------------------------------------
def _clave_de_paso(paso_id: str) -> tuple[int, int, str]:
    """Orden numérico para los ids `pN`; el resto queda al final por texto.

    `p10` después de `p9`: como los ids no llevan ceros a la izquierda, la
    longitud del sufijo ordena igual que su valor y evita parsear el número.
    """
    if PATRON_ID_PASO.match(paso_id):
        return (0, len(paso_id), paso_id)
    return (1, 0, paso_id)


def normalizar_dependencias(plan: Plan | dict) -> Plan:
    """Agrega a `depende_de` las dependencias implícitas en los placeholders.

    Un `{{pN.camino}}` ya es una dependencia: el paso no puede ejecutarse antes
    de `pN`. Exigir además que el modelo repita `pN` en `depende_de` convierte
    una omisión formal en un plan descartado, y por eso el planificador gasta
    reintentos que no arreglan nada. La inferencia es determinista: agrega solo
    ids de pasos que existen en el plan, sin duplicar y ordenados por número,
    y devuelve un `Plan` nuevo sin mutar el recibido. `validar_plan` sigue
    rechazando los placeholders sin dependencia, así que el plan normalizado
    es el único aceptable.
    """
    plan = _como_plan(plan)
    existentes = {paso.id for paso in plan.pasos}
    pasos: list[PasoPlan] = []
    for paso in plan.pasos:
        declaradas = list(paso.depende_de)
        vistas = set(declaradas)
        implicitas = {
            paso_id
            for paso_id, _ in _placeholders(paso.argumentos)
            if paso_id in existentes and paso_id not in vistas
        }
        for paso_id in sorted(implicitas, key=_clave_de_paso):
            declaradas.append(paso_id)
        if declaradas == paso.depende_de:
            pasos.append(paso)
        else:
            pasos.append(paso.model_copy(update={"depende_de": declaradas}))
    normalizado = (
        plan if pasos == plan.pasos else plan.model_copy(update={"pasos": pasos})
    )
    return normalizado


def _argumento_del_placeholder(argumentos: dict[str, Any], paso_id: str,
                               camino: str) -> str | None:
    """Clave del argumento que contiene el placeholder, si se puede ubicar."""
    for clave, valor in argumentos.items():
        if (paso_id, camino) in _placeholders(valor):
            return clave
    return None


def _sin_argumento(plan: Plan, paso_id: str, argumento: str) -> Plan:
    """Copia del plan sin un argumento de un paso, sin mutar el original."""
    pasos: list[PasoPlan] = []
    for paso in plan.pasos:
        if paso.id == paso_id and argumento in paso.argumentos:
            argumentos = {
                clave: valor for clave, valor in paso.argumentos.items()
                if clave != argumento
            }
            pasos.append(paso.model_copy(update={"argumentos": argumentos}))
        else:
            pasos.append(paso)
    return plan.model_copy(update={"pasos": pasos})


def _error_de_ruta(paso: PasoPlan, argumento: str | None,
                   opcional: bool, mensaje: str) -> ErrorPlan:
    """ErrorPlan de ruta con el contexto que usa la poda determinista."""
    error = ErrorPlan(mensaje, "ruta_desconocida")
    error.paso_id = paso.id
    error.argumento = argumento
    error.opcional = bool(argumento and opcional)
    return error


def _validar_ruta(paso: PasoPlan, paso_origen: str, camino: str,
                  herramienta_origen: str | None,
                  argumento: str | None = None,
                  opcional: bool = False) -> None:
    """Valida el camino de un placeholder contra lo que la tool declara.

    Solo se revisa el primer nivel después de `datos`: `{{p1.datos.casos.0.id}}`
    y `{{p1.datos.casos[0].id}}` se aceptan porque `casos` está declarado, y lo
    que hay dentro del caso se resuelve con el dato real al ejecutar. Un camino
    que no empieza por `datos`, o cuya primera clave no está declarada, es un
    error de validación (`ruta_desconocida`) con las claves válidas en el
    mensaje: el modelo puede corregirlo con un reintento en vez de gastar un
    re-plan de ejecución.

    Si la tool del paso de origen no está declarada (una tool inyectada por un
    llamador), no hay contrato que contrastar y el camino se acepta: el error
    real aparece al resolverlo.
    """
    claves = CLAVES_SALIDA.get(herramienta_origen or "")
    if claves is None:
        return
    partes = _partes_camino(camino)
    primero = partes[0] if partes else ""
    segundo = partes[1] if len(partes) > 1 else ""
    validas = ", ".join(claves)
    cierre = (
        f" El argumento '{argumento}' es opcional."
        if argumento and opcional else ""
    )
    cierre += (
        " Si ese dato no existe en la salida real, omite el argumento "
        "opcional en vez de inventar otra clave."
    )
    if primero != "datos":
        raise _error_de_ruta(
            paso, argumento, opcional,
            f"El paso '{paso.id}' usa '{{{{{paso_origen}.{camino}}}}}' y toda "
            f"salida de herramienta vive en 'datos': el camino debe empezar "
            f"por '{{{{{paso_origen}.datos.<clave>}}}}'. Claves válidas de "
            f"{herramienta_origen}: {validas}.{cierre}",
        )
    if segundo not in claves:
        raise _error_de_ruta(
            paso, argumento, opcional,
            f"El paso '{paso.id}' usa '{{{{{paso_origen}.{camino}}}}}', pero "
            f"'{herramienta_origen}' no devuelve '{segundo}' en datos. "
            f"Claves válidas de {herramienta_origen}: {validas}.{cierre}",
        )


def validar_plan(plan: Plan | dict, herramientas: Any,
                 max_pasos: int | None = None) -> None:
    """Rechaza un plan que no se pueda ejecutar de forma segura.

    Devuelve `None` cuando el plan es aceptable y `ErrorPlan` con un código
    estable en cualquier otro caso. Un plan sin pasos es aceptable: el modelo
    declara que la consulta no necesita ninguna herramienta. No modifica el
    plan recibido.
    """
    indice = _indice_herramientas(herramientas)
    plan = _como_plan(plan)
    limite = max(
        1, config.PLAN_MAX_PASOS if max_pasos is None else int(max_pasos)
    )

    if len(plan.pasos) > limite:
        raise ErrorPlan(
            f"El plan tiene {len(plan.pasos)} pasos y el tope es {limite}.",
            "tope_de_pasos",
        )

    vistos: set[str] = set()
    for paso in plan.pasos:
        if not PATRON_ID_PASO.match(paso.id):
            raise ErrorPlan(
                f"El id '{paso.id}' no tiene el formato pN (por ejemplo p1).",
                "id_invalido",
            )
        if paso.id in vistos:
            raise ErrorPlan(f"El id '{paso.id}' está repetido.", "id_duplicado")
        vistos.add(paso.id)

    # `pedir_datos` no alimenta ningún paso: si está, el plan entero es esa
    # pregunta. Un plan mixto ejecutaría pasos detrás de una pregunta que el
    # usuario todavía no respondió.
    pedir_datos = [
        paso for paso in plan.pasos if paso.herramienta == "pedir_datos"
    ]
    if pedir_datos and len(plan.pasos) > 1:
        raise ErrorPlan(
            f"El paso '{pedir_datos[0].id}' pide datos con 'pedir_datos', que "
            "debe ser el único paso del plan: el turno termina en esa pregunta "
            "y no ejecuta nada más.",
            "pedir_datos_no_solo",
        )

    for paso in plan.pasos:
        for dependencia in paso.depende_de:
            if dependencia not in vistos:
                raise ErrorPlan(
                    f"El paso '{paso.id}' depende de '{dependencia}', que no existe.",
                    "dependencia_inexistente",
                )

    # Un ciclo hace imposible el orden de ejecución: se detecta acá y no al
    # ejecutar, cuando ya se podrían haber hecho escrituras irreversibles.
    orden_topologico(plan)

    herramienta_de_paso = {paso.id: paso.herramienta for paso in plan.pasos}

    for paso in plan.pasos:
        herramienta = indice.get(paso.herramienta)
        if herramienta is None:
            disponibles = ", ".join(sorted(indice)) or "ninguna"
            raise ErrorPlan(
                f"Herramienta desconocida: '{paso.herramienta}'. "
                f"Herramientas disponibles: {disponibles}.",
                "herramienta_desconocida",
            )
        declarados = set(paso.depende_de)
        for paso_id, camino in _placeholders(paso.argumentos):
            if paso_id not in declarados:
                raise ErrorPlan(
                    f"El paso '{paso.id}' usa '{{{{{paso_id}.{camino}}}}}' pero no "
                    f"declara '{paso_id}' en depende_de.",
                    "placeholder_no_declarado",
                )
            argumento = _argumento_del_placeholder(
                paso.argumentos, paso_id, camino
            )
            campos = getattr(
                getattr(herramienta, "args_schema", None), "model_fields", {}
            )
            campo = campos.get(argumento) if argumento else None
            _validar_ruta(
                paso, paso_id, camino, herramienta_de_paso.get(paso_id),
                argumento=argumento,
                opcional=bool(campo is not None and not campo.is_required()),
            )
        _validar_argumentos(paso, herramienta)


# --- Planificador --------------------------------------------------------
# Ejemplo canónico que se embebe en el prompt. Es un plan completo y válido
# según `validar_plan`: la evaluación real con el modelo mostró que los fallos
# dominantes (placeholders sin `depende_de`, JSON con comillas simples o
# envuelto en markdown, ids mal formados) caen cuando el prompt trae un objeto
# exacto que copiar. Se interpola sin pasar por `str.format`, así que sus
# llaves van simples y no hay que escaparlas.
EJEMPLO_PLAN_CANONICO = """{
  "objetivo": "Informar el plazo de reparación del CASO-0007",
  "reflexion": "El caso ya está identificado y el plazo depende de su fecha de creación.",
  "version": 1,
  "pasos": [
    {
      "id": "p1",
      "herramienta": "consultar_caso",
      "argumentos": {"id_caso": "CASO-0007"},
      "depende_de": [],
      "criterio_exito": "Se obtuvo el caso con su fecha de creación."
    },
    {
      "id": "p2",
      "herramienta": "calcular_plazo",
      "argumentos": {"tipo_servicio": "reparacion", "fecha_inicio": "{{p1.datos.fecha_creacion}}"},
      "depende_de": ["p1"],
      "criterio_exito": "Se obtuvo la fecha estimada de entrega."
    },
    {
      "id": "p3",
      "herramienta": "agregar_nota",
      "argumentos": {"id_caso": "{{p1.datos.id}}", "nota": "Se informó al cliente el plazo estimado de reparación."},
      "depende_de": ["p1"],
      "criterio_exito": "La nota quedó registrada en el caso."
    }
  ]
}"""

# El prompt del planificador es un texto único. `{linea_cliente}` llega ya
# armado por `src.agente` (línea más separación) o vacío cuando no hay cliente:
# el prompt no decide por sí mismo si hay cliente, solo lo muestra. La regla 16
# (`{regla_cliente}`) viaja con esa línea y no antes: sin cliente declarado no
# hay nada que declarar como identificado, y el prompt no debe nombrar una
# línea que el turno no trae.
PROMPT_SISTEMA_PLAN = """Eres el planificador del agente de soporte de TecnoApoyo (reparación de computadores en Santiago de Chile).

Devuelve EXCLUSIVAMENTE un objeto JSON, sin texto antes ni después y sin bloques de markdown: un solo objeto, nada de prosa, comentarios ni cercas ```json.

Forma exacta del objeto:
{{"objetivo": "...", "reflexion": "...", "version": 1, "pasos": [{{"id": "p1", "herramienta": "...", "argumentos": {{}}, "depende_de": [], "criterio_exito": "..."}}]}}

Ejemplo canónico completo y válido de tres pasos. Úsalo como plantilla literal:
{ejemplo}

Reglas del contrato:
1. Máximo {max_pasos} pasos. Los ids son p1, p2, ... correlativos, sin repetir y sin ceros a la izquierda.
2. Solo puedes usar estas herramientas:
{herramientas}
3. REGLA OBLIGATORIA: TODO PLACEHOLDER {{{{pN...}}}} OBLIGA A DECLARAR ESE PASO `pN` EN EL DEPENDE_DE DEL PASO QUE LO USA. Si un paso escribe {{{{p1.datos.id}}}} en sus argumentos, su `depende_de` debe incluir "p1". TODO PLACEHOLDER EXIGE SU DEPENDENCIA DECLARADA: un placeholder sin su dependencia invalida el plan completo con placeholder_no_declarado. Solo puedes referenciar pasos de tu propio plan.
4. Identificadores: solo hay dos formatos válidos. CASO-NNNN (exactamente cuatro dígitos, por ejemplo CASO-0007) para casos, y PROC-NNN (exactamente tres dígitos, por ejemplo PROC-004) para procedimientos. Cualquier otro formato invalida el paso.
5. Los tipos de servicio de calcular_plazo son solo los de la tabla PLAZOS_HABILES, con estos valores exactos:
{plazos}
6. Los argumentos deben respetar el esquema de la herramienta: no agregues campos que no estén declarados.
7. No inventes datos que no estén en el objetivo o el contexto: deja ese paso fuera del plan.
8. Las herramientas de escritura (crear_caso, agregar_nota, actualizar_estado, escalar_a_humano) requieren confirmación humana; el plan solo las propone.
9. Si la documentación no alcanza para responder, incluye buscar_documentacion igual: el veredicto de evidencia lo decide la herramienta, no el plan.
10. Si el mensaje no requiere ninguna herramienta —un saludo, un agradecimiento o una pregunta sobre lo que puedes hacer— devuelve el plan vacío: {{"objetivo": "...", "version": 1, "pasos": []}}. Un plan vacío significa «respondo sin herramientas y sin inventar datos». No lo uses para consultas de casos, plazos, políticas, procedimientos, precios, garantías ni estado de equipos: esas exigen buscar_documentacion, una herramienta de caso o calcular_plazo. Ante la duda, planifica.
11. Si para responder hace falta un dato que el mensaje del turno no trae y que tampoco está en el contexto disponible (por ejemplo el nombre del cliente, el id del caso o una fecha), devuelve un plan de UN SOLO paso con la herramienta pedir_datos y escribe una pregunta breve y concreta en `pregunta`. Ese plan no incluye ninguna otra herramienta: el turno termina mostrando esa pregunta al usuario. Antes de pedir nada, revisa el mensaje del turno, el historial reciente y los hechos recordados en «Contexto disponible»: si el dato ya está ahí, no lo pidas. En alcance global las herramientas no requieren cliente identificado: no pidas un nombre de cliente para listar o consultar casos. No uses pedir_datos si el dato ya está en el mensaje o en el contexto, y no lo uses para saludar, agradecer ni preguntar qué puedes hacer. Ejemplo: si el historial dice «Mi equipo es un Lenovo IdeaPad», la pregunta «¿qué equipo tengo registrado?» ya tiene respuesta; no pidas el nombre del cliente para buscar un dato que el historial ya trae.
12. Incluye `reflexion`: una o dos frases que expliquen por qué este plan y no otro. No repitas el objetivo ni inventes datos operativos; es la justificación que el operador ve junto al plan.
13. `calcular_plazo` usa la fecha de hoy cuando omites `fecha_inicio`: nunca inventes un paso ni un id (por ejemplo `p0`) para obtener la fecha actual. Si el plazo depende de una fecha de inicio que no está en el mensaje ni en el contexto, omite `fecha_inicio` y la herramienta usa el día de hoy. Para saber cuánto demora un servicio o qué plazo fija la política, usa `buscar_documentacion`: `calcular_plazo` calcula una fecha a partir de un inicio y no reemplaza a la documentación.
14. `buscar_documentacion` devuelve solo `fragmentos`, `mejor_score`, `umbral` y `evidencia_suficiente`: no devuelve campos de caso, procedimientos ni `procedimiento`, así que no escribas un placeholder hacia esas claves. Si el dato que pide un argumento opcional no está en el mensaje ni en el contexto, omite el argumento: no lo rellenes con un valor inventado ni con un supuesto.
15. En `crear_caso`, `procedimiento` es opcional y solo se llena con un `PROC-NNN` presente en el mensaje o el contexto. Nunca lo derives de `buscar_documentacion`; si no hay un PROC, omite el argumento.

{linea_cliente}{regla_cliente}{reglas_dominio}

Un plan que identifique a un cliente con un correo, con un nombre recordado o con cualquier dato de memoria es un plan inválido: la identidad sale del mensaje del turno, y el contexto de memoria que recibes no es ese mensaje.
"""

# Regla 16 (S10): continúa la lista de «Reglas del contrato» y solo se interpola
# cuando el prompt ya trae la línea del cliente. El defecto medido: con
# `Cliente de la sesión: Valentina Soto` el planificador seguía proponiendo
# `identificar_cliente(nombre="Valentina Soto")` y la tool lo rechazaba porque
# el nombre no está en el mensaje del turno. `impedir_identificacion_inventada`
# no cubre este caso: el nombre es real, solo es redundante.
REGLA_CLIENTE_IDENTIFICADO = """16. Si el prompt ya declara «Cliente de la sesión: <nombre>», ese cliente ESTÁ identificado: no propongas identificar_cliente ni un paso que lo prepare. La pregunta de lectura se planifica directo con listar_casos —el alcance de la sesión ya filtra por ese cliente, así que no escribas el nombre como argumento; usa `estado="abierto"` si la pregunta es por los casos abiertos— o con consultar_caso si el mensaje trae un CASO-NNNN. identificar_cliente aplica únicamente cuando no hay cliente declarado y la persona escribe su nombre en el mensaje del turno: ahí sí es el primer paso del plan. La regla de dominio de identificar al cliente antes de consultar supone un cliente todavía sin identificar; aquí ya lo está. Proponerlo con el cliente ya declarado se gasta en un paso imposible, porque la herramienta solo acepta el nombre que aparece en el mensaje del turno.

"""

PROMPT_REINTENTO = """El plan que devolviste no pasó la validación. Corrige esos errores y devuelve el
JSON completo otra vez, como un único objeto JSON sin markdown ni texto alrededor.

Errores de validación de los intentos anteriores:
{errores}
"""


def _quitar_cercas(texto: str) -> str:
    """Quita el bloque ```json ... ``` con el que algunos modelos envuelven el JSON."""
    limpio = texto.strip()
    if limpio.startswith("```"):
        limpio = re.sub(r"^```[a-zA-Z]*\s*", "", limpio)
        limpio = re.sub(r"\s*```$", "", limpio)
    return limpio.strip()


def _extraer_objeto(texto: str) -> str:
    """Recorta al primer `{` y al último `}`: el JSON no trae prosa."""
    inicio = texto.find("{")
    fin = texto.rfind("}")
    if inicio == -1 or fin <= inicio:
        return texto
    return texto[inicio:fin + 1]


def _resumen_descripcion(descripcion: str | None, limite: int = 160) -> str:
    """Primer renglón útil de la descripción de una tool, en una sola línea."""
    for linea in (descripcion or "").splitlines():
        limpio = " ".join(linea.split())
        if not limpio:
            continue
        if len(limpio) <= limite:
            return limpio
        return limpio[:limite - 1].rstrip() + "…"
    return ""


# --- guard de identidad del plan (S3) -----------------------------------
# El defecto medido: el planificador propuso `identificar_cliente` con
# `nombre = Ricardo` para una pregunta que nunca mencionó a Ricardo, y el
# `criterio_exito` afirmó que el cliente quedó identificado. La tool falla
# cerrado, pero el turno se gastaba en un paso imposible: la guarda reemplaza
# el plan antes de ejecutarlo y no consume re-plan, porque no es un fallo del
# modelo sino una corrección de contrato.
HERRAMIENTA_IDENTIFICACION = "identificar_cliente"
HERRAMIENTA_PEDIR_DATOS = "pedir_datos"

CRITERIO_PEDIR_NOMBRE = "El usuario declara su nombre completo en el turno."


def nombre_declarado(nombre: Any, *textos: str) -> bool:
    """Verdadero si `nombre` aparece como palabras propias en alguno de `textos`.

    La comparación es sin tildes ni mayúsculas y por palabras contiguas, no por
    subcadena: "Ana" no queda declarado por "mañana" ni "ricardo" por
    "ricardo@testeo.com". Un nombre vacío nunca cuenta como declarado.
    """
    buscado = normalizar(str(nombre or ""))
    if not buscado:
        return False
    disponible = " " + " ".join(
        normalizar(texto) for texto in textos if texto
    ) + " "
    return f" {buscado} " in disponible


def plan_de_pedir_el_nombre() -> Plan:
    """Plan de un paso que pregunta el nombre en vez de inventarlo.

    Usa la misma pregunta canónica que el atajo de identificación del agente:
    el usuario no puede recibir dos redacciones distintas de lo mismo.
    """
    return Plan(
        objetivo="Pedir el nombre del cliente antes de identificarlo",
        reflexion=(
            "El plan iba a identificar a un cliente cuyo nombre no aparece en "
            "el mensaje, en los turnos previos ni en el contexto: se pregunta "
            "el dato en vez de suponerlo."
        ),
        version=1,
        pasos=[
            PasoPlan(
                id="p1",
                herramienta=HERRAMIENTA_PEDIR_DATOS,
                argumentos={"pregunta": PREGUNTA_CLIENTE},
                depende_de=[],
                criterio_exito=CRITERIO_PEDIR_NOMBRE,
            ),
        ],
    )


def impedir_identificacion_inventada(
    plan: Plan, objetivo: str, contexto: str = "",
) -> Plan:
    """S3: reemplaza el plan si identifica con un nombre que nadie declaró.

    El nombre debe aparecer en el mensaje del turno, en los turnos previos o en
    el contexto de memoria (los tres llegan en `objetivo` y `contexto`). Si no
    aparece, el plan completo se reemplaza por un único `pedir_datos`: el resto
    de los pasos dependía de una identificación que no puede ocurrir.
    """
    for paso in plan.pasos:
        if paso.herramienta != HERRAMIENTA_IDENTIFICACION:
            continue
        if nombre_declarado(paso.argumentos.get("nombre"), objetivo, contexto):
            continue
        return plan_de_pedir_el_nombre()
    return plan


class Planificador:
    """Convierte un objetivo en un `Plan` validado, pidiendo JSON al modelo."""

    # Intentos totales del bucle de planificación: el primero más las
    # reparaciones. Vive acá y no como número suelto para que el tope, los
    # mensajes de error y los tests hablen del mismo valor.
    MAX_INTENTOS = 3

    def __init__(self, llm: Any, herramientas: list[BaseTool],
                 max_pasos: int | None = None,
                 callbacks: list | None = None,
                 linea_cliente: str | None = None):
        self.llm = llm
        self.herramientas = list(herramientas)
        self.indice = por_nombre(self.herramientas)
        self.max_pasos = max(
            1, config.PLAN_MAX_PASOS if max_pasos is None else int(max_pasos)
        )
        # Callbacks de observabilidad (por ejemplo el colector de tokens). Sin
        # ellos no se envía la clave `config` y la llamada queda idéntica a la
        # de versiones anteriores.
        self.callbacks = None if callbacks is None else list(callbacks)
        # Línea del cliente de la sesión. Llega armada y saneada por
        # `src.agente.linea_de_cliente`; acá se vuelve a colapsar a una sola
        # línea porque el prompt del sistema no puede recibir un salto de línea
        # desde un dato del usuario (S3).
        self.linea_cliente = una_sola_linea(linea_cliente)
        # Sin cliente identificado la línea también existe —el planificador
        # necesita saber que falta el nombre para pedirlo—, pero no hay ningún
        # cliente declarado al que la regla 16 pueda referirse.
        cliente_identificado = bool(
            self.linea_cliente
            and self.linea_cliente != ETIQUETA_CLIENTE_SIN_IDENTIFICAR
        )
        self.prompt_sistema = PROMPT_SISTEMA_PLAN.format(
            max_pasos=self.max_pasos,
            herramientas=self._describir_herramientas(),
            plazos=self._describir_plazos(),
            ejemplo=EJEMPLO_PLAN_CANONICO,
            # La línea del cliente va antes de las reglas de dominio: el
            # planificador necesita saber a quién ya está atendiendo antes de
            # decidir si pide un nombre o si lista los casos.
            linea_cliente=(
                f"{self.linea_cliente}\n\n" if self.linea_cliente else ""
            ),
            # La regla 16 sigue a la línea del cliente: sin cliente declarado no
            # hay nada que declarar como identificado y el prompt no la incluye.
            regla_cliente=(
                REGLA_CLIENTE_IDENTIFICADO if cliente_identificado else ""
            ),
            # Misma fuente que el prompt del modo reactivo: el dominio no se
            # duplica en dos textos que puedan divergir.
            reglas_dominio=REGLAS_DOMINIO,
        )

    def _describir_herramientas(self) -> str:
        """Lista cada tool con su firma, la forma de `datos` y su propósito.

        El planificador escribe caminos de placeholder y decide cuándo pedir un
        dato faltante: sin la forma de la salida, adivina claves y gasta los
        intentos en errores que el contrato ya conoce.
        """
        lineas = []
        for nombre in sorted(self.indice):
            herramienta = self.indice[nombre]
            esquema = getattr(herramienta, "args_schema", None)
            campos = getattr(esquema, "model_fields", None)
            firma = ", ".join(campos) if isinstance(campos, dict) else ""
            claves = CLAVES_SALIDA.get(nombre)
            contrato = (
                f"datos{{{', '.join(claves)}}}" if claves
                else "sin contrato declarado"
            )
            resumen = _resumen_descripcion(
                getattr(herramienta, "description", None)
            )
            proposito = f": {resumen}" if resumen else ""
            lineas.append(f"   - {nombre}({firma}) -> {contrato}{proposito}")
        return "\n".join(lineas) or "   (sin herramientas)"

    def _describir_plazos(self) -> str:
        """Lista los tipos reales de `PLAZOS_HABILES` tal como los valida la tool."""
        return "\n".join(
            f"   - {tipo}: {dias} {'día hábil' if dias == 1 else 'días hábiles'}"
            for tipo, dias in sorted(PLAZOS_HABILES.items())
        )

    def _construir_mensajes(self, objetivo: str, contexto: str,
                           error_previo: str | None,
                           errores: list[str]) -> list[Any]:
        partes = [f"Objetivo: {objetivo}"]
        if contexto.strip():
            partes.append(f"Contexto disponible:\n{contexto.strip()}")
        if error_previo:
            partes.append(
                "El intento anterior falló en la ejecución; el plan nuevo debe "
                f"evitar ese fallo:\n{error_previo}"
            )
        mensaje = "\n\n".join(partes)
        if errores:
            mensaje = (
                f"{mensaje}\n\n"
                + PROMPT_REINTENTO.format(
                    errores="\n".join(f"- {error}" for error in errores)
                )
            )
        return [
            SystemMessage(content=self.prompt_sistema),
            HumanMessage(content=mensaje),
        ]

    def _config_llamada(self) -> dict[str, Any] | None:
        """Config de la invocación al modelo: solo se arma si hay callbacks."""
        if self.callbacks is None:
            return None
        return {"callbacks": self.callbacks}

    def _invocar(self, mensajes: list[Any]) -> str:
        """Invoca el modelo y devuelve su texto, venga como mensaje o como str."""
        config_llamada = self._config_llamada()
        try:
            if config_llamada is None:
                respuesta = self.llm.invoke(mensajes)
            else:
                respuesta = self.llm.invoke(mensajes, config=config_llamada)
        except TypeError:
            # Dobles de test (y algunos clientes viejos) aceptan solo un string.
            if config_llamada is None:
                respuesta = self.llm.invoke(mensajes[-1].content)
            else:
                respuesta = self.llm.invoke(mensajes[-1].content,
                                            config=config_llamada)
        if isinstance(respuesta, str):
            return respuesta.strip()
        return _texto_de(respuesta)

    def _parsear(self, texto: str) -> Plan:
        limpio = _extraer_objeto(_quitar_cercas(texto))
        try:
            datos = json.loads(limpio)
        except (TypeError, ValueError) as exc:
            raise ErrorPlan(
                f"La respuesta del modelo no es JSON válido: {exc}", "plan_invalido"
            ) from exc
        try:
            return Plan.model_validate(datos)
        except ValidationError as exc:
            raise ErrorPlan(
                f"El plan no respeta el contrato: {exc}", "plan_invalido"
            ) from exc

    def _podar_argumentos_opcionales(self, plan: Plan) -> Plan:
        """Quita argumentos opcionales cuyo placeholder no tiene contrato.

        Es una reparación determinista: si el argumento es opcional y la clave
        pedida no existe en la salida de la tool de origen, no hay dato que
        resolver. Se poda en vez de gastar reintentos del modelo, que ya recibió
        el error y volvió a inventar el camino (T15: `procedimiento` desde
        `buscar_documentacion`).

        La cota sale de los argumentos del plan y no de `max_pasos`: cada vuelta
        quita un argumento, así que una poda con más argumentos podables que
        pasos permitidos termina completa y el plan se acepta sin reintento.
        """
        argumentos = sum(len(paso.argumentos) for paso in plan.pasos)
        for _ in range(argumentos + 1):
            try:
                validar_plan(plan, self.indice, max_pasos=self.max_pasos)
                return plan
            except ErrorPlan as exc:
                if (exc.codigo != "ruta_desconocida" or not exc.opcional
                        or not exc.paso_id or not exc.argumento):
                    raise
                plan = _sin_argumento(plan, exc.paso_id, exc.argumento)
        return plan

    def planificar(self, objetivo: str, contexto: str = "",
                   error_previo: str | None = None) -> Plan:
        """Pide un plan, lo valida y repara con los errores acumulados.

        Hace hasta `MAX_INTENTOS` invocaciones: la primera y dos reparaciones,
        cada una con los errores de validación de todos los intentos previos.
        Antes de validar normaliza las dependencias implícitas de los
        placeholders: el plan aceptado es el normalizado. Un plan sin pasos se
        acepta: el modelo declara que la consulta no necesita herramientas. Si
        ninguno valida, levanta `ErrorPlan("plan_invalido")`. No existe plan
        por defecto: inventar uno sería decidir por el modelo.
        """
        errores: list[str] = []
        ultimo: ErrorPlan | None = None
        for intento in range(1, self.MAX_INTENTOS + 1):
            mensajes = self._construir_mensajes(
                objetivo, contexto, error_previo, errores
            )
            texto = self._invocar(mensajes)
            try:
                plan = normalizar_dependencias(self._parsear(texto))
                plan = self._podar_argumentos_opcionales(plan)
                validar_plan(plan, self.indice, max_pasos=self.max_pasos)
                return plan
            except ErrorPlan as exc:
                ultimo = exc
                # Los errores se acumulan: la reparación n ve todo lo que ya
                # falló, no solo lo último. Sin esto el modelo repite el mismo
                # defecto en el segundo reintento.
                errores.append(f"Intento {intento}: {exc}")
        raise ErrorPlan(
            f"El modelo no produjo un plan válido tras {self.MAX_INTENTOS} "
            f"intentos. Último error "
            f"({ultimo.codigo if ultimo else 'validacion'}): {ultimo}",
            "plan_invalido",
        )


# --- Ejecución -----------------------------------------------------------
class EjecutorPlan:
    """Ejecuta un `Plan` validado contra las tools reales."""

    def __init__(self, herramientas: list[BaseTool]):
        self.herramientas = list(herramientas)
        self.indice = por_nombre(self.herramientas)

    def ejecutar(self, plan: Plan | dict,
                 salidas_previas: dict[str, Any] | None = None) -> ResultadoPlan:
        """Corre los pasos en orden topológico y se detiene en el primer fallo.

        El error no se propaga: queda en `ResultadoPlan.error` con la traza
        completa, igual que el bucle reactivo, para que el orquestador pueda
        re-planificar o informar.
        """
        plan = _como_plan(plan)
        salidas: dict[str, Any] = dict(salidas_previas or {})
        pasos: list[PasoAgente] = []
        error: str | None = None
        fallo_negocio = False

        for paso in orden_topologico(plan):
            numero = len(pasos) + 1
            herramienta = self.indice.get(paso.herramienta)
            if herramienta is None:
                disponibles = ", ".join(sorted(self.indice)) or "ninguna"
                mensaje = (
                    f"Herramienta desconocida: '{paso.herramienta}'. "
                    f"Herramientas disponibles: {disponibles}."
                )
                pasos.append(PasoAgente(numero=numero, herramienta=paso.herramienta,
                                        argumentos=dict(paso.argumentos),
                                        error=mensaje, ok=False, paso_id=paso.id))
                error = mensaje
                break

            try:
                argumentos = resolver_argumentos(paso.argumentos, salidas)
            except ErrorPlan as exc:
                pasos.append(PasoAgente(numero=numero, herramienta=paso.herramienta,
                                        argumentos=dict(paso.argumentos),
                                        error=str(exc), ok=False, paso_id=paso.id))
                error = str(exc)
                break

            inicio_paso = time.time()
            try:
                # `invoke` vuelve a validar con el `args_schema` de la tool
                # (`extra="forbid"`): el plan no puede colar argumentos extra.
                salida = herramienta.invoke(argumentos)
            except Exception as exc:  # noqa: BLE001 - informar, no abortar
                mensaje = f"{type(exc).__name__}: {exc}"
                pasos.append(PasoAgente(numero=numero, herramienta=paso.herramienta,
                                        argumentos=argumentos, error=mensaje,
                                        ok=False, paso_id=paso.id,
                                        duracion_seg=round(time.time() - inicio_paso, 3)))
                error = mensaje
                break

            contenido = (
                salida
                if isinstance(salida, str)
                else json.dumps(salida, ensure_ascii=False, default=str)
            )
            try:
                payload = json.loads(contenido)
            except (TypeError, ValueError):
                payload = None

            if isinstance(payload, dict):
                ok = bool(payload.get("ok", True))
                detalle = payload.get("error")
            else:
                ok, detalle = True, None

            salidas[paso.id] = payload if payload is not None else contenido
            pasos.append(PasoAgente(
                numero=numero,
                herramienta=paso.herramienta,
                argumentos=argumentos,
                salida=contenido,
                error=None if ok else (str(detalle) if detalle else "La herramienta falló."),
                ok=ok,
                # La traza conserva el id del plan: la consola cruza por ahí y
                # no por los argumentos, que aquí ya vienen resueltos.
                paso_id=paso.id,
                # Misma convención que el bucle reactivo (`_ejecutar_paso`): el
                # reloj rodea el `invoke` y el valor se redondea a milésimas.
                # Sin esto un turno planificado declaraba 0.0 en cada paso.
                duracion_seg=round(time.time() - inicio_paso, 3),
            ))
            if not ok:
                error = pasos[-1].error
                # Un `ok=False` del payload es un resultado negativo de negocio:
                # la herramienta respondió lo que sabe y ningún plan nuevo lo
                # cambia. Un fallo técnico (herramienta desconocida, dato
                # faltante, excepción o argumentos inválidos al invocar) no
                # pasa por aquí y sí deja margen a otro plan.
                fallo_negocio = True
                break

        return ResultadoPlan(
            pasos=pasos,
            salidas=salidas,
            error=error,
            replanes=0,
            evidencia_deficiente=evidencia_deficiente(pasos),
            presupuesto_usado=len(pasos),
            fallo_negocio=fallo_negocio,
        )


# --- Orquestación --------------------------------------------------------

def herramientas_de_escritura() -> frozenset[str]:
    """Herramientas declaradas como escritura en `CAPACIDADES` (S11).

    Se derivan de la declaración única del agente en vez de copiarse: una tool
    nueva bajo la capacidad `escrituras` queda cubierta por la guarda de
    re-plan sin editar esta lista, y el evaluador deriva el mismo conjunto para
    no puntuar contra una copia que derive de la declaración. Se lee
    `capacidades.CAPACIDADES` en cada llamada para que la derivación y la
    declaración no puedan separarse.
    """
    declaradas = capacidades.CAPACIDADES
    return frozenset(
        herramienta
        for identificador, _titulo, _descripcion, herramientas in declaradas
        if identificador == "escrituras"
        for herramienta in herramientas
    )


HERRAMIENTAS_ESCRITURA: frozenset[str] = herramientas_de_escritura()


def _escritura_confirmada(pasos: list[PasoAgente]) -> bool:
    """True si alguno de estos pasos escribió en el dominio y se confirmó (S7).

    Solo cuenta `ok=True`: una escritura rechazada o sin confirmación no dejó
    nada que proteger.
    """
    return any(paso.ok and paso.herramienta in HERRAMIENTAS_ESCRITURA
               for paso in pasos)


# Identidad del plan que ejecutó cada paso de la traza. `PasoAgente` vive en el
# bucle reactivo y no declara el campo, así que se marca como atributo añadido
# en vez de tocar su contrato: la igualdad de la dataclass queda intacta.
ATRIBUTO_INTENTO_PLAN = "intento_plan"


def marcar_intento_plan(pasos: list[PasoAgente], intento: int) -> None:
    """Marca cada paso con el número del plan que lo ejecutó (1, 2, ...).

    Un turno con re-plan acumula en la traza lo ejecutado por el plan
    descartado y lo ejecutado por el plan final, y los ids (`p1`, `p2`, ...) se
    repiten en cada plan. Sin esta marca la consola solo puede cruzar por id o
    por argumentos, y un paso homónimo hereda el veredicto de un plan que ya no
    es el que se muestra.
    """
    for paso in pasos:
        setattr(paso, ATRIBUTO_INTENTO_PLAN, intento)


class OrquestadorPlanificado:
    """Coordina planificación, ejecución y re-planificación con tope duro."""

    def __init__(self, planificador: Planificador, ejecutor: EjecutorPlan,
                 max_replanes: int | None = None):
        self.planificador = planificador
        self.ejecutor = ejecutor
        # El presupuesto es el mismo tope del plan: sumar plan y re-plans no
        # puede superar lo que un plan solo ya tiene permitido.
        self.max_pasos = max(
            1, int(getattr(planificador, "max_pasos", config.PLAN_MAX_PASOS))
        )
        self.max_replanes = max(
            0, config.PLAN_MAX_REPLANES if max_replanes is None else int(max_replanes)
        )

    def resolver(self, objetivo: str, contexto: str = "") -> ResultadoPlanificacion:
        """Resuelve el objetivo con presupuesto global y re-plans acotados.

        Nunca lanza por un fallo de ejecución. Solo hay dos desenlaces con
        `error`: un primer plan inválido (sin nada ejecutado) y el agotamiento
        del presupuesto (`presupuesto:`). Todo lo demás responde con la traza de
        lo que sí se ejecutó:

          - un `ok=False` de la herramienta es un resultado de negocio y corta el
            turno sin consumir re-plan;
          - una escritura confirmada corta el turno sin consumir re-plan: un plan
            nuevo podría repetirla y duplicar el caso (S7);
          - un fallo técnico consume el re-plan permitido, y agotado ese re-plan
            el turno responde con los pasos ejecutados (`error` en `None`).
        """
        pasos: list[PasoAgente] = []
        salidas: dict[str, Any] = {}
        usados = 0
        replanes = 0
        intento_plan = 0
        error: str | None = None
        error_previo: str | None = None
        ultimo_plan: Any = None

        while True:
            restante = self.max_pasos - usados
            if restante <= 0:
                if pasos:
                    # Tope duro alcanzado con pasos ejecutados (S4): no se pide
                    # otro plan y el turno responde con lo que sí se hizo.
                    break
                error = (
                    f"presupuesto: se agotaron los {self.max_pasos} pasos "
                    f"disponibles para la consulta. Último fallo: {error_previo}"
                )
                break

            try:
                plan = self.planificador.planificar(
                    objetivo, contexto, error_previo=error_previo
                )
            except ErrorPlan as exc:
                if pasos:
                    # Ya se ejecutaron pasos con su traza y el fallo de la tool
                    # explica qué pasó: un re-plan que no valida no convierte
                    # el turno en error ni borra esa evidencia. La síntesis
                    # responde con lo ejecutado, igual que en el bucle reactivo.
                    break
                error = f"{exc.codigo}: {exc}"
                break

            # S3: el plan no puede identificar a un cliente cuyo nombre no
            # aparece en el turno, en los turnos previos ni en el contexto. El
            # reemplazo es de un paso y no consume re-plan: no es un fallo del
            # modelo, es una corrección de contrato antes de tocar nada.
            plan = impedir_identificacion_inventada(plan, objetivo, contexto)

            if len(plan.pasos) > restante:
                if pasos:
                    # El plan nuevo no cabe en el presupuesto que queda: se
                    # responde con lo ejecutado en vez de perderlo.
                    break
                error = (
                    f"presupuesto: el plan de {len(plan.pasos)} pasos excede los "
                    f"{restante} pasos que quedan de {self.max_pasos}."
                )
                break

            # Cada plan que llega a ejecutarse es un intento distinto de la
            # misma consulta: la traza lo registra para que la consola no
            # confunda el `p1` de un re-plan con el `p1` que ya falló.
            intento_plan += 1
            ejecutado = self.ejecutor.ejecutar(plan, salidas_previas=salidas)
            marcar_intento_plan(ejecutado.pasos, intento_plan)
            ultimo_plan = plan
            pasos.extend(ejecutado.pasos)
            salidas.update(ejecutado.salidas)
            usados += len(ejecutado.pasos)

            if ejecutado.error is None:
                break

            if ejecutado.fallo_negocio:
                # Resultado negativo de negocio: la herramienta ya respondió lo
                # que sabe (caso no disponible, cliente inexistente, sin cliente
                # identificado). El turno termina respondiendo con lo ejecutado
                # y no consume re-plan: pedir otro plan solo repite el mismo
                # resultado con más latencia y más tokens.
                break

            if _escritura_confirmada(ejecutado.pasos):
                # S7: la escritura de este plan ya ocurrió y es irreversible. El
                # plan nuevo puede volver a proponer el mismo paso y duplicar el
                # caso, y nada lo impide antes de ejecutarlo, así que no se pide
                # otro plan: el turno responde con lo ejecutado.
                break

            error_previo = ejecutado.error
            if replanes >= self.max_replanes:
                # Fallo técnico sin re-plans disponibles: hay pasos ejecutados
                # con su traza, así que el turno responde con ellos. No es un
                # error visible: el usuario recibe lo que sí se hizo y el
                # detalle del paso que falló.
                break
            if usados >= self.max_pasos:
                # Tope duro alcanzado con un fallo técnico: no se piden más
                # pasos, pero hay traza y el turno responde con ella. Aquí
                # `pasos` nunca está vacío: `usados` cuenta pasos ejecutados.
                break
            replanes += 1

        return ResultadoPlanificacion(
            pasos=pasos,
            salidas=salidas,
            error=error,
            replanes=replanes,
            plan=ultimo_plan,
            evidencia_deficiente=evidencia_deficiente(pasos),
            presupuesto_usado=usados,
        )
