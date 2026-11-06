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
     `depende_de`. Nada de esto se le pide al prompt: el prompt sugiere, el
     validador decide.
  3. `EjecutorPlan`: corre el plan en orden topológico, resuelve los
     placeholders contra las salidas reales e invoca `tool.invoke(...)`, que
     vuelve a validar los argumentos con Pydantic antes de escribir. Un paso
     con `ok=False` corta la ejecución y se reporta en el resultado; no
     propaga la excepción.
  4. `OrquestadorPlanificado`: presupuesto global de pasos, re-planificación
     acotada ante un fallo y veredicto de evidencia. Si el presupuesto o los
     re-plans se agotan, el resultado queda con `error` explícito y sin
     respuesta inventada, igual que el tope del bucle reactivo.

Códigos estables de `ErrorPlan`. Existen para que el llamador (y los tests)
distingan la causa sin analizar el texto del mensaje:

  - `validacion`: el objeto recibido no es un `Plan` ni se puede convertir en uno.
  - `plan_invalido`: el modelo no produjo un plan aceptable tras agotar los
    intentos de reparación (`Planificador.MAX_INTENTOS`).
  - `herramienta_desconocida`: la herramienta no está en la allowlist real.
  - `tope_de_pasos`: `len(pasos)` fuera de `[1, PLAN_MAX_PASOS]`.
  - `id_invalido` / `id_duplicado`: ids que no son `pN` o que se repiten.
  - `dependencia_inexistente`: `depende_de` cita un paso que no existe.
  - `ciclo`: las dependencias no se pueden ordenar.
  - `placeholder_no_declarado`: un `{{pN...}}` cita un paso ausente de `depende_de`.
  - `argumentos_invalidos`: los argumentos no validan contra el `args_schema`.
  - `dato_faltante`: al resolver un placeholder, el camino no existe en la salida.

El agotamiento de presupuesto o de re-plans no levanta `ErrorPlan`: queda como
`ResultadoPlanificacion.error` con prefijo `presupuesto:` o `replan_agotado:`,
porque una consulta sin pasos disponibles debe devolver traza y no una
excepción que el llamador tenga que capturar para poder informar.

Todo es comprobable sin red: el `llm` se inyecta y las tools son las reales
sobre un repositorio temporal y un motor falso.
"""
import json
import re
import types
import typing
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src import config
from src.agente import (
    HERRAMIENTA_BUSQUEDA,
    HERRAMIENTAS_CASO,
    PasoAgente,
    _texto_de,
)
from src.herramientas import PLAZOS_HABILES, por_nombre

# Id de paso: `p1`, `p2`, ... Sin ceros a la izquierda ni sufijos, para que el
# orden topológico y los mensajes de error hablen el mismo idioma.
PATRON_ID_PASO = re.compile(r"^p[1-9]\d*$")

# Placeholder: `{{p1.datos.id}}`. El camino admite puntos e índices de lista.
PATRON_PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z0-9_]+)\.([^{}]+?)\s*\}\}")

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
    """

    pasos: list[PasoAgente] = field(default_factory=list)
    salidas: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    replanes: int = 0
    evidencia_deficiente: bool = False
    presupuesto_usado: int = 0


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


def _resolver_camino(paso_id: str, camino: str, salidas: dict[str, Any]) -> Any:
    """Navega `datos.fragmentos.0.score` dentro de la salida de un paso."""
    if paso_id not in salidas:
        raise ErrorPlan(
            f"No hay salida registrada para el paso '{paso_id}'.", "dato_faltante"
        )
    actual = salidas[paso_id]
    for parte in camino.split("."):
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
def validar_plan(plan: Plan | dict, herramientas: Any,
                 max_pasos: int | None = None) -> None:
    """Rechaza un plan que no se pueda ejecutar de forma segura.

    Devuelve `None` cuando el plan es aceptable y `ErrorPlan` con un código
    estable en cualquier otro caso. No modifica el plan recibido.
    """
    indice = _indice_herramientas(herramientas)
    plan = _como_plan(plan)
    limite = max(
        1, config.PLAN_MAX_PASOS if max_pasos is None else int(max_pasos)
    )

    if not plan.pasos:
        raise ErrorPlan("El plan no tiene pasos.", "tope_de_pasos")
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

PROMPT_SISTEMA_PLAN = """Eres el planificador del agente de soporte de TecnoApoyo (reparación de computadores en Santiago de Chile).

Devuelve EXCLUSIVAMENTE un objeto JSON, sin texto antes ni después y sin bloques de markdown: un solo objeto, nada de prosa, comentarios ni cercas ```json.

Forma exacta del objeto:
{{"objetivo": "...", "version": 1, "pasos": [{{"id": "p1", "herramienta": "...", "argumentos": {{}}, "depende_de": [], "criterio_exito": "..."}}]}}

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


class Planificador:
    """Convierte un objetivo en un `Plan` validado, pidiendo JSON al modelo."""

    # Intentos totales del bucle de planificación: el primero más las
    # reparaciones. Vive acá y no como número suelto para que el tope, los
    # mensajes de error y los tests hablen del mismo valor.
    MAX_INTENTOS = 3

    def __init__(self, llm: Any, herramientas: list[BaseTool],
                 max_pasos: int | None = None,
                 callbacks: list | None = None):
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
        self.prompt_sistema = PROMPT_SISTEMA_PLAN.format(
            max_pasos=self.max_pasos,
            herramientas=self._describir_herramientas(),
            plazos=self._describir_plazos(),
            ejemplo=EJEMPLO_PLAN_CANONICO,
        )

    def _describir_herramientas(self) -> str:
        lineas = []
        for nombre in sorted(self.indice):
            esquema = getattr(self.indice[nombre], "args_schema", None)
            campos = getattr(esquema, "model_fields", None)
            firma = ", ".join(campos) if isinstance(campos, dict) else ""
            lineas.append(f"   - {nombre}({firma})")
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

    def planificar(self, objetivo: str, contexto: str = "",
                   error_previo: str | None = None) -> Plan:
        """Pide un plan, lo valida y repara con los errores acumulados.

        Hace hasta `MAX_INTENTOS` invocaciones: la primera y dos reparaciones,
        cada una con los errores de validación de todos los intentos previos.
        Si ninguna valida, levanta `ErrorPlan("plan_invalido")`. No existe plan
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
                plan = self._parsear(texto)
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
                                        error=mensaje, ok=False))
                error = mensaje
                break

            try:
                argumentos = resolver_argumentos(paso.argumentos, salidas)
            except ErrorPlan as exc:
                pasos.append(PasoAgente(numero=numero, herramienta=paso.herramienta,
                                        argumentos=dict(paso.argumentos),
                                        error=str(exc), ok=False))
                error = str(exc)
                break

            try:
                # `invoke` vuelve a validar con el `args_schema` de la tool
                # (`extra="forbid"`): el plan no puede colar argumentos extra.
                salida = herramienta.invoke(argumentos)
            except Exception as exc:  # noqa: BLE001 - informar, no abortar
                mensaje = f"{type(exc).__name__}: {exc}"
                pasos.append(PasoAgente(numero=numero, herramienta=paso.herramienta,
                                        argumentos=argumentos, error=mensaje,
                                        ok=False))
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
            ))
            if not ok:
                error = pasos[-1].error
                break

        return ResultadoPlan(
            pasos=pasos,
            salidas=salidas,
            error=error,
            replanes=0,
            evidencia_deficiente=evidencia_deficiente(pasos),
            presupuesto_usado=len(pasos),
        )


# --- Orquestación --------------------------------------------------------
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

        Nunca lanza por un fallo de ejecución: devuelve `error` explícito y la
        traza de lo que sí se ejecutó. Una respuesta inventada sería peor que
        un error visible.
        """
        pasos: list[PasoAgente] = []
        salidas: dict[str, Any] = {}
        usados = 0
        replanes = 0
        error: str | None = None
        error_previo: str | None = None
        ultimo_plan: Any = None

        while True:
            restante = self.max_pasos - usados
            if restante <= 0:
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
                error = f"{exc.codigo}: {exc}"
                break

            if len(plan.pasos) > restante:
                error = (
                    f"presupuesto: el plan de {len(plan.pasos)} pasos excede los "
                    f"{restante} pasos que quedan de {self.max_pasos}."
                )
                break

            ejecutado = self.ejecutor.ejecutar(plan, salidas_previas=salidas)
            ultimo_plan = plan
            pasos.extend(ejecutado.pasos)
            salidas.update(ejecutado.salidas)
            usados += len(ejecutado.pasos)

            if ejecutado.error is None:
                break

            error_previo = ejecutado.error
            if replanes >= self.max_replanes:
                error = (
                    "replan_agotado: se agotaron las "
                    f"{self.max_replanes} re-planificaciones. "
                    f"Último fallo: {ejecutado.error}"
                )
                break
            if usados >= self.max_pasos:
                error = (
                    f"presupuesto: se agotaron los {self.max_pasos} pasos "
                    f"disponibles para la consulta. Último fallo: {ejecutado.error}"
                )
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
