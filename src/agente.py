"""Bucle del agente de TecnoApoyo (RA2 / IL2.1).

El agente no genera la respuesta de una sola pasada: usa *function calling*
nativo del proveedor compatible con OpenAI. El modelo decide qué herramientas
llamar, el bucle las ejecuta, devuelve el resultado como `ToolMessage` y el
modelo continúa hasta que deja de pedir herramientas.

Tres invariantes sostienen este módulo, y las tres son del ejecutor, no del
modelo:

  1. Tope duro de pasos (`max_pasos`). Sin él, un modelo que encadena
     llamadas puede girar indefinidamente. Al agotarse el tope el resultado es
     vacío con un error explícito: nunca se inventa una respuesta final.
  2. Guard de abstención. `buscar_documentacion` ya decide si la evidencia
     alcanza el umbral del motor RAG; si la última búsqueda dijo que no
     alcanza, el ejecutor impone `FRASE_ABSTENCION` aunque el modelo insista.
     La abstención queda así por debajo de la generación, no a su merced.
  3. Errores contenidos. Una herramienta inexistente, argumentos que no pasan
     la validación Pydantic o una excepción dentro de la herramienta se
     registran en el paso y el bucle sigue: el error se le informa al modelo,
     que puede corregir el rumbo, en vez de abortar la conversación.

El modelo se inyecta (`llm`), de modo que todo el bucle —`bind_tools`,
ejecución, `ToolMessage`, tope— se puede probar sin red. `ChatOpenAI` solo se
construye si hace falta de verdad, en la primera consulta.
"""
import json
import time
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from pydantic import SecretStr

from src import config
from src.herramientas import por_nombre
from src.rag import ColectorTokens

# Frase canónica de abstención. Es la misma del prompt del motor RAG
# (`rag.PROMPT_FUNDAMENTADO`): una sola frase en todo el asistente para que la
# abstención sea reconocible y evaluable.
FRASE_ABSTENCION = (
    "No tengo información suficiente en la documentación de TecnoApoyo para responder eso."
)

# Prefijo con el que el modelo se abstiene por su cuenta (misma familia de
# redacción que el motor RAG).
PREFIJO_ABSTENCION = "No tengo información suficiente"

# Nombre de la herramienta cuyo veredicto de evidencia controla el guard.
HERRAMIENTA_BUSQUEDA = "buscar_documentacion"

PROMPT_SISTEMA = """Eres el agente de soporte de TecnoApoyo, una pyme de reparación de computadores en Santiago de Chile.

Atiendes consultas de clientes y de recepción usando las herramientas disponibles. Nunca inventes datos, precios, plazos ni políticas.

Herramientas, agrupadas en tres familias:
- Consulta: buscar_documentacion (documentación oficial), consultar_caso y listar_casos (casos de soporte).
- Escritura: crear_caso, agregar_nota, actualizar_estado y escalar_a_humano. Todas requieren confirmación de una persona: tú solo propones la acción, no puedes autorizarla.
- Razonamiento: calcular_plazo, que calcula fechas comprometidas en días hábiles sin adivinar.

Reglas estrictas:
1. Ante preguntas sobre políticas, procedimientos, horarios, precios o garantías, busca primero en la documentación con buscar_documentacion.
2. Si buscar_documentacion devuelve evidencia_suficiente en False, debes abstenerte: responde exactamente "No tengo información suficiente en la documentación de TecnoApoyo para responder eso." y no completes con conocimiento general.
3. Cuando la evidencia alcance, responde en español, breve y claro, citando los datos de los documentos.
4. Usa calcular_plazo para las fechas comprometidas en días hábiles; no calcules plazos por tu cuenta.
5. Si una herramienta responde ok en False, explica el problema al usuario en vez de repetir la misma llamada.
"""

# Centinela interno: distingue "no hubo veredicto de evidencia" de un
# veredicto explícito False, que es el que dispara el guard.
_SIN_VEREDICTO = object()


@dataclass
class PasoAgente:
    """Una ejecución de herramienta, tal como quedó en la traza.

    `salida` es el JSON serializable que devolvió la herramienta (lo mismo que
    vio el modelo en el `ToolMessage`) y `error` explica por qué falló. El
    default de `ok` es False porque un paso sin ejecución confirmada no debe
    leerse como exitoso: el bucle siempre lo escribe de forma explícita.
    """

    numero: int
    herramienta: str
    argumentos: dict
    salida: str | None = None
    error: str | None = None
    ok: bool = False


@dataclass
class RespuestaAgente:
    """Resultado de una consulta al agente, con su traza de pasos."""

    pregunta: str
    respuesta: str
    abstencion: bool
    pasos: list[PasoAgente]
    iteraciones: int
    tokens_total: int
    latencia_seg: float
    error: str | None = None


def _texto_de(mensaje: Any) -> str:
    """Extrae el texto de un `AIMessage`, que puede venir en bloques."""
    contenido = getattr(mensaje, "content", "") or ""
    if isinstance(contenido, str):
        return contenido.strip()
    if isinstance(contenido, list):
        partes = []
        for bloque in contenido:
            if isinstance(bloque, str):
                partes.append(bloque)
            elif isinstance(bloque, dict) and bloque.get("type") == "text":
                partes.append(str(bloque.get("text", "")))
        return "".join(partes).strip()
    return str(contenido).strip()


def _json_error(mensaje: str) -> str:
    """Payload de error que recibe el modelo cuando el paso no se ejecutó."""
    return json.dumps({"ok": False, "datos": None, "error": mensaje}, ensure_ascii=False)


def _campo_de(llamada: Any, campo: str) -> str:
    """Lee `name` o `id` de una llamada venga como objeto o como dict."""
    valor = getattr(llamada, campo, None)
    if valor is None and isinstance(llamada, dict):
        valor = llamada.get(campo)
    return str(valor) if valor is not None else ""


def _args_de(llamada: Any) -> dict[str, Any]:
    """Normaliza los argumentos de una llamada a herramienta."""
    argumentos = getattr(llamada, "args", None)
    if argumentos is None and isinstance(llamada, dict):
        argumentos = llamada.get("args")
    return argumentos if isinstance(argumentos, dict) else {}


class AgenteTecnoApoyo:
    """Ejecutor del ciclo de herramientas sobre las tools de `src.herramientas`."""

    def __init__(
        self,
        herramientas: list[BaseTool],
        llm: Any | None = None,
        max_pasos: int | None = None,
        prompt_sistema: str | None = None,
    ):
        self.herramientas = list(herramientas)
        self.indice = por_nombre(self.herramientas)
        # Un tope <= 0 dejaría al agente sin poder ejecutar ninguna herramienta.
        self.max_pasos = max(
            1, config.AGENTE_MAX_PASOS if max_pasos is None else int(max_pasos)
        )
        self.prompt_sistema = prompt_sistema or PROMPT_SISTEMA
        self.colector_tokens = ColectorTokens()
        self._llm = llm
        # Con un modelo inyectado el agente queda armado desde el minuto uno;
        # el `ChatOpenAI` real se construye más abajo, solo al consultar, para
        # que los tests nunca toquen la red.
        self.llm_con_tools = llm.bind_tools(self.herramientas) if llm is not None else None

    # -- modelo -----------------------------------------------------------
    def _modelo(self) -> Any:
        """Devuelve el modelo con las tools enlazadas, construyéndolo si falta."""
        if self.llm_con_tools is None:
            from langchain_openai import ChatOpenAI

            self._llm = ChatOpenAI(
                model=config.AGENTE_MODELO,
                base_url=config.LLM_BASE_URL,
                api_key=SecretStr(config.LLM_API_KEY),
                temperature=config.TEMPERATURA,
            )
            self.llm_con_tools = self._llm.bind_tools(self.herramientas)
        return self.llm_con_tools

    # -- ejecución de una herramienta -------------------------------------
    def _ejecutar(self, llamada: Any, numero: int) -> tuple[PasoAgente, str, object]:
        """Ejecuta una llamada sin propagar errores.

        Devuelve el paso registrado, el contenido que irá en el `ToolMessage` y
        el veredicto de evidencia (`True`, `False` o el centinela cuando la
        herramienta no opina sobre la suficiencia documental).
        """
        nombre = _campo_de(llamada, "name")
        argumentos = _args_de(llamada)

        herramienta = self.indice.get(nombre)
        if herramienta is None:
            disponibles = ", ".join(sorted(self.indice)) or "ninguna"
            mensaje = (
                f"Herramienta desconocida: '{nombre}'. "
                f"Herramientas disponibles: {disponibles}."
            )
            paso = PasoAgente(
                numero=numero,
                herramienta=nombre,
                argumentos=argumentos,
                salida=None,
                error=mensaje,
                ok=False,
            )
            return paso, _json_error(mensaje), _SIN_VEREDICTO

        try:
            # `invoke` valida con el `args_schema` de la tool antes de ejecutar
            # (Pydantic, `extra="forbid"`): argumentos inválidos no escriben.
            salida = herramienta.invoke(argumentos)
        except Exception as exc:  # noqa: BLE001 - informar al modelo, no abortar
            mensaje = f"{type(exc).__name__}: {exc}"
            paso = PasoAgente(
                numero=numero,
                herramienta=nombre,
                argumentos=argumentos,
                salida=None,
                error=mensaje,
                ok=False,
            )
            return paso, _json_error(mensaje), _SIN_VEREDICTO

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
            error = payload.get("error")
            datos = payload.get("datos")
        else:
            # Una tool que devuelve texto libre no trae veredicto propio.
            ok, error, datos = True, None, None

        veredicto: object = _SIN_VEREDICTO
        if (
            nombre == HERRAMIENTA_BUSQUEDA
            and ok
            and isinstance(datos, dict)
            and "evidencia_suficiente" in datos
        ):
            veredicto = bool(datos["evidencia_suficiente"])

        paso = PasoAgente(
            numero=numero,
            herramienta=nombre,
            argumentos=argumentos,
            salida=contenido,
            error=None if ok else (str(error) if error else "La herramienta falló."),
            ok=ok,
        )
        return paso, contenido, veredicto

    # -- ciclo principal --------------------------------------------------
    def consultar(
        self, pregunta: str, return_intermediate_steps: bool = True
    ) -> RespuestaAgente:
        """Resuelve `pregunta` con hasta `max_pasos` iteraciones de herramientas."""
        inicio = time.time()
        self.colector_tokens.reset()

        mensajes: list[Any] = [
            SystemMessage(content=self.prompt_sistema),
            HumanMessage(content=pregunta),
        ]
        modelo = self._modelo()
        configuracion = {"callbacks": [self.colector_tokens]}

        pasos: list[PasoAgente] = []
        iteraciones = 0
        respuesta = ""
        error: str | None = None
        ultima_evidencia: object = _SIN_VEREDICTO
        finalizado = False

        for _ in range(self.max_pasos):
            iteraciones += 1
            mensaje = modelo.invoke(mensajes, config=configuracion)
            mensajes.append(mensaje)

            llamadas = list(getattr(mensaje, "tool_calls", None) or [])
            if not llamadas:
                # Sin llamadas pendientes, este texto es la respuesta final.
                respuesta = _texto_de(mensaje)
                finalizado = True
                break

            for llamada in llamadas:
                paso, contenido, veredicto = self._ejecutar(llamada, len(pasos) + 1)
                pasos.append(paso)
                if veredicto is not _SIN_VEREDICTO:
                    ultima_evidencia = veredicto
                mensajes.append(
                    ToolMessage(
                        content=contenido,
                        tool_call_id=_campo_de(llamada, "id") or f"call_{paso.numero}",
                    )
                )

        if not finalizado:
            # Tope duro: se corta sin respuesta inventada.
            respuesta = ""
            error = (
                f"Se alcanzó el tope de {self.max_pasos} pasos del agente "
                "sin una respuesta final del modelo."
            )
            abstencion = False
        else:
            abstencion = respuesta.startswith(PREFIJO_ABSTENCION)
            if ultima_evidencia is False:
                # Guard duro: la última búsqueda no alcanzó el umbral del RAG,
                # así que la abstención se impone sobre lo que diga el modelo.
                respuesta = FRASE_ABSTENCION
                abstencion = True

        return RespuestaAgente(
            pregunta=pregunta,
            respuesta=respuesta,
            abstencion=abstencion,
            pasos=pasos if return_intermediate_steps else [],
            iteraciones=iteraciones,
            tokens_total=self.colector_tokens.total_tokens,
            latencia_seg=round(time.time() - inicio, 2),
            error=error,
        )
