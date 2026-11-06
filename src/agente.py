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
from src.herramientas import crear_herramienta_recordar, por_nombre
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

# Herramientas que aportan datos de un caso concreto. Un caso recuperado con
# éxito es evidencia operativa válida aunque la documentación no cubra la
# pregunta, así que limpia un veredicto documental previo negativo.
HERRAMIENTAS_CASO = ("consultar_caso", "listar_casos")

MENSAJE_SIN_RESPUESTA = "El modelo no devolvió una respuesta final."

# Encabezados de los dos bloques de memoria que viajan en el prompt. Son
# constantes porque el mismo par (historial + hechos) se reutiliza en el modo
# planificado, y dos redacciones distintas serían dos contratos distintos.
ENCABEZADO_HISTORIAL = "Historial reciente:"
ENCABEZADO_HECHOS = "Memoria relevante del usuario:"

# Tope del bloque de historial. Una sesión larga no puede desplazar a los hechos
# del usuario ni al contexto recuperado; el recorte conserva la cola.
LIMITE_HISTORIAL = 1500

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

# Síntesis del modo planificado: el texto final se redacta a partir de lo que
# devolvieron las herramientas, no de lo que el modelo recuerde.
PROMPT_SINTESIS = """Responde al cliente de TecnoApoyo usando SOLO los resultados de las herramientas ejecutadas.

Si los resultados no contienen información suficiente, responde exactamente: "No tengo información suficiente en la documentación de TecnoApoyo para responder eso."

Objetivo del cliente: {objetivo}

Resultados de las herramientas:
{pasos}

Respuesta final, breve y en español, sin explicar tu proceso.
"""


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
    # Duración de la ejecución de la herramienta: la traza necesita latencia
    # por paso, no solo la del turno completo.
    duracion_seg: float = 0.0


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
    # Modo planificado: plan ejecutado y cuántas veces se re-planificó.
    replanes: int = 0
    plan: Any = None


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
        memoria: Any | None = None,
        modo: str | None = None,
    ):
        self.memoria = memoria
        self.herramientas = list(herramientas)
        # La memoria no es solo un bloque de prompt: también aporta una action
        # propia (`recordar`). Se agrega solo si hay memoria y si el llamador no
        # trajo ya una tool con ese nombre.
        if memoria is not None and "recordar" not in {
            herramienta.name for herramienta in self.herramientas
        }:
            self.herramientas.append(crear_herramienta_recordar(memoria))
        self.indice = por_nombre(self.herramientas)
        # Modo de resolución: 'reactivo' es el bucle de herramientas de siempre;
        # 'planificado' pasa por src/planificador.py. Un valor no reconocido no
        # detiene la sesión: cae a reactivo, igual que config.AGENTE_MODO.
        elegido = (modo or config.AGENTE_MODO).strip().lower()
        self.modo = elegido if elegido in config.MODOS_AGENTE else "reactivo"
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

        inicio_paso = time.time()
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
                duracion_seg=round(time.time() - inicio_paso, 3),
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
            duracion_seg=round(time.time() - inicio_paso, 3),
        )
        return paso, contenido, veredicto

    # -- memoria ----------------------------------------------------------
    def _historial_acotado(self) -> str:
        """`corto.texto()` recortado a `LIMITE_HISTORIAL`, conservando la cola.

        Las estrategias (buffer/ventana/resumen) ya están aplicadas dentro de
        `texto()`: aquí solo se pone el tope de tamaño. Se corta por el final
        porque los turnos más nuevos son los que sostienen un seguimiento del
        tipo "¿y en qué estado está?".
        """
        if self.memoria is None:
            return ""
        texto = (self.memoria.corto.texto() or "").strip()
        if len(texto) > LIMITE_HISTORIAL:
            texto = "…" + texto[-(LIMITE_HISTORIAL - 1):]
        return texto

    def _bloque_de_memoria(self, pregunta: str) -> str:
        """Historial reciente y hechos relevantes, en ese orden.

        Devuelve "" cuando no hay memoria, cuando no hay nada que inyectar o
        cuando leerla falla. El historial llega hasta aquí sin el turno en
        curso: `_registrar_turnos` corre después de responder, así que la
        pregunta actual no se duplica en el prompt.
        """
        if self.memoria is None:
            return ""
        partes: list[str] = []
        # Cada mitad se protege por separado: un historial ilegible no debe
        # llevarse por delante los hechos del usuario, que sí son recuperables.
        try:
            historial = self._historial_acotado()
        except Exception:  # noqa: BLE001 - sin memoria el agente sigue operando
            historial = ""
        if historial:
            partes.append(f"{ENCABEZADO_HISTORIAL}\n{historial}")
        try:
            hechos = self.memoria.bloque_para(pregunta)
        except Exception:  # noqa: BLE001 - sin memoria el agente sigue operando
            hechos = ""
        if hechos:
            partes.append(hechos)
        return "\n\n".join(partes)

    def _contexto_de_memoria(self, pregunta: str) -> str:
        """Contexto para el planificador: historial + hechos, igual que el prompt."""
        return self._bloque_de_memoria(pregunta)

    def _prompt_con_memoria(self, pregunta: str) -> str:
        """Prompt del sistema con historial reciente y hechos relevantes."""
        bloque = self._bloque_de_memoria(pregunta)
        return f"{self.prompt_sistema}\n\n{bloque}" if bloque else self.prompt_sistema

    def _registrar_turnos(self, pregunta: str, respuesta: str) -> None:
        """Persiste el turno. Una falla de memoria no tumba la conversación.

        La respuesta vacía no se guarda: en un corte por tope o por error del
        proveedor no hubo turno del asistente, y guardarla dejaría un hueco que
        la recarga leería como respuesta real.
        """
        if self.memoria is None:
            return
        try:
            self.memoria.registrar("usuario", pregunta)
            if respuesta:
                self.memoria.registrar("asistente", respuesta)
        except Exception:  # noqa: BLE001 - la memoria es accesoria, no bloqueante
            return

    # -- ciclo principal --------------------------------------------------
    def consultar(
        self, pregunta: str, return_intermediate_steps: bool = True
    ) -> RespuestaAgente:
        """Resuelve `pregunta` con hasta `max_pasos` iteraciones de herramientas."""
        if self.modo == "planificado":
            return self._consultar_planificado(pregunta, return_intermediate_steps)
        inicio = time.time()
        self.colector_tokens.reset()

        mensajes: list[Any] = [
            SystemMessage(content=self._prompt_con_memoria(pregunta)),
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
            try:
                mensaje = modelo.invoke(mensajes, config=configuracion)
            except Exception as exc:  # noqa: BLE001 - el REPL no debe morir
                # Mismo criterio que el motor RAG: un 401/429/timeout es un
                # error del proveedor, no una abstención ni un cierre de sesión.
                error = f"{type(exc).__name__}: {exc}"
                break
            mensajes.append(mensaje)

            llamadas = list(getattr(mensaje, "tool_calls", None) or [])
            if not llamadas:
                # Sin llamadas pendientes, este texto es la respuesta final.
                respuesta = _texto_de(mensaje)
                if respuesta:
                    finalizado = True
                else:
                    # Una respuesta vacía no es un éxito: no dice nada.
                    error = MENSAJE_SIN_RESPUESTA
                break

            for llamada in llamadas:
                paso, contenido, veredicto = self._ejecutar(llamada, len(pasos) + 1)
                pasos.append(paso)
                if veredicto is not _SIN_VEREDICTO:
                    ultima_evidencia = veredicto
                elif paso.ok and paso.herramienta in HERRAMIENTAS_CASO:
                    # El caso trajo datos: el veredicto documental anterior ya
                    # no decide la respuesta.
                    ultima_evidencia = _SIN_VEREDICTO
                mensajes.append(
                    ToolMessage(
                        content=contenido,
                        tool_call_id=_campo_de(llamada, "id") or f"call_{paso.numero}",
                    )
                )

        if not finalizado and error is None:
            # Tope duro: se corta sin respuesta inventada.
            respuesta = ""
            error = (
                f"Se alcanzó el tope de {self.max_pasos} pasos del agente "
                "sin una respuesta final del modelo."
            )

        if error is not None:
            # Un fallo del proveedor o una respuesta vacía no son abstención:
            # marcarlas así falsearía las métricas de abstención.
            abstencion = False
        else:
            abstencion = respuesta.startswith(PREFIJO_ABSTENCION)
            if ultima_evidencia is False:
                # Guard duro: la última búsqueda no alcanzó el umbral del RAG,
                # así que la abstención se impone sobre lo que diga el modelo.
                respuesta = FRASE_ABSTENCION
                abstencion = True

        if self.memoria is not None:
            self._registrar_turnos(pregunta, respuesta)

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

    # -- modo planificado --------------------------------------------------
    def _consultar_planificado(
        self, pregunta: str, return_intermediate_steps: bool = True
    ) -> RespuestaAgente:
        """Resuelve con planificación explícita y síntesis sobre los resultados.

        La diferencia con el modo reactivo no es cosmética: aquí el modelo
        produce un plan validable antes de tocar ninguna herramienta, y la
        respuesta final se redacta sobre lo que esas herramientas devolvieron.
        """
        from src.planificador import (
            EjecutorPlan,
            OrquestadorPlanificado,
            Planificador,
        )

        inicio = time.time()
        self.colector_tokens.reset()

        modelo = self._llm
        if modelo is None:
            self._modelo()
            modelo = self._llm

        pasos: list[PasoAgente] = []
        plan = None
        replanes = 0
        error: str | None = None
        respuesta = ""
        evidencia_deficiente = False

        try:
            # El plan se acota con PLAN_MAX_PASOS, no con AGENTE_MAX_PASOS: el
            # primero gobierna el plan y el segundo el bucle reactivo. Pasar el
            # tope del bucle dejaba PLAN_MAX_PASOS sin efecto real.
            planificador = Planificador(modelo, self.herramientas)
            orquestador = OrquestadorPlanificado(
                planificador, EjecutorPlan(self.herramientas)
            )
            resultado = orquestador.resolver(
                pregunta, self._contexto_de_memoria(pregunta)
            )
        except Exception as exc:  # noqa: BLE001 - la sesión no debe caer
            error = f"{type(exc).__name__}: {exc}"
        else:
            pasos = list(resultado.pasos)
            plan = resultado.plan
            replanes = resultado.replanes
            evidencia_deficiente = bool(resultado.evidencia_deficiente)
            error = resultado.error
            if error is None:
                respuesta, error = self._sintetizar(modelo, pregunta, resultado)

        abstencion = False
        if error is None:
            if evidencia_deficiente:
                # Mismo guard que el modo reactivo: sin evidencia documental no
                # se responde, aunque la síntesis haya redactado algo.
                respuesta = FRASE_ABSTENCION
                abstencion = True
            elif respuesta.startswith(PREFIJO_ABSTENCION):
                abstencion = True

        if self.memoria is not None:
            self._registrar_turnos(pregunta, respuesta)

        return RespuestaAgente(
            pregunta=pregunta,
            respuesta=respuesta,
            abstencion=abstencion,
            pasos=pasos if return_intermediate_steps else [],
            iteraciones=len(pasos),
            tokens_total=self.colector_tokens.total_tokens,
            latencia_seg=round(time.time() - inicio, 2),
            error=error,
            replanes=replanes,
            plan=plan,
        )

    def _sintetizar(self, modelo: Any, pregunta: str,
                    resultado: Any) -> tuple[str, str | None]:
        """Redacta la respuesta final a partir de las salidas de las tools."""
        lineas = []
        for paso in resultado.pasos:
            salida = paso.salida if paso.ok else (paso.error or "sin salida")
            # Tope por paso: un payload enorme desplazaría al objetivo del
            # prompt y encarecería la síntesis sin aportar.
            lineas.append(f"- {paso.herramienta}: {str(salida)[:800]}")
        detalle = "\n".join(lineas) or "(sin resultados)"
        mensajes = [
            SystemMessage(content=self.prompt_sistema),
            HumanMessage(content=PROMPT_SINTESIS.format(
                objetivo=pregunta, pasos=detalle)),
        ]
        try:
            mensaje = modelo.invoke(
                mensajes, config={"callbacks": [self.colector_tokens]}
            )
        except Exception as exc:  # noqa: BLE001 - error del proveedor, no caída
            return "", f"{type(exc).__name__}: {exc}"
        texto = _texto_de(mensaje)
        if not texto:
            return "", MENSAJE_SIN_RESPUESTA
        return texto, None
