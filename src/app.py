# -*- coding: utf-8 -*-
"""
Interfaz de consola del asistente documental de TecnoApoyo (Fase 4).

Provee una sesión interactiva con rich: saludo, historial por turno,
trazabilidad opcional por consulta y persistencia de la conversación en
data/output/. Incluye memoria de sesión (machine_prompt del curso) y
exposición de la decisión de abstinencia y los documentos recuperados.

Comandos especiales:
    /trazo on|off      muestra/oculta los documentos recuperados
    /plan              muestra el JSON del plan del último turno
    /plan on|off       muestra/oculta el plan al final de cada turno planificado
    /modo              muestra o cambia el modo del agente
    /cliente           muestra el cliente de la sesión
    /cliente <nombre>  acota los casos al cliente indicado
    /cliente borrar    olvida el cliente de la sesión
    /guardar           guarda la sesión como reanudable (SQLite)
    /sesiones          lista las sesiones guardadas
    /cargar <id>       restaura una sesión en la memoria de corto plazo
    /memoria           lista los hechos recordados del usuario
    /memoria borrar <clave>  borra un hecho recordado
    /stats             resumen de la sesión actual
    /salir             termina la sesión

Uso:
    python -m src.app
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rich.console import Console
from rich.live import Live  # noqa: E402 - imports tras el sys.path.insert
from rich.markdown import Markdown
from rich.markup import escape  # noqa: E402 - imports tras el sys.path.insert
from rich.panel import Panel
from rich.syntax import Syntax  # noqa: E402 - imports tras el sys.path.insert
from rich.table import Table
from rich.text import Text  # noqa: E402 - imports tras el sys.path.insert

from src import arranque, config, seleccion  # noqa: E402
from src.agente import AgenteTecnoApoyo  # noqa: E402
from src.casos import RepositorioCasos  # noqa: E402
from src.herramientas import crear_herramientas  # noqa: E402
from src.memoria import ErrorMemoria, Memoria, crear_resumidor  # noqa: E402
from src.planificador import ATRIBUTO_INTENTO_PLAN  # noqa: E402
from src.observabilidad import (  # noqa: E402
    escribir_traza,
    fuentes_de,
    leer_trazas,
    resumen_stats,
    traza_de_turno,
)
from src.rag import RAGEngine  # noqa: E402

console = Console()
# El motor y la memoria se construyen en `preparar_entorno()`, que corre al
# arrancar la consola: importar el módulo no puede exigir la clave ni el índice,
# o un checkout limpio (y la suite) no podría importarlo.
engine = None
memoria = None


def preparar_entorno():
    """Prepara la sesión de consola: flujos, credenciales, motor y memoria.

    Se llama desde `main()`, no al importar el módulo. Sin clave se corta con
    la instrucción que la configura y sin índice FAISS con la que lo construye:
    el traceback del proveedor o del cargador no dice qué hacer.
    """
    global engine, memoria
    # La reconexión UTF-8 y la guarda de credenciales viven en `src.arranque`:
    # la consola y los scripts comparten una sola definición.
    arranque.reconectar_flujos()
    # Un entorno ya preparado (o inyectado por un test) no se reconstruye.
    if engine is not None:
        return
    # Sin clave se corta antes de construir el cliente, con la instrucción del
    # comando que la resuelve; el traceback del proveedor no dice qué hacer.
    arranque.exigir_clave()
    try:
        motor = RAGEngine()
    except FileNotFoundError as error:
        # Sin índice el motor no puede responder: se corta con la instrucción
        # que lo construye, no con un traceback.
        raise SystemExit(str(error)) from None
    engine = motor
    # El resumidor de la estrategia 'resumen' usa el mismo modelo del motor y
    # cae al resumidor local si el proveedor falla (config.MEMORIA_RESUMIDOR).
    memoria = Memoria(
        resumidor=crear_resumidor(config.MEMORIA_RESUMIDOR, engine.llm)
    )


# El agente se construye al primer uso para no pagar el repositorio de casos
# cuando la sesión solo consulta documentación. Se reconstruye si cambia /modo
# o si cambia el cliente de la sesión.
_agente = None
_agente_modo = None
_agente_cliente = None

# Raya con la que se muestran los valores vacíos en la confirmación de una
# escritura. Un `None` de Python no es información para quien confirma.
VALOR_VACIO = "—"

# Nombre de la variable de sesión que vive en el módulo `config`. No está
# declarada en `src/config.py` porque es estado de la consola, no configuración
# del proyecto; aquí queda documentado y se lee con `getattr`.
CLAVE_CLIENTE_SESION = "CLIENTE_ACTUAL"


def cliente_actual() -> str:
    """Cliente identificado en esta sesión (cadena vacía si no hay)."""
    return (getattr(config, CLAVE_CLIENTE_SESION, None) or "").strip()


def establecer_cliente(nombre: str | None) -> None:
    """Fija (o borra, con `None`) el cliente de la sesión.

    Se escribe sobre el módulo `config` en tiempo de ejecución y no en
    `src/config.py`: el cliente es estado de la consola, no configuración del
    proyecto, y `config.py` no declara esta variable. El acceso es por
    `setattr`/`getattr` a propósito, para que el nombre aparezca en un solo
    lugar y no se confunda con un default del proyecto.

    Las consultas siguientes ven el cambio porque `obtener_agente` compara el
    cliente con el que se construyeron las tools y las vuelve a crear: una tool
    que recordara el cliente anterior seguiría filtrando por él.
    """
    setattr(config, CLAVE_CLIENTE_SESION, (nombre or "").strip() or None)


_activo_trazo = True
# Si está activo, el plan validado se muestra al final de cada turno.
_activo_plan = False
# Transmisión del turno en curso. Vive a nivel de módulo porque la confirmación
# de escritura la consulta desde dentro de una herramienta, sin parámetros
# extra entre la tool y la consola: es el único punto donde ambos se cruzan.
_transmision_actual = None


class _TransmisionMarkdown:
    """Dibuja la respuesta como Markdown mientras llega, trozo a trozo.

    Un solo `Live` para todo el turno: se abre con el primer trozo, se redibuja
    con cada uno y se cierra en el `finally` de la consulta. Se abre tarde a
    propósito: un `Live` sin contenido solo taparía la pantalla. El texto
    acumulado se pasa por `Markdown`, de modo que la respuesta se tipografía
    mientras llega y nunca se ven los asteriscos crudos del modelo.

    `detener_para_prompt` lo cierra antes de una confirmación de escritura
    (Rich no admite entrada mientras redibuja). Si todavía no había texto
    pintado, el `Live` se reabre después del menú y la transmisión continúa. Si
    ya había texto, el turno queda marcado como interrumpido: no se vuelve a
    transmitir —el resto de los trozos se acumulan en silencio— y la respuesta
    final se imprime una sola vez, como Markdown, en el panel de siempre. El
    bloque a medias se borra (`transient`) para que ese texto no quede dos
    veces en pantalla.
    """

    def __init__(self, consola):
        self.consola = consola
        self._partes: list[str] = []
        self._vivo = None
        # El turno se cortó por una confirmación con texto ya pintado.
        self.interrumpida = False
        # El `Live` estaba abierto pero vacío cuando llegó la confirmación.
        self._reanudar = False

    @property
    def activo(self) -> bool:
        """Si hay un `Live` redibujando ahora mismo."""
        return self._vivo is not None

    def texto(self) -> str:
        """Todo lo transmitido en el turno, tal como llegó."""
        return "".join(self._partes)

    def agregar(self, trozo: str) -> None:
        """Suma un trozo del stream y refresca el `Live` si corresponde."""
        self._partes.append(trozo)
        self._abrir()
        if self._vivo is not None:
            self._vivo.update(Markdown(self.texto()), refresh=True)

    def _abrir(self) -> None:
        if self._vivo is not None or self.interrumpida:
            return
        self._vivo = Live(
            Markdown(self.texto()),
            console=self.consola,
            auto_refresh=False,
            transient=False,
            # La respuesta completa se queda en pantalla aunque no quepa en el
            # terminal: recortarla por altura perdería texto del turno.
            vertical_overflow="visible",
            redirect_stdout=False,
            redirect_stderr=False,
        )
        self._vivo.start(refresh=True)

    def detener_para_prompt(self) -> None:
        """Cierra el `Live` para que la confirmación pueda leer del teclado."""
        if self._vivo is None:
            return
        if self.texto().strip():
            # Lo ya pintado se borra: la respuesta final llega completa en el
            # panel y duplicarla en pantalla no aporta nada.
            self._vivo.transient = True
            self.interrumpida = True
        else:
            # Un salto de línea no es una respuesta: se reabre tras el menú.
            self._reanudar = True
        self._vivo.stop()
        self._vivo = None

    def reanudar_tras_prompt(self) -> None:
        """Reabre el `Live` si la confirmación lo dejó cerrado y vacío."""
        if not self._reanudar:
            return
        self._reanudar = False
        self._abrir()

    def cerrar(self) -> None:
        """Cierra el `Live` del turno. Deja lo renderizado en pantalla."""
        if self._vivo is not None:
            self._vivo.stop()
            self._vivo = None


def _num_chunks():
    try:
        info = json.loads(config.RUTA_DOCUMENTOS.read_text(encoding="utf-8"))
        return info.get("num_chunks", "?")
    except Exception:
        return "?"


def _valor_visible(valor):
    """Valor de un resumen de escritura, con raya para lo que está vacío.

    La confirmación mostraba `procedimiento: None`: el `None` de Python no dice
    nada a quien decide si autoriza la escritura.
    """
    if valor is None:
        return VALOR_VACIO
    texto = str(valor)
    return texto if texto.strip() else VALOR_VACIO


def _confirmar_en_consola(accion, resumen):
    """Confirmación humana de una escritura (IL2.1). Sin «sí», no hay escritura.

    La pregunta es un menú de flechas (`src.seleccion`), no una línea «s/n»: la
    opción segura queda preseleccionada, Escape y un Enter vacío cancelan, y una
    respuesta inesperada no puede leerse como autorización. Antes de leer se
    cierra el `Live` del streaming, porque Rich no admite entrada mientras
    redibuja: con el spinner abierto la confirmación parecía colgada.
    """
    detalle = "\n".join(
        f"  {clave}: {_valor_visible(valor)}" for clave, valor in resumen.items()
    )
    console.print(Panel(f"[bold]{accion}[/bold]\n{detalle}",
                        title="Escritura propuesta", border_style="yellow"))
    if _transmision_actual is not None:
        _transmision_actual.detener_para_prompt()
    eleccion = seleccion.elegir(
        console,
        "¿Confirmas la escritura?",
        (
            seleccion.Opcion(
                "Sí, escribir", "si",
                descripcion=f"Ejecuta {accion} y la deja registrada.",
            ),
            seleccion.Opcion(
                "No, cancelar", "no", predeterminada=True,
                descripcion="No se escribe nada y la acción se descarta.",
            ),
        ),
        # El cursor arranca en la opción segura: un Enter accidental cancela.
        indice_inicial=1,
    )
    if _transmision_actual is not None:
        _transmision_actual.reanudar_tras_prompt()
    return eleccion.valor == "si"


def obtener_agente():
    """Construye (o reconstruye) el agente de soporte.

    Se reconstruye cuando cambia el modo o el cliente de la sesión: las tools
    llevan el alcance grabado al construirse, así que un cambio de cliente sin
    reconstruirlas dejaría el listado filtrado por el cliente anterior.
    """
    global _agente, _agente_modo, _agente_cliente
    cliente = cliente_actual()
    if (_agente is None or _agente_modo != config.AGENTE_MODO
            or _agente_cliente != cliente):
        repositorio = RepositorioCasos()
        # La consola trabaja siempre con alcance de sesión: sin cliente
        # identificado, `listar_casos` se niega a listar en vez de mostrar los
        # casos de otros clientes.
        herramientas = crear_herramientas(
            repositorio, engine, _confirmar_en_consola,
            alcance="sesion", cliente_actual=cliente_actual,
            al_identificar=establecer_cliente,
        )
        _agente = AgenteTecnoApoyo(
            herramientas=herramientas, memoria=memoria, modo=config.AGENTE_MODO,
            # El mismo lector que reciben las tools: el agente lo consulta en
            # cada turno para poner el cliente en el prompt del sistema y del
            # planificador, y para el atajo determinista de identificación.
            cliente_actual=cliente_actual,
        )
        _agente_modo = config.AGENTE_MODO
        _agente_cliente = cliente
    return _agente


def plan_como_json(plan):
    """JSON legible del plan ejecutado, o None si el turno no tuvo plan."""
    if plan is None:
        return None
    if hasattr(plan, "model_dump"):
        plan = plan.model_dump()
    return json.dumps(plan, ensure_ascii=False, indent=2)


def _plan_como_dict(plan):
    if plan is None:
        return None
    return plan.model_dump() if hasattr(plan, "model_dump") else plan


def _valor_de_argumento(valor):
    """Valor de un argumento en una línea, con vocabulario JSON."""
    if isinstance(valor, (dict, list)):
        return json.dumps(valor, ensure_ascii=False)
    if isinstance(valor, bool):
        return "true" if valor else "false"
    if valor is None:
        return "null"
    return str(valor)


def _clave_de_argumentos(argumentos):
    """Identidad canónica de unos argumentos, para cruzar plan y traza."""
    try:
        return json.dumps(
            argumentos or {}, sort_keys=True, ensure_ascii=False, default=str
        )
    except (TypeError, ValueError):
        return repr(argumentos)


def _pasos_del_ultimo_plan(pasos_ejecutados):
    """Pasos de la traza que ejecutó el plan que la card dibuja.

    La card muestra el último plan del turno, pero `pasos_ejecutados` puede
    traer también lo del plan que se descartó: sin esta depuración un paso que
    el plan mostrado nunca ejecutó hereda el veredicto del paso homónimo del
    intento anterior. Sin marca de intento —modo reactivo, trazas previas— la
    traza se usa tal cual, que es el comportamiento de siempre.
    """
    pasos = list(pasos_ejecutados or [])
    intentos = [getattr(paso, ATRIBUTO_INTENTO_PLAN, None) for paso in pasos]
    conocidos = [intento for intento in intentos if intento is not None]
    if not conocidos:
        return pasos
    ultimo = max(conocidos)
    return [
        paso for paso in pasos
        if getattr(paso, ATRIBUTO_INTENTO_PLAN, None) == ultimo
    ]


def _veredictos_por_paso(pasos_ejecutados):
    """Veredictos de la traza: por `paso_id` y, de respaldo, por argumentos.

    La card se dibuja a partir del plan, que no sabe qué pasó al ejecutarse. La
    traza sí lo sabe y se cruza primero por el id del paso, que el ejecutor
    copia del plan. El respaldo (herramienta, argumentos canónicos) solo cubre
    las trazas sin id: no sirve cuando el plan escribe un placeholder
    —`{{p1.datos.fecha_creacion}}`— porque la traza lleva los argumentos ya
    resueltos y no coinciden. Un paso que no aparece en la traza no tiene
    veredicto: la card no lo inventa.

    Solo cuenta la ejecución del último plan del turno. Con re-plan la traza
    acumula pasos de varios planes y los ids se repiten, así que el veredicto
    de un `p1` descartado marcaría como fallido un `p1` que el plan mostrado sí
    ejecutó bien: el intento viaja en la marca que pone el orquestador.
    """
    por_id = {}
    por_argumentos = {}
    for paso in _pasos_del_ultimo_plan(pasos_ejecutados):
        herramienta = getattr(paso, "herramienta", None)
        if not herramienta:
            continue
        ok = bool(getattr(paso, "ok", False))
        paso_id = getattr(paso, "paso_id", None)
        if paso_id:
            por_id[str(paso_id)] = ok
        clave = (
            str(herramienta),
            _clave_de_argumentos(getattr(paso, "argumentos", None)),
        )
        por_argumentos[clave] = ok
    return por_id, por_argumentos


def _veredicto_de(paso, veredictos_por_id, veredictos_por_argumentos):
    """`ok` del paso según la traza, o None si la traza no lo ejecutó."""
    paso_id = paso.get("id")
    if paso_id is not None and str(paso_id) in veredictos_por_id:
        return veredictos_por_id[str(paso_id)]
    clave = (
        str(paso.get("herramienta", "?")),
        _clave_de_argumentos(paso.get("argumentos")),
    )
    return veredictos_por_argumentos.get(clave)


def plan_como_texto(plan, pasos_ejecutados=None):
    """Card del plan sin sintaxis JSON: pasos, argumentos y criterios.

    `pasos_ejecutados` es la traza del turno: con ella, un paso que se ejecutó
    con `ok=False` se dibuja como fallido en vez de leerse como un hecho
    consumado.
    """
    datos = _plan_como_dict(plan)
    if datos is None:
        return None
    texto = Text()
    objetivo = datos.get("objetivo")
    if objetivo:
        texto.append(f"{objetivo}\n", style="dim")
    reflexion = datos.get("reflexion")
    if reflexion:
        texto.append(f"por qué · {reflexion}\n", style="dim italic")
    pasos = datos.get("pasos") or []
    if not pasos:
        texto.append("La consulta no necesita herramientas.", style="dim italic")
        return texto
    veredictos_por_id, veredictos_por_argumentos = _veredictos_por_paso(
        pasos_ejecutados
    )
    for paso in pasos:
        fallo = _veredicto_de(
            paso, veredictos_por_id, veredictos_por_argumentos
        ) is False
        texto.append("\n")
        texto.append(f"{paso.get('id', '?')}  ", style="dim")
        if fallo:
            texto.append("✗ ", style="bold red")
        texto.append(str(paso.get("herramienta", "?")), style="bold")
        dependencias = paso.get("depende_de") or []
        if dependencias:
            texto.append(
                f"  ← {' '.join(str(d) for d in dependencias)}", style="dim"
            )
        if fallo:
            texto.append("  falló", style="bold red")
        texto.append("\n")
        argumentos = paso.get("argumentos") or {}
        if argumentos:
            for clave_arg, valor in argumentos.items():
                texto.append(f"    {clave_arg} = ", style="dim")
                texto.append(f"{_valor_de_argumento(valor)}\n")
        else:
            texto.append("    sin argumentos\n", style="dim")
        criterio = paso.get("criterio_exito")
        if criterio:
            # Es un criterio, no un hecho: presentarlo como "criterio" a secas
            # hacía que el operador leyera como logrado algo que aún no pasó.
            texto.append(f"    criterio de éxito · {criterio}\n", style="dim italic")
    texto.rstrip()
    return texto


def mostrar_pensamiento(texto, consola=None):
    """Línea de razonamiento del modelo; no se usa si el stream ya la mostró."""
    (consola or console).print(Text(f"· {texto}", style="dim italic"))


def mostrar_plan(plan, consola=None, como_json=False, pasos_ejecutados=None):
    """Dibuja el plan validado (card o JSON) y dice si había uno."""
    destino = consola or console
    datos = _plan_como_dict(plan)
    if datos is None:
        return False
    if como_json:
        texto = plan_como_json(datos)
        if texto is None:
            return False
        destino.print(Panel(
            Syntax(texto, "json", word_wrap=True),
            title="Plan · JSON", title_align="left", border_style="cyan",
        ))
        return True
    pasos = datos.get("pasos") or []
    if pasos:
        titulo = f"Plan · {len(pasos)} {'paso' if len(pasos) == 1 else 'pasos'}"
    else:
        titulo = "Plan · vacío"
    render = plan_como_texto(datos, pasos_ejecutados=pasos_ejecutados)
    if render is None:
        return False
    destino.print(Panel(
        render,
        title=titulo, title_align="left", border_style="cyan",
    ))
    return True


def mostrar_respuesta_agente(respuesta, ya_mostrada=False):
    """Impresión de una respuesta del agente con fuentes y pasos.

    `ya_mostrada` indica que el texto final ya se escribió por streaming: en
    ese caso no se repite en un panel —solo queda el pie con métricas y, si
    `/trazo` está activo, la tabla de fuentes—. Repetirlo duplicaba la respuesta
    en pantalla. El texto se renderiza como Markdown porque el modelo escribe
    Markdown, y de paso los `[...]` del modelo dejan de interpretarse como
    marcado de Rich.
    """
    if respuesta.error:
        estilo = "bold red"
    elif respuesta.abstencion:
        estilo = "bold yellow"
    else:
        estilo = "bold green"
    texto = respuesta.respuesta or "(sin respuesta)"
    if not ya_mostrada or not respuesta.respuesta:
        # Un turno transmitido que terminó sin texto (error temprano) sí
        # necesita panel: lo que se muestra es el aviso, no una respuesta.
        console.print(Panel(Markdown(texto), title="Asistente", border_style=estilo))
    if respuesta.error:
        console.print(f"[bold red]Error:[/bold red] {escape(respuesta.error)}")
    if _activo_trazo and respuesta.pasos and not ya_mostrada:
        # Con streaming cada paso ya se imprimió al ejecutarse: repetir la
        # tabla gastaría pantalla en la misma información.
        tabla = Table(title=f"Pasos ejecutados ({config.AGENTE_MODO})",
                      show_lines=True, width=console.width)
        tabla.add_column("Herramienta", style="cyan")
        tabla.add_column("Ok", justify="center")
        tabla.add_column("Error")
        for paso in respuesta.pasos:
            tabla.add_row(paso.herramienta, "sí" if paso.ok else "no",
                          str(paso.error or ""))
        console.print(tabla)

    fuentes = fuentes_de(respuesta.pasos) if _activo_trazo else []
    if fuentes:
        tabla_fuentes = Table(title="Fuentes documentales", show_lines=True,
                              width=console.width)
        tabla_fuentes.add_column("Documento", style="cyan")
        tabla_fuentes.add_column("Título")
        tabla_fuentes.add_column("Relevancia", justify="right")
        for fuente in fuentes:
            tabla_fuentes.add_row(
                str(fuente.get("document_id", "?")),
                str(fuente.get("titulo", "?")),
                f"{float(fuente.get('score', 0)):.4f}",
            )
        console.print(tabla_fuentes)
    console.print(f"[dim]Modo: {config.AGENTE_MODO} | pasos: {len(respuesta.pasos)} "
                  f"| re-planes: {respuesta.replanes} | "
                  f"tokens: {respuesta.tokens_total} | "
                  f"latencia: {respuesta.latencia_seg}s[/dim]")


def banner():
    console.print(Panel(
        "[bold cyan]Asistente documental de TecnoApoyo[/bold cyan]\n"
        "Asistente de atención al cliente basado en RAG "
        "(LangChain + FAISS + endpoint compatible con OpenAI)\n"
        f"Modelo: [green]{config.LLM_MODELO}[/green] | "
        f"Embeddings: [green]{config.EMBEDDING_MODELO}[/green] | "
        f"Modo: [green]{config.AGENTE_MODO}[/green] | "
        f"Memoria: [green]{config.MEMORIA_ESTRATEGIA}[/green] | "
        f"Chunks: [green]{_num_chunks()}[/green]",
        title="RAG - EP1 Ingeniería de Soluciones con IA",
        border_style="cyan",
    ))


def guardar_sesion():
    """Persiste el historial como sesión reanudable en SQLite (IL2.2).

    Reemplaza la exportación a texto plano: un .txt no se podía recargar, así
    que no era memoria. Los turnos quedan en `sesiones`/`turnos` y se recuperan
    con `/cargar <id>`.
    """
    id_sesion = memoria.abrir_sesion()
    turnos = 0
    for turno in engine.historial:
        memoria.largo.guardar_turno(id_sesion, "usuario", turno["pregunta"])
        memoria.largo.guardar_turno(id_sesion, "asistente", turno["respuesta"])
        turnos += 2
    console.print(f"[green]Sesión guardada:[/green] {id_sesion} ({turnos} turnos). "
                  f"Recupérela con /cargar {id_sesion}")
    return id_sesion


def main():
    # Sin esta declaración la asignación de /trazo creaba una variable local y
    # el módulo nunca se enteraba: /trazo off no ocultaba nada.
    global _activo_trazo, _activo_plan, _transmision_actual
    # Flujos, credenciales y motor: al arrancar, no al importar el módulo.
    preparar_entorno()
    banner()
    console.print("[dim]Escribe tu consulta. Usa /ayuda para ver los comandos.[/dim]\n")
    ultima = ""
    ultimo_plan = None
    # Traza del turno del plan: la card la necesita para marcar los pasos que
    # se ejecutaron con `ok=no`, incluso cuando `/plan` la redibuja después.
    ultimos_pasos = []
    while True:
        try:
            ultima = console.input("[bold magenta]Tú > [/bold magenta]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Sesión finalizada.[/dim]")
            break
        if not ultima:
            continue

        if ultima.lower() in ("/salir", "/exit", "/quit"):
            console.print("[dim]¡Hasta pronto![/dim]")
            break
        if ultima.lower() == "/ayuda":
            console.print(Markdown(
                "- `/trazo on|off`: muestra/oculta los documentos recuperados\n"
                "- `/plan`: card del plan del último turno; `/plan json` muestra "
                "el JSON crudo; `/plan on|off` lo muestra al final de cada "
                "turno planificado\n"
                "- `/modo`: muestra el modo; `/modo reactivo|planificado` lo cambia\n"
                "- `/cliente`: muestra el cliente de la sesión; "
                "`/cliente <nombre>` lo identifica y `/cliente borrar` lo quita\n"
                "- `/guardar`: guarda la sesión como reanudable (SQLite)\n"
                "- `/sesiones`: lista las sesiones guardadas\n"
                "- `/cargar <id>`: restaura una sesión en el corto plazo\n"
                "- `/memoria`: lista los hechos recordados\n"
                "- `/memoria borrar <clave>`: borra un hecho recordado\n"
                "- `/stats`: resumen persistido de consultas, pasos y herramientas\n"
                "- `/trazas`: últimas 10 trazas del día\n"
                "- `/salir`: terminar la sesión"
            ))
            continue
        if ultima.lower().startswith("/trazo"):
            partes = ultima.split()
            # Un argumento ausente o no reconocido no cambia NADA: antes
            # "/trazo" a secas (o "/trazo 1") comparaba contra "on" y apagaba la
            # trazabilidad que el usuario no había pedido tocar.
            if len(partes) < 2 or partes[1].lower() not in ("on", "off"):
                console.print("[dim]Uso: /trazo on | /trazo off[/dim]")
                continue
            _activo_trazo = partes[1].lower() == "on"
            console.print(f"[dim]Trazo: {'activado' if _activo_trazo else 'desactivado'}[/dim]")
            continue
        if ultima.lower() == "/plan" or ultima.lower().startswith("/plan "):
            partes = ultima.split()
            if len(partes) == 1:
                if not mostrar_plan(ultimo_plan, pasos_ejecutados=ultimos_pasos):
                    if config.AGENTE_MODO == "reactivo":
                        aviso = "El modo reactivo no genera plan."
                    else:
                        aviso = (
                            "No hay un plan ejecutado en esta sesión: haz una "
                            "consulta en modo planificado (un plan rechazado "
                            "por validación no llega a ejecutarse)."
                        )
                    console.print(f"[dim]{aviso}[/dim]")
                continue
            if len(partes) == 2 and partes[1].lower() == "json":
                if not mostrar_plan(ultimo_plan, como_json=True,
                                     pasos_ejecutados=ultimos_pasos):
                    if config.AGENTE_MODO == "reactivo":
                        aviso = "El modo reactivo no genera plan."
                    else:
                        aviso = (
                            "No hay un plan ejecutado en esta sesión: haz una "
                            "consulta en modo planificado (un plan rechazado "
                            "por validación no llega a ejecutarse)."
                        )
                    console.print(f"[dim]{aviso}[/dim]")
                continue
            if partes[1].lower() not in ("on", "off"):
                console.print("[dim]Uso: /plan | /plan on | /plan off | /plan json[/dim]")
                continue
            _activo_plan = partes[1].lower() == "on"
            console.print(
                f"[dim]Plan: {'activado' if _activo_plan else 'desactivado'}[/dim]"
            )
            continue
        if ultima.lower() == "/modo" or ultima.lower().startswith("/modo "):
            partes = ultima.split()
            if len(partes) == 1:
                console.print(f"[cyan]Modo actual: {config.AGENTE_MODO}[/cyan]")
                continue
            elegido = partes[1].strip().lower()
            if elegido not in config.MODOS_AGENTE:
                console.print(f"[yellow]Modo '{partes[1]}' no válido. "
                              f"Se mantiene {config.AGENTE_MODO}. "
                              f"Opciones: {', '.join(config.MODOS_AGENTE)}[/yellow]")
                continue
            config.AGENTE_MODO = elegido
            console.print(f"[green]Modo: {elegido}[/green]")
            continue
        if ultima.lower() == "/cliente" or ultima.lower().startswith("/cliente "):
            # Se corta en el primer espacio para admitir nombres con espacios
            # ("/cliente Colegio San Ramón").
            partes = ultima.split(None, 1)
            if len(partes) == 1:
                actual = cliente_actual()
                if actual:
                    console.print(
                        f"[cyan]Cliente de la sesión: {escape(actual)}[/cyan]"
                    )
                else:
                    console.print(
                        "[yellow]Cliente de la sesión: sin identificar. "
                        "Usa /cliente <nombre> para acotar los casos.[/yellow]"
                    )
                continue
            nombre = partes[1].strip()
            if nombre.lower() == "borrar":
                establecer_cliente(None)
                console.print(
                    "[green]Cliente borrado: la sesión queda sin identificar, "
                    "así que no se listarán casos.[/green]"
                )
                continue
            if not nombre:
                console.print(
                    "[dim]Uso: /cliente | /cliente <nombre> | /cliente borrar[/dim]"
                )
                continue
            establecer_cliente(nombre)
            console.print(
                f"[green]Cliente de la sesión: {escape(cliente_actual())}[/green]"
            )
            console.print(
                "[dim]Las consultas de casos quedan acotadas a ese cliente.[/dim]"
            )
            console.print(
                "[yellow]Aviso: /cliente no verifica identidad; "
                "la memoria y el historial son compartidos.[/yellow]"
            )
            continue
        if ultima.lower() == "/memoria" or ultima.lower().startswith("/memoria "):
            partes = ultima.split()
            if len(partes) == 1:
                hechos = memoria.listar_hechos()
                if not hechos:
                    console.print("[dim]No hay hechos recordados.[/dim]")
                    continue
                tabla = Table(title="Memoria de largo plazo", show_lines=True,
                              width=console.width)
                tabla.add_column("Clave", style="cyan")
                tabla.add_column("Valor")
                tabla.add_column("Actualizado", justify="center")
                for hecho in hechos:
                    tabla.add_row(str(hecho["clave"]), str(hecho["valor"]),
                                  str(hecho.get("actualizado_en", "?")))
                console.print(tabla)
                continue
            if len(partes) == 3 and partes[1].lower() == "borrar":
                if memoria.borrar_hecho(partes[2]):
                    console.print(f"[green]Hecho '{partes[2]}' borrado.[/green]")
                else:
                    console.print(f"[yellow]No existe el hecho '{partes[2]}'.[/yellow]")
                continue
            console.print("[dim]Uso: /memoria | /memoria borrar <clave>[/dim]")
            continue
        if ultima.lower() == "/sesiones":
            sesiones = memoria.largo.listar_sesiones()
            if not sesiones:
                console.print("[dim]No hay sesiones guardadas.[/dim]")
                continue
            for sesion in sesiones:
                console.print(f"[cyan]{sesion['id']}[/cyan] "
                              f"({sesion.get('creada_en', '?')})")
            continue
        if ultima.lower().startswith("/cargar"):
            partes = ultima.split()
            if len(partes) != 2:
                console.print("[dim]Uso: /cargar <id>[/dim]")
                continue
            try:
                cargados = memoria.cargar_sesion(partes[1])
            except ErrorMemoria as exc:
                # La carga falla completa: no deja el corto plazo limpio ni una
                # sesión activa inexistente.
                console.print(f"[yellow]{exc}[/yellow]")
                continue
            if cargados:
                console.print(f"[green]{cargados} turnos restaurados de "
                              f"{partes[1]}.[/green]")
            else:
                console.print(f"[yellow]La sesión '{partes[1]}' no tiene turnos.[/yellow]")
            continue
        if ultima.lower().startswith("/guardar"):
            if len(ultima.split()) > 1:
                console.print("[dim]Uso: /guardar (sin argumentos)[/dim]")
                continue
            guardar_sesion()
            continue
        if ultima.lower() == "/trazas":
            trazas = leer_trazas(10)
            if not trazas:
                console.print("[dim]No hay trazas registradas.[/dim]")
                continue
            tabla = Table(title="Últimas 10 trazas", show_lines=True,
                          width=console.width)
            tabla.add_column("Hora", style="cyan")
            tabla.add_column("Modo", justify="center")
            tabla.add_column("Pasos", justify="right")
            tabla.add_column("Tokens", justify="right")
            tabla.add_column("Abst.", justify="center")
            tabla.add_column("Error")
            for traza in trazas:
                error = traza.get("error")
                tabla.add_row(
                    str(traza.get("timestamp", "?"))[11:19],
                    str(traza.get("modo", "?")),
                    str(len(traza.get("pasos") or [])),
                    str(traza.get("tokens", 0)),
                    "sí" if traza.get("abstencion") else "no",
                    str(error)[:40] if error else "",
                )
            console.print(tabla)
            continue
        if ultima.lower() == "/stats":
            datos = resumen_stats()
            console.print(f"[cyan]Consultas: {datos.get('consultas', 0)} | "
                          f"Abstenciones: {datos.get('abstenciones', 0)} | "
                          f"Pasos: {datos.get('pasos', 0)} | "
                          f"Modo actual: {config.AGENTE_MODO}[/cyan]")
            por_herramienta = datos.get("por_herramienta") or {}
            if por_herramienta:
                tabla = Table(title="Distribución de herramientas (trazas)",
                              width=console.width)
                tabla.add_column("Herramienta", style="cyan")
                tabla.add_column("Usos", justify="right")
                for nombre, usos in sorted(por_herramienta.items(),
                                           key=lambda par: par[1], reverse=True):
                    tabla.add_row(str(nombre), str(usos))
                console.print(tabla)
            por_modo = datos.get("por_modo") or {}
            if por_modo:
                resumen_modos = " | ".join(f"{modo}: {usos}"
                                           for modo, usos in sorted(por_modo.items()))
                console.print(f"[dim]Por modo: {resumen_modos}[/dim]")
            continue

        # La consulta ya no corre dentro de `console.status`: Rich Live no
        # admite entrada mientras redibuja, así que la confirmación de escritura
        # se perdía entre el spinner y había que pulsar Enter para soltarla.
        console.print("[dim]Consultando…[/dim]")
        transmision = _TransmisionMarkdown(console) if config.LLM_STREAM else None

        def _paso_ejecutado(paso):
            estado = "sí" if paso.ok else "no"
            console.print(f"[dim]  → {escape(paso.herramienta)} | ok={estado}[/dim]")

        _transmision_actual = transmision
        try:
            respuesta = obtener_agente().consultar(
                ultima,
                al_paso=_paso_ejecutado,
                al_emitir=transmision.agregar if transmision is not None else None,
                al_pensamiento=(
                    mostrar_pensamiento if transmision is None else None
                ),
            )
        finally:
            # El `Live` no sobrevive al turno: ni con error del proveedor ni con
            # una excepción inesperada puede quedar redibujando la pantalla.
            _transmision_actual = None
            if transmision is not None:
                transmision.cerrar()

        # El guard de abstención, un corte del proveedor o una confirmación en
        # medio del stream pueden separar el texto final del transmitido. El
        # panel se omite solo cuando el texto es exactamente el que ya quedó
        # renderizado y el stream llegó completo.
        texto_transmitido = transmision.texto().strip() if transmision else ""
        interrumpida = transmision is not None and transmision.interrumpida
        ya_mostrada = (
            bool(texto_transmitido)
            and not interrumpida
            and respuesta.respuesta == texto_transmitido
        )
        # El historial del motor sigue siendo la fuente de /stats y /guardar.
        engine.historial.append({
            "pregunta": respuesta.pregunta,
            "respuesta": respuesta.respuesta,
            "abstencion": respuesta.abstencion,
            "error": respuesta.error,
            "modo": config.AGENTE_MODO,
            "tokens": respuesta.tokens_total,
            "pasos": [{"herramienta": paso.herramienta, "ok": paso.ok}
                      for paso in respuesta.pasos],
        })
        # Traza por turno (IL2.4). Un fallo de escritura no afecta la respuesta:
        # `escribir_traza` devuelve None en vez de propagar.
        try:
            hechos_inyectados = len(memoria.largo.hechos_relevantes(ultima))
        except Exception:  # noqa: BLE001 - la métrica no puede romper el turno
            hechos_inyectados = 0
        escribir_traza(traza_de_turno(
            respuesta,
            modo=config.AGENTE_MODO,
            estrategia_memoria=config.MEMORIA_ESTRATEGIA,
            hechos_inyectados=hechos_inyectados,
        ))
        mostrar_respuesta_agente(respuesta, ya_mostrada=ya_mostrada)
        ultimo_plan = respuesta.plan
        ultimos_pasos = list(respuesta.pasos)
        if _activo_plan and respuesta.plan is not None:
            # La traza viaja a la card: un paso con `ok=no` se muestra fallido.
            mostrar_plan(respuesta.plan, pasos_ejecutados=respuesta.pasos)


if __name__ == "__main__":
    main()
