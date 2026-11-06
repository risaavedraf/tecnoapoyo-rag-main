# -*- coding: utf-8 -*-
"""
Interfaz de consola del asistente documental de TecnoApoyo (Fase 4).

Provee una sesión interactiva con rich: saludo, historial por turno,
trazabilidad opcional por consulta y persistencia de la conversación en
data/output/. Incluye memoria de sesión (machine_prompt del curso) y
exposición de la decisión de abstinencia y los documentos recuperados.

Comandos especiales:
    /trazo on|off      muestra/oculta los documentos recuperados
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
import contextlib
from pathlib import Path

# Reconexión UTF-8 en Windows: evita que los acentos se vean como "?" al
# escribir en la consola o al leer la entrada del usuario.
for flujo in (sys.stdin, sys.stdout, sys.stderr):
    # getattr en lugar de llamar directo: el tipo declarado es TextIO, que no
    # expone reconfigure, y algunos entornos de test entregan flujos que
    # tampoco lo implementan.
    reconectar = getattr(flujo, "reconfigure", None)
    if reconectar is None:
        continue
    with contextlib.suppress(ValueError):  # entornos no reconectables o no tty
        reconectar(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm  # noqa: E402 - mismo bloque que los imports de abajo
from rich.table import Table

from src import config  # noqa: E402
from src.agente import AgenteTecnoApoyo  # noqa: E402
from src.casos import RepositorioCasos  # noqa: E402
from src.herramientas import crear_herramientas  # noqa: E402
from src.memoria import ErrorMemoria, Memoria, crear_resumidor  # noqa: E402
from src.rag import RAGEngine  # noqa: E402

console = Console()
engine = RAGEngine()
# El resumidor de la estrategia 'resumen' usa el mismo modelo del motor y cae
# al resumidor local si el proveedor falla (config.MEMORIA_RESUMIDOR).
memoria = Memoria(resumidor=crear_resumidor(config.MEMORIA_RESUMIDOR, engine.llm))

# El agente se construye al primer uso para no pagar el repositorio de casos
# cuando la sesión solo consulta documentación. Se reconstruye si cambia /modo.
_agente = None
_agente_modo = None

_activo_trazo = True


def _num_chunks():
    try:
        info = json.loads(config.RUTA_DOCUMENTOS.read_text(encoding="utf-8"))
        return info.get("num_chunks", "?")
    except Exception:
        return "?"


def _confirmar_en_consola(accion, resumen):
    """Confirmación humana de una escritura (IL2.1). Sin s, no hay escritura."""
    detalle = "\n".join(f"  {clave}: {valor}" for clave, valor in resumen.items())
    console.print(Panel(f"[bold]{accion}[/bold]\n{detalle}",
                        title="Escritura propuesta", border_style="yellow"))
    return Confirm.ask("¿Confirmas la escritura?", default=False)


def obtener_agente():
    """Construye (o reconstruye si cambió el modo) el agente de soporte."""
    global _agente, _agente_modo
    if _agente is None or _agente_modo != config.AGENTE_MODO:
        repositorio = RepositorioCasos()
        herramientas = crear_herramientas(repositorio, engine,
                                          _confirmar_en_consola)
        _agente = AgenteTecnoApoyo(
            herramientas=herramientas, memoria=memoria, modo=config.AGENTE_MODO,
        )
        _agente_modo = config.AGENTE_MODO
    return _agente


def mostrar_respuesta_agente(respuesta):
    """Impresión de una respuesta del agente con su traza de pasos."""
    if respuesta.error:
        estilo = "bold red"
    elif respuesta.abstencion:
        estilo = "bold yellow"
    else:
        estilo = "bold green"
    console.print(Panel(respuesta.respuesta or "(sin respuesta)",
                        title="Asistente", border_style=estilo))
    if respuesta.error:
        console.print(f"[bold red]Error:[/bold red] {respuesta.error}")
    if _activo_trazo and respuesta.pasos:
        tabla = Table(title=f"Pasos ejecutados ({config.AGENTE_MODO})",
                      show_lines=True, width=console.width)
        tabla.add_column("Herramienta", style="cyan")
        tabla.add_column("Ok", justify="center")
        tabla.add_column("Salida / error")
        for paso in respuesta.pasos:
            tabla.add_row(paso.herramienta, "sí" if paso.ok else "no",
                          str(paso.error or paso.salida)[:80])
        console.print(tabla)
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
    global _activo_trazo
    banner()
    console.print("[dim]Escribe tu consulta. Usa /ayuda para ver los comandos.[/dim]\n")
    ultima = ""
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
                "- `/modo`: muestra el modo; `/modo reactivo|planificado` lo cambia\n"
                "- `/guardar`: guarda la sesión como reanudable (SQLite)\n"
                "- `/sesiones`: lista las sesiones guardadas\n"
                "- `/cargar <id>`: restaura una sesión en el corto plazo\n"
                "- `/memoria`: lista los hechos recordados\n"
                "- `/memoria borrar <clave>`: borra un hecho recordado\n"
                "- `/stats`: resumen de la sesión actual\n"
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
        if ultima.lower() == "/stats":
            n = len(engine.historial)
            abst = sum(1 for h in engine.historial if h["abstencion"])
            console.print(f"[cyan]Consultas: {n} | Abstenciones: {abst} "
                          f"({round(100*abst/max(n,1),1)}%)[/cyan]")
            continue

        with console.status("[cyan]Consultando la documentación...[/cyan]"):
            respuesta = obtener_agente().consultar(ultima)
        # El historial del motor sigue siendo la fuente de /stats y /guardar.
        engine.historial.append({
            "pregunta": respuesta.pregunta,
            "respuesta": respuesta.respuesta,
            "abstencion": respuesta.abstencion,
            "error": respuesta.error,
        })
        mostrar_respuesta_agente(respuesta)


if __name__ == "__main__":
    main()