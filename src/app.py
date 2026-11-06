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

from src import config, credenciales  # noqa: E402
from src.agente import AgenteTecnoApoyo  # noqa: E402
from src.casos import RepositorioCasos  # noqa: E402
from src.herramientas import crear_herramientas  # noqa: E402
from src.memoria import ErrorMemoria, Memoria, crear_resumidor  # noqa: E402
from src.observabilidad import (  # noqa: E402
    escribir_traza,
    fuentes_de,
    leer_trazas,
    resumen_stats,
    traza_de_turno,
)
from src.rag import RAGEngine  # noqa: E402

console = Console()
if not config.LLM_API_KEY:
    # El cliente del proveedor se construye aquí, al importar el módulo: sin
    # clave, el traceback de OpenAI no dice qué hacer. Se corta antes y con la
    # instrucción del comando que la resuelve.
    raise SystemExit(credenciales.MENSAJE_FALTA_CLAVE)
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
    """Impresión de una respuesta del agente con fuentes y pasos."""
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
        tabla.add_column("Error")
        for paso in respuesta.pasos:
            tabla.add_row(paso.herramienta, "sí" if paso.ok else "no",
                          str(paso.error or ""))
        console.print(tabla)

        fuentes = fuentes_de(respuesta.pasos)
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

        with console.status("[cyan]Consultando la documentación...[/cyan]"):
            respuesta = obtener_agente().consultar(ultima)
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
        mostrar_respuesta_agente(respuesta)


if __name__ == "__main__":
    main()