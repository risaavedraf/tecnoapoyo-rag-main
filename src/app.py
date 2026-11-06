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
from rich.table import Table

from src import config  # noqa: E402
from src.memoria import ErrorMemoria, Memoria, crear_resumidor  # noqa: E402
from src.rag import RAGEngine, ResultadoConsulta  # noqa: E402

console = Console()
engine = RAGEngine()
# El resumidor de la estrategia 'resumen' usa el mismo modelo del motor y cae
# al resumidor local si el proveedor falla (config.MEMORIA_RESUMIDOR).
memoria = Memoria(resumidor=crear_resumidor(config.MEMORIA_RESUMIDOR, engine.llm))

_activo_trazo = True


def _num_chunks():
    try:
        info = json.loads(config.RUTA_DOCUMENTOS.read_text(encoding="utf-8"))
        return info.get("num_chunks", "?")
    except Exception:
        return "?"


def banner():
    console.print(Panel(
        "[bold cyan]Asistente documental de TecnoApoyo[/bold cyan]\n"
        "Asistente de atención al cliente basado en RAG "
        "(LangChain + FAISS + endpoint compatible con OpenAI)\n"
        f"Modelo: [green]{config.LLM_MODELO}[/green] | "
        f"Embeddings: [green]{config.EMBEDDING_MODELO}[/green] | "
        f"Memoria: [green]{config.MEMORIA_ESTRATEGIA}[/green] | "
        f"Chunks: [green]{_num_chunks()}[/green]",
        title="RAG - EP1 Ingeniería de Soluciones con IA",
        border_style="cyan",
    ))


def mostrar_respuesta(resultado: ResultadoConsulta):
    """Muestra la respuesta y, si está activo el trazo, los documentos usados."""
    # La consola no pasa por rag.mostrar_resultado, así que el error del
    # proveedor hay que mostrarlo aquí o el usuario solo vería el mensaje
    # genérico sin saber que la llamada al LLM falló.
    if resultado.error:
        estilo = "bold red"
    elif resultado.abstencion:
        estilo = "bold yellow"
    else:
        estilo = "bold green"
    console.print(Panel(resultado.respuesta,
                        title="Asistente", border_style=estilo))
    if resultado.error:
        console.print(f"[bold red]Error del proveedor:[/bold red] "
                      f"{resultado.error}")

    if _activo_trazo:
        tabla = Table(title="Trazabilidad - documentos recuperados",
                      show_lines=True, width=console.width)
        tabla.add_column("Documento", style="cyan")
        tabla.add_column("Relevancia", justify="right")
        tabla.add_column("Versión", justify="center")
        tabla.add_column("Fecha", justify="center")
        for doc in resultado.documentos_recuperados:
            tabla.add_row(doc["document_id"], f"{doc['score']:.4f}",
                          f"v{doc['version']}", doc["fecha"])
        console.print(tabla)
        console.print(f"[dim]Latencia: {resultado.latencia_seg}s | "
                      f"Tokens: {resultado.tokens_total} "
                      f"({resultado.tokens_entrada} in / "
                      f"{resultado.tokens_salida} out)[/dim]")


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
            resultado = engine.consultar(ultima)
        mostrar_respuesta(resultado)


if __name__ == "__main__":
    main()