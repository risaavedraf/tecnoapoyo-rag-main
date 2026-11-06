# -*- coding: utf-8 -*-
"""
Interfaz de consola del asistente documental de TecnoApoyo (Fase 4).

Provee una sesión interactiva con rich: saludo, historial por turno,
trazabilidad opcional por consulta y persistencia de la conversación en
data/output/. Incluye memoria de sesión (machine_prompt del curso) y
exposición de la decisión de abstinencia y los documentos recuperados.

Comandos especiales:
    /trazo on|off      muestra/oculta los documentos recuperados
    /guardar [nombre]  exporta la sesión a data/output/sesiones/
    /resultados        imprime las 3 consultas finales de evaluación
    /salir             termina la sesión

Uso:
    python -m src.app
"""
import sys
import json
from datetime import datetime
from pathlib import Path

# Reconexión UTF-8 en Windows: evita que los acentos se vean como "?" al
# escribir en la consola o al leer la entrada del usuario.
for flujo in (sys.stdin, sys.stdout, sys.stderr):
    try:
        flujo.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):  # entornos no reconectables o no tty
        pass

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from src import config
from src.rag import RAGEngine, ResultadoConsulta

console = Console()
engine = RAGEngine()

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
        "Asistente de atención al cliente basado en RAG (LangChain + FAISS + Ollama)\n"
        f"Modelo: [green]{config.LLM_MODELO}[/green] | "
        f"Embeddings: [green]{config.EMBEDDING_MODELO}[/green] | "
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


def guardar_sesion(nombre="sesion"):
    """Persiste la sesión en data/output/sesiones/<nombre>_<fecha>.txt."""
    dir_salida = config.DIR_OUTPUT / "sesiones"
    dir_salida.mkdir(parents=True, exist_ok=True)
    # Segundos en la marca de tiempo: dos /guardar con el mismo nombre dentro
    # del mismo minuto generaban el mismo archivo y el segundo pisaba al
    # primero, perdiendo la sesión.
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # El nombre lo escribe el usuario en la consola: se conserva solo el
    # basename y se sustituyen los caracteres no admitidos en un nombre de
    # archivo, para que /guardar no pueda escribir fuera de este directorio
    # (path traversal con "../../" o con un separador de ruta).
    base = Path(str(nombre).strip()).name
    base = "".join(c if (c.isalnum() or c in "._-") else "_"
                   for c in base).strip("._") or "sesion"
    # Tope de longitud: Windows limita a 255 caracteres cada componente de la
    # ruta, así que un nombre largo pasa el saneado pero write_text revienta con
    # OSError. El try de main() solo cubre console.input, de modo que el fallo
    # mataba la sesión con traceback en vez de avisar al usuario.
    base = base[:80]
    ruta = dir_salida / f"{base}_{ts}.txt"
    # Guardia final: el destino resuelto debe seguir dentro del directorio de
    # sesiones (no basta con sanear, se verifica).
    if ruta.resolve().parent != dir_salida.resolve():
        base = "sesion"
        ruta = dir_salida / f"{base}_{ts}.txt"
    lineas = [f"Sesión de atención TecnoApoyo - {ts}\n", "=" * 60]
    for turno in engine.historial:
        lineas.append(f"P: {turno['pregunta']}")
        if turno["abstencion"]:
            lineas.append(f"A: [ABSTENCIÓN] {turno['respuesta']}")
        else:
            lineas.append(f"A: {turno['respuesta']}")
        lineas.append("-" * 40)
    ruta.write_text("\n".join(lineas), encoding="utf-8")
    console.print(f"[green]Sesión guardada en:[/green] {ruta}")


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
                "- `/guardar [nombre]`: exporta la sesión a data/output/sesiones/\n"
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
        if ultima.lower().startswith("/guardar"):
            nombre = ultima.split(maxsplit=1)[1] if " " in ultima else "sesion"
            guardar_sesion(nombre)
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