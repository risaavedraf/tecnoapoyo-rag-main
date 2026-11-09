"""Comparación de modos del agente: reactivo vs. planificado (RA2 / IL2.3).

Corre las mismas consultas por ambos modos y deja el resultado en
`evaluation/results/`. Es la evidencia de IL2.3: no basta con tener un
planificador, hay que poder compararlo contra el bucle reactivo.

Requiere red y el índice FAISS construido; no se ejecuta en los tests. Corre
siempre los dos modos, con independencia del default de la sesión
(`AGENTE_MODO=planificado`, el valor de `src/config.py`). Uso:

    python scripts/comparar_modos.py
    python scripts/comparar_modos.py "¿cuánto demora el diagnóstico?"
"""
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import arranque, config  # noqa: E402
from src.agente import AgenteTecnoApoyo  # noqa: E402
from src.casos import RepositorioCasos  # noqa: E402
from src.herramientas import crear_herramientas  # noqa: E402
from src.memoria import Memoria  # noqa: E402
from src.rag import RAGEngine  # noqa: E402

CONSULTAS_POR_DEFECTO = [
    "¿Cuánto demora el diagnóstico de un notebook?",
    "¿Qué casos tengo abiertos?",
    "¿Cuál es el plazo de una reparación con repuesto?",
    "¿Reparan tablets?",
    "¿En qué estado está el caso CASO-0004?",
]


def confirmador_no(accion, resumen):
    """La comparación no escribe: mide selección y orquestación, no el s/n."""
    return False


def medir(consulta: str, modo: str, motor: RAGEngine,
          repositorio: RepositorioCasos) -> dict:
    herramientas = crear_herramientas(repositorio, motor, confirmador_no)
    agente = AgenteTecnoApoyo(
        herramientas=herramientas,
        memoria=Memoria(),
        modo=modo,
    )
    respuesta = agente.consultar(consulta)
    return {
        "consulta": consulta,
        "modo": modo,
        "respuesta": respuesta.respuesta,
        "abstencion": respuesta.abstencion,
        "error": respuesta.error,
        "herramientas": [paso.herramienta for paso in respuesta.pasos],
        "pasos": len(respuesta.pasos),
        "replanes": respuesta.replanes,
        "tokens": respuesta.tokens_total,
        "latencia_seg": respuesta.latencia_seg,
        "plan": respuesta.plan.model_dump() if respuesta.plan is not None else None,
    }


def main() -> int:
    # Flujos en UTF-8 y clave antes de construir el motor: sin ella el cliente
    # del proveedor falla más tarde y con un error que no dice qué hacer.
    arranque.reconectar_flujos()
    arranque.exigir_clave()
    consultas = sys.argv[1:] or CONSULTAS_POR_DEFECTO
    motor = RAGEngine()
    repositorio = RepositorioCasos()
    filas = []
    for consulta in consultas:
        for modo in config.MODOS_AGENTE:
            print(f"[{modo}] {consulta}")
            filas.append(medir(consulta, modo, motor, repositorio))

    marca = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    config.DIR_EVAL_RESULTADOS.mkdir(parents=True, exist_ok=True)
    destino = config.DIR_EVAL_RESULTADOS / f"comparacion_modos_{marca}.json"
    destino.write_text(
        json.dumps({"generado_en": marca, "filas": filas}, ensure_ascii=False,
                   indent=2),
        encoding="utf-8",
    )
    print(f"Resultado en {destino}")

    print(f"\n{'consulta':48} {'modo':12} {'pasos':>5} {'repl':>4} "
          f"{'tok':>6} {'lat':>6} abst")
    for fila in filas:
        print(f"{fila['consulta'][:48]:48} {fila['modo']:12} "
              f"{fila['pasos']:5} {fila['replanes']:4} {fila['tokens']:6} "
              f"{fila['latencia_seg']:6} {'sí' if fila['abstencion'] else 'no'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
