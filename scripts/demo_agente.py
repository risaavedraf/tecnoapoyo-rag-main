"""Demostración de resolución del agente de TecnoApoyo (RA2 / IL2.1).

Esta demo no *responde* una consulta: **resuelve** una petición de recepción
ejecutando herramientas de escritura reales (`crear_caso`, `agregar_nota`,
`escalar_a_humano`) y luego **lee el repositorio de vuelta** para probar que el
efecto persistió. El código de salida es la verificación: 0 solo si el caso de
la clienta quedó en estado `escalado` y con al menos una nota; 1 en cualquier
otro caso.

Aislamiento de estado (importante):
  - Todo el estado mutable vive en un `tempfile.TemporaryDirectory` por
    corrida: la base de casos, la base de memoria y el directorio de trazas se
    pasan explícitos. Nunca se toca `data/output/casos.db`,
    `data/output/memoria.db` ni `data/output/trazas/`.
  - El índice FAISS se usa en modo lectura (el `RAGEngine` lo carga y no lo
    reescribe).

La aprobación humana está automatizada **solo para esta demo**: el confirmador
imprime lo mismo que vería una persona en la consola real y aprueba en
consecuencia, dejando constancia de que la aprobación es automática.

Uso:
    python scripts/demo_agente.py
    python scripts/demo_agente.py --modo planificado
    python scripts/demo_agente.py --pregunta "Registra un caso para ..."
    python scripts/demo_agente.py --limite-pasos 10

Advertencia: una corrida real consume tokens del proveedor del LLM.
"""
import argparse
import contextlib
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# El script se ejecuta desde la raíz (`python scripts/demo_agente.py`), pero
# Python pone en sys.path el directorio del script, no el del proyecto. Sin esta
# línea, `import src...` falla al arrancar aunque los tests pasen.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Reconexión UTF-8, mismo bloque que src/app.py y scripts/comparar_modos.py: en
# Windows la consola usa la codepage del sistema y las tildes se imprimen rotas.
for flujo in (sys.stdin, sys.stdout, sys.stderr):
    reconectar = getattr(flujo, "reconfigure", None)
    if reconectar is None:
        continue
    with contextlib.suppress(ValueError):  # flujos no reconectables o no tty
        reconectar(encoding="utf-8")

from src import config  # noqa: E402 - después del ajuste de sys.path

# Pregunta guionada por defecto: exige tres escrituras encadenadas (crear el
# caso, agregar la nota sobre el cargador y escalar a nivel 2). No es una
# pregunta documental: sin las tres herramientas no hay resolución.
PREGUNTA_POR_DEFECTO = (
    "Registra un caso para la clienta Camila Rojas por un notebook que no "
    "enciende, agrégale una nota indicando que el equipo no enciende ni con "
    "el cargador conectado, y escálalo a soporte de nivel 2."
)

# Condiciones que la demo exige para considerar la petición resuelta.
CLIENTE_ESPERADO = "Camila Rojas"
ESTADO_ESPERADO = "escalado"

# Aviso de costo: se imprime antes de construir el agente, porque en cuanto la
# corrida arranca ya hay consumo facturable.
AVISO_TOKENS = (
    "[demo] Aviso: esta corrida consume tokens del proveedor del LLM "
    "(una consulta real al modelo)."
)

PREFIJO = "[demo]"


@dataclass
class PiezasDemo:
    """Dependencias armadas de la demo, para poder inyectarlas en los tests."""

    agente: Any
    repositorio: Any
    memoria: Any
    motor: Any
    ruta_casos: Path
    ruta_memoria: Path
    ruta_trazas: Path


def confirmador_demo(salida=None):
    """Confirmador que imprime la propuesta de escritura y aprueba.

    La firma es la que exigen las herramientas: `(accion, resumen) -> bool`.
    En la consola real esa respuesta la da una persona; aquí se aprueba
    siempre, y el texto impreso lo dice para que nadie confunda la demo con una
    autorización real.
    """

    def confirmar(accion, resumen):
        flujo = salida if salida is not None else sys.stdout
        detalle = ", ".join(f"{clave}={valor}" for clave, valor in resumen.items())
        print(
            f"{PREFIJO} ¿Autorizas {accion}? {detalle} -> sí "
            "(aprobación automática solo para esta demo; en la consola real "
            "responde una persona)",
            file=flujo,
        )
        return True

    return confirmar


def construir_agente(
    ruta_casos,
    ruta_memoria,
    ruta_trazas,
    confirmador,
    modo="reactivo",
    max_pasos=None,
    motor=None,
    llm=None,
):
    """Arma el motor, el repositorio, la memoria y el agente con rutas dadas.

    Las rutas son parámetros obligatorios: la demo nunca debe caer en los
    defaults de `config`, que apuntan a `data/output/`. Los imports de los
    módulos pesados (LangChain, FAISS) viven aquí dentro para que el módulo se
    pueda importar —y testear— sin red ni índice.

    `motor` y `llm` son inyectables: los tests pasan dobles y construyen el
    agente real sin tocar el proveedor. Si no se inyecta motor se construye el
    `RAGEngine` real, que solo lee el índice FAISS.
    """
    from src.agente import AgenteTecnoApoyo
    from src.casos import RepositorioCasos
    from src.herramientas import crear_herramientas
    from src.memoria import Memoria, MemoriaLargoPlazo, crear_resumidor

    if motor is None:
        from src.rag import RAGEngine

        motor = RAGEngine()

    repositorio = RepositorioCasos(ruta=ruta_casos)
    herramientas = crear_herramientas(repositorio, motor, confirmador)
    # El resumidor de la estrategia 'resumen' usa el modelo del motor y cae al
    # respaldo local si el proveedor falla: mismo cableado que src/app.py.
    memoria = Memoria(
        largo=MemoriaLargoPlazo(ruta=ruta_memoria),
        resumidor=crear_resumidor(
            config.MEMORIA_RESUMIDOR, getattr(motor, "llm", None)
        ),
    )
    agente = AgenteTecnoApoyo(
        herramientas=herramientas,
        llm=llm,
        memoria=memoria,
        modo=modo,
        max_pasos=max_pasos,
    )
    return PiezasDemo(
        agente=agente,
        repositorio=repositorio,
        memoria=memoria,
        motor=motor,
        ruta_casos=Path(ruta_casos),
        ruta_memoria=Path(ruta_memoria),
        ruta_trazas=Path(ruta_trazas),
    )


def verificar_resolucion(repositorio, cliente: str = CLIENTE_ESPERADO,
                         estado: str = ESTADO_ESPERADO) -> dict | None:
    """Devuelve el caso que prueba la resolución, o None si no existe.

    La verificación se hace **contra el repositorio**, no contra lo que dijo el
    modelo: que el agente anuncie "listo" no es evidencia de nada. Se exige un
    caso del cliente esperado, en el estado esperado y con al menos una nota.
    """
    esperado = (cliente or "").strip().casefold()
    estado_esperado = (estado or "").strip().lower()
    for caso in reversed(repositorio.listar_casos(limite=100)):
        if str(caso.get("estado") or "").strip().lower() != estado_esperado:
            continue
        if esperado not in str(caso.get("cliente") or "").casefold():
            continue
        if not caso.get("notas"):
            continue
        return caso
    return None


def _imprimir_pasos(respuesta):
    """Resumen de pasos: herramienta, ok y error, en una línea por paso."""
    for paso in getattr(respuesta, "pasos", []) or []:
        estado = "sí" if getattr(paso, "ok", False) else "no"
        linea = f"{PREFIJO} Paso {paso.numero}: {paso.herramienta} | ok={estado}"
        error = getattr(paso, "error", None)
        if error:
            linea += f" | error={error}"
        print(linea)


def _imprimir_caso(caso):
    """Lectura de vuelta: id, cliente, estado y notas persistidas."""
    print(f"{PREFIJO} Caso persistido: {caso['id']} | cliente: {caso['cliente']} "
          f"| estado: {caso['estado']}")
    notas = caso.get("notas") or []
    print(f"{PREFIJO} Notas ({len(notas)}):")
    for nota in notas:
        print(f"  - {nota}")


def _escribir_traza(respuesta, ruta_trazas, modo, memoria):
    """Persiste la traza en el directorio aislado de la corrida.

    `escribir_traza` ya falla en silencio (una traza es accesoria); aquí solo se
    elige el directorio de la demo, nunca `config.RUTA_TRAZAS`.
    """
    from src.observabilidad import escribir_traza, traza_de_turno

    estrategia = getattr(getattr(memoria, "corto", None), "estrategia", "")
    try:
        traza = traza_de_turno(respuesta, modo=modo, estrategia_memoria=estrategia)
        escribir_traza(traza, ruta=Path(ruta_trazas))
    except Exception:  # noqa: BLE001 - perder una traza no invalida la demo
        print(f"{PREFIJO} Aviso: no se pudo escribir la traza de la corrida.")


def _parser():
    """Argumentos de la demo. El modo se valida contra la allowlist de config."""
    parser = argparse.ArgumentParser(
        description="Demuestra que el agente resuelve una petición de soporte "
                    "de punta a punta, con estado aislado en un directorio "
                    "temporal.",
    )
    parser.add_argument(
        "--modo",
        choices=config.MODOS_AGENTE,
        default=config.AGENTE_MODO,
        help="Modo de resolución del agente (por defecto: AGENTE_MODO).",
    )
    parser.add_argument(
        "--pregunta",
        default=PREGUNTA_POR_DEFECTO,
        help="Consulta a resolver (por defecto: la petición guionada de "
             "Camila Rojas).",
    )
    parser.add_argument(
        "--limite-pasos",
        type=int,
        default=None,
        dest="limite_pasos",
        help="Tope de pasos del turno. Sin valor, usa AGENTE_MAX_PASOS.",
    )
    return parser


def main(argv=None, constructor=None, confirmador=None):
    """Corre la demo y devuelve el código de salida de la verificación.

    `constructor` y `confirmador` son inyectables para los tests offline: se
    puede pasar un agente falso o un constructor que arme el agente real con un
    modelo guionado, sin red.
    """
    args = _parser().parse_args(argv)
    print(AVISO_TOKENS)
    print(f"{PREFIJO} Modo: {args.modo} | pregunta: {args.pregunta}")

    armar = constructor if constructor is not None else construir_agente
    aprobar = confirmador if confirmador is not None else confirmador_demo()

    with tempfile.TemporaryDirectory(prefix="demo_agente_") as temporal:
        base = Path(temporal)
        rutas = {
            "ruta_casos": base / "casos.db",
            "ruta_memoria": base / "memoria.db",
            "ruta_trazas": base / "trazas",
        }
        print(f"{PREFIJO} Estado aislado en: {base}")

        piezas = armar(
            **rutas,
            confirmador=aprobar,
            modo=args.modo,
            max_pasos=args.limite_pasos,
        )
        repositorio = piezas.repositorio
        casos_antes = repositorio.contar()
        print(f"{PREFIJO} Casos antes: {casos_antes}")

        respuesta = piezas.agente.consultar(args.pregunta)

        _imprimir_pasos(respuesta)
        print(f"{PREFIJO} Respuesta final: {respuesta.respuesta or '(sin respuesta)'}")
        if respuesta.error:
            print(f"{PREFIJO} Error del agente: {respuesta.error}")
        print(f"{PREFIJO} Modo: {args.modo} | pasos: {len(respuesta.pasos)} | "
              f"re-planes: {respuesta.replanes} | "
              f"tokens: {respuesta.tokens_total} | "
              f"latencia: {respuesta.latencia_seg}s")

        _escribir_traza(respuesta, rutas["ruta_trazas"], args.modo, piezas.memoria)

        casos_despues = repositorio.contar()
        print(f"{PREFIJO} Casos después: {casos_despues}")

        # La prueba no es el texto del modelo: es la fila que quedó en SQLite.
        caso = verificar_resolucion(repositorio)
        if caso is None:
            print(
                f"{PREFIJO} ERROR: No se pudo verificar la resolución. Se "
                f"esperaba un caso de '{CLIENTE_ESPERADO}' con estado "
                f"'{ESTADO_ESPERADO}' y al menos una nota en el repositorio "
                f"de la corrida ({rutas['ruta_casos']})."
            )
            return 1

        _imprimir_caso(caso)
        print(
            f"{PREFIJO} RESOLUCIÓN VERIFICADA: el caso {caso['id']} pertenece a "
            f"'{caso['cliente']}', está en estado '{caso['estado']}' y tiene "
            f"{len(caso.get('notas') or [])} nota(s)."
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
