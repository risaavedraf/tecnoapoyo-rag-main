"""Evaluación del agente de TecnoApoyo (RA2 / IL2.1-IL2.4).

Mide el comportamiento del **agente**, no la recuperación del motor RAG: el
motor ya tiene su propia evaluación (`scripts/evaluate.py`,
`evaluation/dataset_evaluacion.json`). Aquí el contrato es
`evaluation/dataset_agente.json` y lo que se puntúa es la traza:

  - selección de herramienta (`esperado.herramienta`);
  - argumentos clave que efectivamente viajaron a esa herramienta;
  - abstención (precisión y recall, no solo "acertó");
  - uso del contexto conversacional en las tareas de memoria;
  - confirmación de escritura: se deniega y se comprueba que el conteo de
    casos no cambió;
  - validez del plan en las tareas de planificación.

Dos reglas sostienen este script:

  1. La puntuación es código, no criterio del modelo. Un juez LLM podría
     cambiar de opinión entre corridas; estas métricas se recalculan igual
     sobre la misma traza, y por eso son comparables entre modos y versiones.
  2. El camino offline no construye nada. `--verificar-dataset` solo valida el
     esquema y lista las tareas: no crea `RAGEngine`, ni repositorio, ni
     agente, ni toca la red. Es el camino que corre en los tests.

Uso:
    python scripts/evaluar_agente.py --verificar-dataset
    python scripts/evaluar_agente.py
    python scripts/evaluar_agente.py --modo planificado --limite 5
    python scripts/evaluar_agente.py --salida evaluation/results/agente.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config

RUTA_DATASET = config.RAIZ / "evaluation" / "dataset_agente.json"

# Contrato del dataset. El orden importa: es el que usan los informes.
TIPOS_VALIDOS = (
    "seleccion_herramienta",
    "abstencion",
    "memoria",
    "escritura",
    "planificacion",
)
CAMPOS_ESPERADO = (
    "herramienta",
    "argumentos_clave",
    "abstencion",
    "requiere_confirmacion",
    "min_pasos_plan",
)

# Herramientas que escriben en el repositorio de casos. Se declaran aquí, y no
# se importan de `src.herramientas`, para que la verificación del dataset y la
# puntuación no arrastren LangChain: el camino offline debe poder correr sin
# dependencias de red ni de proveedor.
HERRAMIENTAS_ESCRITURA = (
    "crear_caso",
    "agregar_nota",
    "actualizar_estado",
    "escalar_a_humano",
)

# Tarifa usada para el costo estimado. Es un global de módulo —no un parámetro
# de `agregar_metricas`— porque la firma pública de esa función es de un solo
# argumento; `main` lo fija con `--tarifa-usd-por-millon` antes de agregar.
# Unidades: USD por millón de tokens totales (entrada + salida, que es lo único
# que `RespuestaAgente` expone hoy).
TARIFA_USD_POR_MILLON = 0.0

# Palabras que no identifican contexto: aparecen en cualquier pregunta y
# marcarían como "usó la memoria" a una respuesta que no la usó.
_PALABRAS_IGNORADAS = frozenset({
    "para", "como", "este", "esta", "esto", "esos", "esas", "pero", "solo",
    "todo", "toda", "todos", "todas", "cuando", "donde", "porque", "sobre",
    "desde", "hasta", "entre", "tengo", "tiene", "puedo", "puede", "quiero",
    "anotado", "usuario", "asistente", "cliente",
})

_RE_ID = re.compile(r"\b[A-Za-z]{2,}-\d+\b")
_RE_PALABRA = re.compile(r"[a-z]{4,}")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def cargar_dataset(ruta: Path | None = None) -> dict:
    """Lee el dataset de tareas. No valida: eso es `validar_dataset`."""
    destino = Path(ruta) if ruta is not None else RUTA_DATASET
    return json.loads(destino.read_text(encoding="utf-8"))


def _es_entero(valor: Any) -> bool:
    """True para enteros reales: `bool` es subclase de `int` en Python."""
    return isinstance(valor, int) and not isinstance(valor, bool)


def validar_dataset(datos: dict) -> list[str]:
    """Devuelve los errores de esquema del dataset; lista vacía = válido.

    Valida forma, no semántica de negocio: no comprueba que el nombre de una
    herramienta exista en `src.herramientas` porque ese catálogo vive en un
    módulo que no queremos importar en el camino offline, y una lista
    duplicada aquí quedaría desactualizada sin que nadie lo note.
    """
    if not isinstance(datos, dict):
        return ["El dataset debe ser un objeto JSON con la clave 'tareas'."]
    tareas = datos.get("tareas")
    if not isinstance(tareas, list) or not tareas:
        return ["El dataset debe traer una lista 'tareas' con al menos una tarea."]

    errores: list[str] = []
    vistos: dict[str, int] = {}

    for indice, tarea in enumerate(tareas, 1):
        if not isinstance(tarea, dict):
            errores.append(f"Tarea #{indice}: debe ser un objeto JSON.")
            continue

        tarea_id = str(tarea.get("id") or "").strip()
        etiqueta = tarea_id or f"#{indice}"
        if not tarea_id:
            errores.append(f"Tarea #{indice}: falta 'id'.")
        elif tarea_id in vistos:
            errores.append(
                f"Id duplicado: '{tarea_id}' en las tareas "
                f"{vistos[tarea_id]} y {indice}."
            )
        else:
            vistos[tarea_id] = indice

        tipo = tarea.get("tipo")
        if tipo not in TIPOS_VALIDOS:
            errores.append(
                f"Tarea {etiqueta}: tipo desconocido {tipo!r}. "
                f"Validos: {', '.join(TIPOS_VALIDOS)}."
            )

        pregunta = tarea.get("pregunta")
        if not isinstance(pregunta, str) or not pregunta.strip():
            errores.append(f"Tarea {etiqueta}: falta 'pregunta' (texto no vacio).")

        turnos = tarea.get("turnos_previos")
        if not isinstance(turnos, list):
            errores.append(f"Tarea {etiqueta}: 'turnos_previos' debe ser una lista.")
        else:
            for posicion, turno in enumerate(turnos, 1):
                if not isinstance(turno, dict):
                    errores.append(
                        f"Tarea {etiqueta}: turno {posicion} debe ser un objeto."
                    )
                    continue
                rol = str(turno.get("rol") or "").strip()
                contenido = turno.get("contenido")
                if not rol:
                    errores.append(
                        f"Tarea {etiqueta}: turno {posicion} sin 'rol'."
                    )
                if not isinstance(contenido, str) or not contenido.strip():
                    errores.append(
                        f"Tarea {etiqueta}: turno {posicion} sin 'contenido'."
                    )
            if tipo == "memoria" and not turnos:
                errores.append(
                    f"Tarea {etiqueta}: una tarea de memoria necesita "
                    "'turnos_previos' con el contexto que debe recordar."
                )

        esperado = tarea.get("esperado")
        if not isinstance(esperado, dict):
            errores.append(f"Tarea {etiqueta}: falta 'esperado'.")
            continue

        for campo in CAMPOS_ESPERADO:
            if campo not in esperado:
                errores.append(
                    f"Tarea {etiqueta}: 'esperado' no trae '{campo}'."
                )

        herramienta = esperado.get("herramienta")
        if herramienta is not None and (
            not isinstance(herramienta, str) or not herramienta.strip()
        ):
            errores.append(
                f"Tarea {etiqueta}: 'esperado.herramienta' debe ser un nombre "
                "o null."
            )

        argumentos = esperado.get("argumentos_clave")
        if not isinstance(argumentos, dict):
            errores.append(
                f"Tarea {etiqueta}: 'esperado.argumentos_clave' debe ser un "
                "objeto (vacio si no se esperan claves)."
            )

        for booleano in ("abstencion", "requiere_confirmacion"):
            if booleano in esperado and not isinstance(esperado[booleano], bool):
                errores.append(
                    f"Tarea {etiqueta}: 'esperado.{booleano}' debe ser booleano."
                )

        if "min_pasos_plan" in esperado:
            minimo = esperado["min_pasos_plan"]
            if minimo is not None and (not _es_entero(minimo) or minimo < 1):
                errores.append(
                    f"Tarea {etiqueta}: 'esperado.min_pasos_plan' debe ser un "
                    "entero >= 1 o null."
                )

    return errores


# ---------------------------------------------------------------------------
# Utilidades de comparación
# ---------------------------------------------------------------------------
def _sin_acentos(texto: str) -> str:
    descompuesto = unicodedata.normalize("NFD", texto)
    return "".join(c for c in descompuesto if unicodedata.category(c) != "Mn")


def _normalizar(valor: Any) -> str:
    """Texto comparable: sin acentos, sin mayúsculas y sin espacios sobrantes."""
    return _sin_acentos(str(valor)).casefold().strip()


def _leer(obj: Any, nombre: str, defecto: Any = None) -> Any:
    """Lee un campo de un objeto o de un dict, indistintamente.

    La traza real son dataclasses (`RespuestaAgente`, `PasoAgente`), pero un
    resultado releído desde JSON son dicts: la puntuación debe aceptar ambos.
    """
    if isinstance(obj, dict):
        return obj.get(nombre, defecto)
    return getattr(obj, nombre, defecto)


def _pasos_de(respuesta: Any) -> list[Any]:
    pasos = _leer(respuesta, "pasos", None) or []
    return list(pasos) if isinstance(pasos, (list, tuple)) else []


def _argumentos_de(paso: Any) -> dict:
    argumentos = _leer(paso, "argumentos", None)
    return argumentos if isinstance(argumentos, dict) else {}


def _argumento_coincide(clave: str, esperado: Any, obtenido: Any) -> bool:
    """Compara un argumento esperado contra el que viajó en la traza.

    Reglas del contrato: `id_caso` se compara en mayúsculas exactas (es un
    identificador, no prosa) y el resto por contenido parcial, sin acentos ni
    mayúsculas. La comparación parcial es deliberada — el modelo reformula la
    pregunta ("¿Cuánto demora el diagnóstico?" -> pregunta="demora del
    diagnóstico") y eso sigue siendo la misma consulta.
    """
    if obtenido is None:
        return False

    if clave.lower() == "id_caso":
        return str(obtenido).strip().upper() == str(esperado).strip().upper()

    if isinstance(esperado, bool) or isinstance(obtenido, bool):
        return bool(obtenido) == bool(esperado)

    if isinstance(esperado, (int, float)) and not isinstance(esperado, bool):
        try:
            return float(obtenido) == float(esperado)
        except (TypeError, ValueError):
            return _normalizar(obtenido) == _normalizar(esperado)

    esperado_normalizado = _normalizar(esperado)
    if not esperado_normalizado:
        return False
    return esperado_normalizado in _normalizar(obtenido)


# ---------------------------------------------------------------------------
# Memoria
# ---------------------------------------------------------------------------
def _tokens_de_contexto(turnos_previos: Any, pregunta: str) -> tuple[set[str], set[str]]:
    """Identificadores y palabras que solo pueden venir del contexto previo.

    Se miran los turnos del **usuario**: son los que llevan el dato recordable
    (un id de caso, un equipo). Los del asistente son en su mayoría eco
    ("Anotado: ...") y agregarían vocabulario genérico que cualquier respuesta
    cumpliría. Las palabras de la pregunta actual se descartan porque no
    prueban memoria: están a la vista del modelo sin ella.
    """
    if not isinstance(turnos_previos, list):
        return set(), set()

    usuario = [
        str(turno.get("contenido") or "")
        for turno in turnos_previos
        if isinstance(turno, dict)
        and str(turno.get("rol") or "").strip().lower() == "usuario"
    ]
    texto = " ".join(usuario)
    identificadores = {hallado.upper() for hallado in _RE_ID.findall(texto)}
    palabras = set(_RE_PALABRA.findall(_sin_acentos(texto).lower()))
    palabras -= set(_RE_PALABRA.findall(_sin_acentos(str(pregunta or "")).lower()))
    palabras -= _PALABRAS_IGNORADAS
    return identificadores, palabras


def _evidencia_de_contexto(respuesta: Any, pasos: list[Any]) -> str:
    """Texto donde buscar rastros del contexto: la respuesta y los argumentos."""
    partes = [str(_leer(respuesta, "respuesta", "") or "")]
    for paso in pasos:
        argumentos = _argumentos_de(paso)
        partes.extend(str(valor) for valor in argumentos.values())
    return " ".join(partes)


def _contexto_usado(tarea: dict, respuesta: Any, pasos: list[Any]) -> bool:
    identificadores, palabras = _tokens_de_contexto(
        tarea.get("turnos_previos"), tarea.get("pregunta", "")
    )
    if not identificadores and not palabras:
        # Sin nada distintivo en el contexto previo no hay nada que exigir.
        return True
    evidencia = _normalizar(_evidencia_de_contexto(respuesta, pasos))
    if any(identificador in evidencia.upper() for identificador in identificadores):
        return True
    return any(palabra in evidencia for palabra in palabras)


# ---------------------------------------------------------------------------
# Puntuación de una tarea
# ---------------------------------------------------------------------------
def puntuar_tarea(
    tarea: dict,
    respuesta: Any,
    *,
    escrituras_antes: int | None = None,
    escrituras_despues: int | None = None,
) -> dict:
    """Puntúa una traza contra su tarea. Puro: no ejecuta ni llama al modelo.

    `escrituras_antes`/`escrituras_despues` son el conteo de casos del
    repositorio alrededor de la consulta. Son la prueba de que la escritura
    denegada no ocurrió: una tool de escritura puede responder `ok=False` y
    aun así haber escrito, y este conteo es el que lo delata.
    """
    esperado = (tarea.get("esperado") if isinstance(tarea, dict) else None) or {}
    tipo = tarea.get("tipo") if isinstance(tarea, dict) else None
    herramienta_esperada = esperado.get("herramienta")
    claves = esperado.get("argumentos_clave") or {}
    if not isinstance(claves, dict):
        claves = {}
    abstencion_esperada = bool(esperado.get("abstencion", False))
    requiere_confirmacion = bool(esperado.get("requiere_confirmacion", False))
    minimo_plan = esperado.get("min_pasos_plan")
    if not _es_entero(minimo_plan):
        minimo_plan = None

    pasos = _pasos_de(respuesta)
    nombres = [str(_leer(paso, "herramienta", "") or "") for paso in pasos]
    abstencion_obtenida = bool(_leer(respuesta, "abstencion", False))

    # -- selección de herramienta -----------------------------------------
    if not herramienta_esperada:
        herramienta_ok = True
    else:
        herramienta_ok = herramienta_esperada in nombres

    # -- argumentos --------------------------------------------------------
    candidatos = (
        [
            paso
            for paso, nombre in zip(pasos, nombres, strict=True)
            if nombre == herramienta_esperada
        ]
        if herramienta_esperada
        else list(pasos)
    )
    if not claves:
        argumentos_ok = True
    else:
        argumentos_ok = any(
            all(
                _argumento_coincide(clave, valor, _argumentos_de(paso).get(clave))
                for clave, valor in claves.items()
            )
            for paso in candidatos
        )

    # -- abstención --------------------------------------------------------
    abstencion_ok = abstencion_obtenida == abstencion_esperada

    # -- memoria -----------------------------------------------------------
    if tipo == "memoria":
        memoria_ok = (not abstencion_obtenida) and _contexto_usado(
            tarea, respuesta, pasos
        )
    else:
        memoria_ok = True

    # -- confirmación de escritura ----------------------------------------
    if requiere_confirmacion:
        pasos_escritura = [
            paso
            for paso, nombre in zip(pasos, nombres, strict=True)
            if nombre in HERRAMIENTAS_ESCRITURA
        ]
        escritura_denegada = any(
            not bool(_leer(paso, "ok", False)) for paso in pasos_escritura
        )
        # Sin los dos conteos no se puede afirmar que no hubo escritura. Se
        # falla cerrado: un cambio no comprobado no es un cambio descartado.
        conteo_estable = (
            escrituras_antes is not None
            and escrituras_despues is not None
            and escrituras_antes == escrituras_despues
        )
        confirmacion_ok = escritura_denegada and conteo_estable
    else:
        confirmacion_ok = True

    # -- plan --------------------------------------------------------------
    plan = _leer(respuesta, "plan", None)
    if minimo_plan is None:
        plan_ok = True
    else:
        plan_ok = plan is not None and len(pasos) >= minimo_plan

    puntaje = {
        "id": tarea.get("id") if isinstance(tarea, dict) else None,
        "tipo": tipo,
        "pregunta": tarea.get("pregunta") if isinstance(tarea, dict) else None,
        "herramienta_esperada": herramienta_esperada,
        "herramienta_ok": bool(herramienta_ok),
        "argumentos_ok": bool(argumentos_ok),
        "abstencion_esperada": abstencion_esperada,
        "abstencion_obtenida": abstencion_obtenida,
        "abstencion_ok": bool(abstencion_ok),
        "memoria_ok": bool(memoria_ok),
        "requiere_confirmacion": requiere_confirmacion,
        "confirmacion_ok": bool(confirmacion_ok),
        "min_pasos_plan": minimo_plan,
        "plan_ok": bool(plan_ok),
        "pasos": len(pasos),
        "herramientas": nombres,
        "replanes": int(_leer(respuesta, "replanes", 0) or 0),
        "tokens": int(_leer(respuesta, "tokens_total", 0) or 0),
        "latencia_seg": round(float(_leer(respuesta, "latencia_seg", 0.0) or 0.0), 3),
        "error": _leer(respuesta, "error", None),
        "escrituras_antes": escrituras_antes,
        "escrituras_despues": escrituras_despues,
    }
    puntaje["motivos"] = _motivos_de_fallo(puntaje)
    return puntaje


_METRICAS_LEGIBLES = {
    "herramienta_ok": "herramienta",
    "argumentos_ok": "argumentos",
    "abstencion_ok": "abstencion",
    "memoria_ok": "memoria",
    "confirmacion_ok": "confirmacion",
    "plan_ok": "plan",
}


def _motivos_de_fallo(puntaje: dict) -> list[str]:
    """Nombres de las métricas que fallaron, para el informe legible."""
    return [
        etiqueta
        for clave, etiqueta in _METRICAS_LEGIBLES.items()
        if not puntaje.get(clave, True)
    ]


# ---------------------------------------------------------------------------
# Agregación
# ---------------------------------------------------------------------------
def _tasa(valores: list[bool]) -> float | None:
    if not valores:
        return None
    return round(sum(1 for valor in valores if valor) / len(valores), 4)


def _abstencion_acertada(puntaje: dict) -> bool:
    """Acierto de abstención; se deriva si el artefacto no trae la bandera."""
    if "abstencion_ok" in puntaje:
        return bool(puntaje["abstencion_ok"])
    return bool(puntaje.get("abstencion_esperada")) == bool(
        puntaje.get("abstencion_obtenida")
    )


def _metricas_de_grupo(puntajes: list[dict], tarifa: float) -> dict:
    """Tasas de un grupo (todas las tareas, o las de un tipo).

    `tasa_memoria` solo considera tareas de tipo 'memoria': en el resto la
    métrica es trivialmente cierta, y promediarla con las demás inflaría el
    indicador hasta hacerlo inútil.
    """
    total = len(puntajes)

    def columna(nombre: str) -> list[bool]:
        return [bool(puntaje.get(nombre)) for puntaje in puntajes]

    # TP/FP/FN de abstención, con la abstención como clase positiva:
    #   TP = se abstuvo y debía abstenerse;
    #   FP = se abstuvo y no debía (falsa abstención);
    #   FN = no se abstuvo y debía (abstención omitida).
    tp = sum(
        1 for p in puntajes
        if p.get("abstencion_esperada") and p.get("abstencion_obtenida")
    )
    fp = sum(
        1 for p in puntajes
        if not p.get("abstencion_esperada") and p.get("abstencion_obtenida")
    )
    fn = sum(
        1 for p in puntajes
        if p.get("abstencion_esperada") and not p.get("abstencion_obtenida")
    )

    tokens = sum(int(puntaje.get("tokens") or 0) for puntaje in puntajes)
    latencia = sum(float(puntaje.get("latencia_seg") or 0.0) for puntaje in puntajes)
    replanes = [int(puntaje.get("replanes") or 0) for puntaje in puntajes]

    return {
        "n_tareas": total,
        "tasa_seleccion": _tasa(columna("herramienta_ok")),
        "tasa_argumentos": _tasa(columna("argumentos_ok")),
        "tasa_abstencion": _tasa([_abstencion_acertada(p) for p in puntajes]),
        "abstencion_tp": tp,
        "abstencion_fp": fp,
        "abstencion_fn": fn,
        "abstencion_precision": round(tp / (tp + fp), 4) if (tp + fp) else None,
        "abstencion_recall": round(tp / (tp + fn), 4) if (tp + fn) else None,
        "tasa_memoria": _tasa(
            [bool(p.get("memoria_ok")) for p in puntajes if p.get("tipo") == "memoria"]
        ),
        "tasa_confirmacion": _tasa(columna("confirmacion_ok")),
        "tasa_plan_valido": _tasa(columna("plan_ok")),
        "promedio_replanes": round(sum(replanes) / total, 2) if total else 0.0,
        "tokens_totales": tokens,
        "latencia_total_seg": round(latencia, 2),
        "costo_estimado_usd": round((tokens / 1_000_000) * tarifa, 6),
        "tarifa_usd_por_millon": tarifa,
    }


def agregar_metricas(puntajes: list[dict]) -> dict:
    """Agrega los puntajes por tipo de tarea y en global.

    La tarifa del costo estimado sale de `TARIFA_USD_POR_MILLON` (global del
    módulo, fijado por `main`): la firma pública es de un solo argumento.
    """
    tarifa = float(TARIFA_USD_POR_MILLON)
    tipos = list(TIPOS_VALIDOS)
    # Un tipo que no esté en la lista canónica igual se reporta: perder tareas
    # en la agregación es peor que mostrar una categoría inesperada.
    tipos += sorted({
        str(puntaje.get("tipo")) for puntaje in puntajes
    } - set(TIPOS_VALIDOS))

    por_tipo = {}
    for tipo in tipos:
        grupo = [p for p in puntajes if str(p.get("tipo")) == tipo]
        if grupo:
            por_tipo[tipo] = _metricas_de_grupo(grupo, tarifa)

    return {
        "global": _metricas_de_grupo(puntajes, tarifa),
        "por_tipo": por_tipo,
        "tarifa_usd_por_millon": tarifa,
    }


# ---------------------------------------------------------------------------
# Corrida real (red)
# ---------------------------------------------------------------------------
def confirmar_denegando(_accion: str, _resumen: dict) -> bool:
    """Confirmador que siempre deniega: la evaluación no escribe casos.

    Es la forma de medir confirmación sin efectos: el agente debe proponer la
    escritura, la tool debe responder `ok=False`, y el conteo de casos no debe
    moverse.
    """
    return False


def _preparar_memoria(memoria: Any, tarea: dict) -> None:
    """Carga los turnos previos de la tarea y descarta los de la anterior.

    Sin este reinicio, el contexto de una tarea de memoria se filtraría a la
    siguiente y la métrica de memoria mediría arrastre, no memoria.
    """
    corto = getattr(memoria, "corto", None)
    if corto is None:
        return
    corto.limpiar()
    for turno in tarea.get("turnos_previos") or []:
        if not isinstance(turno, dict):
            continue
        rol = str(turno.get("rol") or "").strip()
        contenido = str(turno.get("contenido") or "").strip()
        if rol and contenido:
            corto.agregar(rol, contenido)


def _respuesta_de_error(pregunta: str, exc: Exception, inicio: float) -> dict:
    """Traza mínima cuando `consultar` lanza: se puntúa como fallo, no se pierde."""
    return {
        "pregunta": pregunta,
        "respuesta": "",
        "abstencion": False,
        "pasos": [],
        "iteraciones": 0,
        "tokens_total": 0,
        "latencia_seg": round(time.time() - inicio, 2),
        "error": f"{type(exc).__name__}: {exc}",
        "replanes": 0,
        "plan": None,
    }


def _correr(datos: dict, modos: list[str], limite: int | None) -> dict:
    """Corre el dataset contra el agente real. Requiere red e índice FAISS.

    Las importaciones son locales: `--verificar-dataset` no debe cargar
    LangChain, FAISS ni el proveedor para validar un JSON.
    """
    from src.agente import AgenteTecnoApoyo
    from src.casos import RepositorioCasos
    from src.herramientas import crear_herramientas
    from src.memoria import Memoria, MemoriaLargoPlazo
    from src.rag import RAGEngine

    tareas = list(datos.get("tareas") or [])
    if limite is not None:
        tareas = tareas[:max(1, limite)]

    motor = RAGEngine()
    repositorio = RepositorioCasos()

    corridas = []
    with tempfile.TemporaryDirectory(prefix="evaluar_agente_") as dir_temp:
        # Memoria aislada en un directorio temporal: la corrida no debe dejar
        # hechos ni turnos en la memoria real del usuario.
        memoria = Memoria(
            largo=MemoriaLargoPlazo(ruta=Path(dir_temp) / "memoria.db")
        )
        for modo in modos:
            herramientas = crear_herramientas(
                repositorio, motor, confirmar_denegando
            )
            agente = AgenteTecnoApoyo(
                herramientas=herramientas, memoria=memoria, modo=modo
            )
            puntajes = []
            for posicion, tarea in enumerate(tareas, 1):
                pregunta = str(tarea.get("pregunta") or "")
                _preparar_memoria(memoria, tarea)
                antes = repositorio.contar()
                inicio = time.time()
                try:
                    respuesta = agente.consultar(pregunta)
                except Exception as exc:  # noqa: BLE001 - se puntúa como fallo
                    respuesta = _respuesta_de_error(pregunta, exc, inicio)
                despues = repositorio.contar()

                puntaje = puntuar_tarea(
                    tarea,
                    respuesta,
                    escrituras_antes=antes,
                    escrituras_despues=despues,
                )
                puntajes.append(puntaje)
                estado = "OK" if not puntaje["motivos"] else "FALLA"
                detalle = ",".join(puntaje["motivos"]) or "-"
                print(
                    f"[{modo}][{puntaje['id']}] {posicion:02d}/{len(tareas)} "
                    f"{estado:5s} | {detalle:30s} | {pregunta[:60]}"
                )

            corridas.append({
                "modo": modo,
                "puntajes": puntajes,
                "metricas": agregar_metricas(puntajes),
            })
    return {"corridas": corridas, "tareas_evaluadas": len(tareas)}


# ---------------------------------------------------------------------------
# Artefactos
# ---------------------------------------------------------------------------
def _reservar_par(ruta_json: Path) -> tuple[Path, Path]:
    """Devuelve un par (.json, .md) que no pisa ningún archivo existente.

    Se busca el primer sufijo libre para los DOS archivos a la vez: si solo se
    comprobara el .json, un .md homónimo previo se sobrescribiría en silencio.
    Solo se descarta la extensión .json; cualquier otro sufijo del nombre se
    conserva, para que una ruta como `agente.v1` no pierda el `.v1`.
    """
    tallo = (
        ruta_json.parent / ruta_json.stem
        if ruta_json.suffix.lower() == ".json"
        else ruta_json
    )
    for intento in range(1, 1000):
        nombre = tallo.name if intento == 1 else f"{tallo.name}_{intento}"
        candidato_json = tallo.parent / f"{nombre}.json"
        candidato_md = tallo.parent / f"{nombre}.md"
        if not candidato_json.exists() and not candidato_md.exists():
            return candidato_json, candidato_md
    raise RuntimeError(f"No se encontró un nombre libre para {ruta_json}.")


def _pct(valor: Any) -> str:
    return "n/d" if valor is None else f"{valor:.1%}"


def _markdown(resumen: dict) -> str:
    """Resumen legible: una tabla global por modo y el detalle de fallos."""
    lineas = [
        "# Evaluación del agente de TecnoApoyo",
        "",
        f"- Fecha: {resumen['fecha']}",
        f"- Modelo del agente: {resumen['modelo_agente']}",
        f"- Dataset: {resumen['dataset']} "
        f"({resumen['verificacion_dataset']['tareas']} tareas)",
        f"- Tarifa usada: US$ {resumen['tarifa_usd_por_millon']} por millón de tokens",
        "",
    ]
    for corrida in resumen["corridas"]:
        global_ = corrida["metricas"]["global"]
        lineas += [
            f"## Modo {corrida['modo']}",
            "",
            "| Métrica | Valor |",
            "| --- | --- |",
            f"| Tareas evaluadas | {global_['n_tareas']} |",
            f"| Tasa de selección de herramienta | {_pct(global_['tasa_seleccion'])} |",
            f"| Tasa de argumentos correctos | {_pct(global_['tasa_argumentos'])} |",
            f"| Tasa de abstención correcta | {_pct(global_['tasa_abstencion'])} |",
            f"| Precisión de abstención | {_pct(global_['abstencion_precision'])} |",
            f"| Recall de abstención | {_pct(global_['abstencion_recall'])} |",
            f"| Tasa de memoria usada | {_pct(global_['tasa_memoria'])} |",
            f"| Tasa de confirmación de escritura | {_pct(global_['tasa_confirmacion'])} |",
            f"| Tasa de plan válido | {_pct(global_['tasa_plan_valido'])} |",
            f"| Promedio de re-planes | {global_['promedio_replanes']} |",
            f"| Tokens totales | {global_['tokens_totales']} |",
            f"| Latencia total (s) | {global_['latencia_total_seg']} |",
            f"| Costo estimado (USD) | {global_['costo_estimado_usd']} |",
            "",
            "### Por tipo de tarea",
            "",
            "| Tipo | n | Selección | Argumentos | Abstención | Memoria | Confirmación | Plan |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for tipo, metricas in corrida["metricas"]["por_tipo"].items():
            lineas.append(
                f"| {tipo} | {metricas['n_tareas']} | "
                f"{_pct(metricas['tasa_seleccion'])} | "
                f"{_pct(metricas['tasa_argumentos'])} | "
                f"{_pct(metricas['tasa_abstencion'])} | "
                f"{_pct(metricas['tasa_memoria'])} | "
                f"{_pct(metricas['tasa_confirmacion'])} | "
                f"{_pct(metricas['tasa_plan_valido'])} |"
            )

        fallos = [p for p in corrida["puntajes"] if p.get("motivos")]
        lineas += ["", f"### Tareas con fallos ({len(fallos)})", ""]
        if not fallos:
            lineas.append("Sin fallos: todas las tareas cumplieron su contrato.")
        for puntaje in fallos:
            lineas.append(
                f"- {puntaje.get('id')} ({puntaje.get('tipo')}): "
                f"{', '.join(puntaje.get('motivos') or [])} — "
                f"herramientas={puntaje.get('herramientas')} "
                f"pasos={puntaje.get('pasos')} "
                f"error={puntaje.get('error')}"
            )
        lineas.append("")
    return "\n".join(lineas)


def _escribir(resumen: dict, ruta_json: Path) -> tuple[Path, Path]:
    """Escribe el JSON y el Markdown del resumen. Nunca sobrescribe."""
    ruta_json, ruta_md = _reservar_par(ruta_json)
    ruta_json.parent.mkdir(parents=True, exist_ok=True)
    ruta_json.write_text(
        json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    ruta_md.write_text(_markdown(resumen), encoding="utf-8")
    return ruta_json, ruta_md


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evalúa el agente de TecnoApoyo (herramientas, argumentos, "
            "abstención, memoria, confirmación y plan)."
        )
    )
    parser.add_argument(
        "--modo",
        choices=("reactivo", "planificado", "ambos"),
        default="ambos",
        help="Modo del agente a evaluar (default: ambos).",
    )
    parser.add_argument(
        "--limite",
        type=int,
        default=None,
        help="Evalúa solo las primeras N tareas del dataset.",
    )
    parser.add_argument(
        "--salida",
        type=Path,
        default=None,
        help=(
            "Ruta del JSON de resultados (el .md se escribe al lado). "
            "Default: evaluation/results/agente_<fecha>.json"
        ),
    )
    parser.add_argument(
        "--tarifa-usd-por-millon",
        type=float,
        default=0.0,
        help="USD por millón de tokens para el costo estimado (default: 0.0).",
    )
    parser.add_argument(
        "--verificar-dataset",
        action="store_true",
        help=(
            "Solo valida el esquema del dataset y lista las tareas. "
            "No crea el agente ni llama al modelo."
        ),
    )
    return parser


def _verificar_dataset() -> int:
    """Camino offline: valida y lista. No construye agente ni usa red."""
    try:
        datos = cargar_dataset()
    except (OSError, ValueError) as exc:
        print(f"[ERROR] No se pudo leer {RUTA_DATASET}: {exc}")
        return 1

    errores = validar_dataset(datos)
    if errores:
        print(f"[ERROR] Dataset inválido: {RUTA_DATASET}")
        for error in errores:
            print(f"  - {error}")
        return 1

    tareas = datos["tareas"]
    print(f"Dataset: {RUTA_DATASET}")
    print(f"Tareas: {len(tareas)} (esquema válido)")
    for tarea in tareas:
        print(f"  [{tarea['id']}] {tarea['tipo']:22s} {tarea['pregunta']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    global TARIFA_USD_POR_MILLON

    args = _parser().parse_args(argv)

    if args.verificar_dataset:
        return _verificar_dataset()

    try:
        datos = cargar_dataset()
    except (OSError, ValueError) as exc:
        print(f"[ERROR] No se pudo leer {RUTA_DATASET}: {exc}")
        return 1

    errores = validar_dataset(datos)
    if errores:
        print(f"[ERROR] Dataset inválido: {RUTA_DATASET}")
        for error in errores:
            print(f"  - {error}")
        return 1

    TARIFA_USD_POR_MILLON = float(args.tarifa_usd_por_millon)

    modos: list[str] = (
        list(config.MODOS_AGENTE) if args.modo == "ambos" else [str(args.modo)]
    )
    marca = datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta_json = (
        Path(args.salida)
        if args.salida is not None
        else config.DIR_EVAL_RESULTADOS / f"agente_{marca}.json"
    )

    print("=" * 72)
    print("EVALUACIÓN DEL AGENTE - TECNOAPOYO")
    print(f"Dataset: {RUTA_DATASET.name} | modos: {', '.join(modos)} | "
          f"tarifa: US$ {TARIFA_USD_POR_MILLON}/M tokens")
    print("=" * 72)

    try:
        corrida = _correr(datos, modos, args.limite)
    except Exception as exc:  # noqa: BLE001 - sin RAGEngine/índice no hay corrida
        print(f"\n[ERROR] La corrida no pudo arrancar: {type(exc).__name__}: {exc}")
        print("Revisa el índice FAISS (python scripts/build_index.py) y el .env.")
        return 1

    puntajes = [
        puntaje
        for resultado in corrida["corridas"]
        for puntaje in resultado["puntajes"]
    ]
    if puntajes and all(puntaje.get("error") for puntaje in puntajes):
        # Mismo criterio que la evaluación del motor: si el proveedor nunca
        # respondió, no hay evidencia válida y no se escribe ningún artefacto.
        print("\n" + "=" * 72)
        print("EVALUACIÓN ABORTADA: TODAS LAS TAREAS FALLARON CON EL PROVEEDOR")
        print(f"  Primer error: {puntajes[0]['error']}")
        print("  No se escribieron archivos: una corrida sin respuestas del")
        print("  modelo no produce evidencia. Revisa LLM_BASE_URL y LLM_API_KEY.")
        print("=" * 72)
        return 1

    resumen = {
        "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "modelo_agente": config.AGENTE_MODELO,
        "dataset": str(RUTA_DATASET.relative_to(config.RAIZ)),
        "verificacion_dataset": {"tareas": len(datos["tareas"]), "errores": []},
        "limite": args.limite,
        "tarifa_usd_por_millon": TARIFA_USD_POR_MILLON,
        "corridas": corrida["corridas"],
    }
    ruta_json, ruta_md = _escribir(resumen, ruta_json)

    for resultado in corrida["corridas"]:
        global_ = resultado["metricas"]["global"]
        print(
            f"\n[{resultado['modo']}] selección={_pct(global_['tasa_seleccion'])} "
            f"argumentos={_pct(global_['tasa_argumentos'])} "
            f"abstención={_pct(global_['tasa_abstencion'])} "
            f"memoria={_pct(global_['tasa_memoria'])} "
            f"confirmación={_pct(global_['tasa_confirmacion'])} "
            f"plan={_pct(global_['tasa_plan_valido'])} "
            f"replanes={global_['promedio_replanes']} "
            f"tokens={global_['tokens_totales']} "
            f"latencia={global_['latencia_total_seg']}s "
            f"costo=US$ {global_['costo_estimado_usd']}"
        )

    print(f"\n[OK] Resultados -> {ruta_json}")
    print(f"[OK] Resumen    -> {ruta_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
