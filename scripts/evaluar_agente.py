"""Evaluación del agente de TecnoApoyo (RA2 / IL2.1-IL2.4).

Mide el comportamiento del **agente**, no la recuperación del motor RAG: el
motor ya tiene su propia evaluación (`scripts/evaluate.py`,
`evaluation/dataset_evaluacion.json`). Aquí el contrato es
`evaluation/dataset_agente.json` y lo que se puntúa es la traza:

  - selección de herramienta (`esperado.herramienta`);
  - argumentos clave que efectivamente viajaron a esa herramienta;
  - abstención (precisión y recall, no solo "acertó");
  - uso del contexto conversacional en las tareas de memoria;
  - confirmación de escritura, en dos hechos separados: si hubo intento de
    escritura (`escritura_intentada`) y si ese intento se denegó sin mover el
    conteo de casos (`confirmacion_respetada`);
  - validez del plan en las tareas de planificación (solo modo planificado:
    en reactivo la métrica se reporta como `n/a`, no como 0 %).

El costo se estima con tarifas de referencia y una limitación declarada:
`RespuestaAgente` solo expone `tokens_total`, sin desglose entrada/salida, así
que se aplica la tarifa de entrada al total y se dice explícitamente en el
JSON y en el `.md`. Si no hay tarifa, el resumen dice "costo no configurado":
un `US$ 0.0` sin tarifa no es un costo, es una medición ausente.

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
import contextlib
import json
import re
import sys
import tempfile
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any

# Reconexión UTF-8, mismo bloque que src/app.py: en Windows la consola usa la
# codepage del sistema y las tildes y los signos de apertura se imprimen rotos.
for flujo in (sys.stdin, sys.stdout, sys.stderr):
    reconectar = getattr(flujo, "reconfigure", None)
    if reconectar is None:
        continue
    with contextlib.suppress(ValueError):  # flujos no reconectables o no tty
        reconectar(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402 - requiere el sys.path de arriba

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

# Tarifas de referencia del costo estimado, en USD por millón de tokens. Son
# globales de módulo porque `agregar_metricas` conserva su firma de un solo
# argumento posicional; `main` las fija con `--tarifa-in-usd-por-millon` y
# `--tarifa-out-usd-por-millon` (o las anula con `--sin-tarifa`).
#
# Limitación declarada: `RespuestaAgente` solo expone `tokens_total`, sin
# desglose entrada/salida. Por eso el costo se estima aplicando la tarifa de
# ENTRADA a los tokens totales. La tarifa de salida se registra en el artefacto
# para dejar constancia de con qué precios se calculó, pero no se aplica:
# inventar un reparto entrada/salida sería peor que declarar la limitación.
# Son referencias, no facturación del proveedor configurado.
TARIFA_IN_USD_POR_MILLON = 0.145
TARIFA_OUT_USD_POR_MILLON = 0.580

# Texto único para el costo no medido. Un `US$ 0.0` con tarifa ausente se lee
# como "la corrida no costó nada" cuando en realidad no se midió nada.
COSTO_NO_CONFIGURADO = "costo no configurado"

NOTA_COSTO = (
    "Estimación sobre tokens totales: RespuestaAgente no expone desglose "
    "entrada/salida, así que se aplica la tarifa de entrada al total de tokens. "
    "Es una estimación con tarifas de referencia, no una facturación."
)

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
    modo: str | None = None,
) -> dict:
    """Puntúa una traza contra su tarea. Puro: no ejecuta ni llama al modelo.

    `escrituras_antes`/`escrituras_despues` son el conteo de casos del
    repositorio alrededor de la consulta. Son la prueba de que la escritura
    denegada no ocurrió: una tool de escritura puede responder `ok=False` y
    aun así haber escrito, y este conteo es el que lo delata.

    `modo` es el modo del agente que produjo la traza. En `reactivo` no hay
    planificador, así que `plan_ok` queda en `None` (no aplica) en vez de
    `False`: una capacidad que el modo no tiene no puede penalizarlo.
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
    # Dos hechos distintos que antes se fundían en un solo booleano:
    #   `escritura_intentada`    -> hubo un intento real de escritura;
    #   `confirmacion_respetada` -> el intento se denegó y el conteo de casos
    #                               no se movió.
    # Una tarea que nunca llegó a escribir no "falló la confirmación": no hay
    # nada que confirmar. Eso se reporta como `None` (n/d), no como `False`.
    pasos_escritura = [
        paso
        for paso, nombre in zip(pasos, nombres, strict=True)
        if nombre in HERRAMIENTAS_ESCRITURA
    ]
    conteo_conocido = (
        escrituras_antes is not None and escrituras_despues is not None
    )
    conteo_estable = conteo_conocido and escrituras_antes == escrituras_despues
    # Un salto en el conteo también prueba que hubo intento, aunque la traza no
    # haya registrado el paso: una tool puede escribir y luego fallar.
    escritura_intentada = bool(pasos_escritura) or (
        conteo_conocido and not conteo_estable
    )
    if not escritura_intentada or not conteo_conocido:
        # Sin intento no hay confirmación que verificar; con intento pero sin
        # los dos conteos no se puede afirmar que la escritura no ocurrió. En
        # ambos casos el resultado es desconocido, no un fallo.
        confirmacion_respetada = None
    else:
        escritura_denegada = any(
            not bool(_leer(paso, "ok", False)) for paso in pasos_escritura
        )
        confirmacion_respetada = bool(escritura_denegada and conteo_estable)

    # -- plan --------------------------------------------------------------
    # En reactivo no hay planificador: la métrica no aplica y se excluye del
    # promedio en lugar de contarse como plan inválido.
    plan = _leer(respuesta, "plan", None)
    error_texto = str(_leer(respuesta, "error", None) or "")
    plan_aplica = modo is None or _normalizar(modo) != "reactivo"
    if not plan_aplica:
        plan_ok = None
    elif "plan_invalido" in error_texto or "no produjo un plan válido" in error_texto:
        # El planificador no logró un plan válido tras sus intentos. Eso es un
        # plan inválido aunque la tarea no declare `min_pasos_plan`: antes se
        # contaba como válido por defecto e inflaba la tasa.
        plan_ok = False
    elif minimo_plan is None:
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
        "escritura_intentada": bool(escritura_intentada),
        "confirmacion_respetada": (
            None if confirmacion_respetada is None else bool(confirmacion_respetada)
        ),
        "min_pasos_plan": minimo_plan,
        "plan_aplica": bool(plan_aplica),
        "plan_ok": None if plan_ok is None else bool(plan_ok),
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
    "confirmacion_respetada": "confirmacion",
    "plan_ok": "plan",
}


def _motivos_de_fallo(puntaje: dict) -> list[str]:
    """Nombres de las métricas que fallaron, para el informe legible.

    Solo un `False` explícito es un fallo. `None` significa "no aplica" o "no
    verificado" y no debe aparecer como error: reportar un dato desconocido
    como fallo es tan incorrecto como reportarlo como éxito.
    """
    return [
        etiqueta
        for clave, etiqueta in _METRICAS_LEGIBLES.items()
        if puntaje.get(clave, True) is False
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


def _confirmacion_de(puntaje: dict) -> bool | None:
    """Confirmación respetada, o `None` si no hubo intento o no se verificó.

    Acepta artefactos viejos con la clave `confirmacion_ok`, pero un puntaje
    que declara `escritura_intentada=False` nunca aporta a la tasa: no hubo
    escritura que confirmar.
    """
    if puntaje.get("escritura_intentada") is False:
        return None
    valor = puntaje.get("confirmacion_respetada", puntaje.get("confirmacion_ok"))
    return None if valor is None else bool(valor)


def _tarifas_actuales() -> dict:
    """Tarifas de referencia vigentes, en USD por millón de tokens.

    `configurada` es falso cuando la tarifa de entrada —la que efectivamente se
    multiplica por los tokens— es cero. Con `--sin-tarifa`, o con ambas tarifas
    en cero, el resumen dice "costo no configurado" en lugar de `US$ 0.0`.
    """
    tarifa_in = max(0.0, float(TARIFA_IN_USD_POR_MILLON))
    tarifa_out = max(0.0, float(TARIFA_OUT_USD_POR_MILLON))
    return {
        "in_usd_por_millon": tarifa_in,
        "out_usd_por_millon": tarifa_out,
        "aplicada": "in_usd_por_millon",
        "configurada": tarifa_in > 0.0,
        "base": "tokens_total",
        "desglose_entrada_salida": False,
        "nota": NOTA_COSTO,
    }


def _costo_estimado(tokens: int, tarifas: dict) -> float | None:
    """Costo estimado con la tarifa de entrada sobre el total de tokens."""
    if not tarifas["configurada"]:
        return None
    return round((tokens / 1_000_000) * tarifas["in_usd_por_millon"], 6)


def _texto_costo(valor: float | None) -> str:
    return COSTO_NO_CONFIGURADO if valor is None else f"US$ {valor:.6f}"


def _metricas_de_grupo(
    puntajes: list[dict], tarifas: dict, modo: str | None = None
) -> dict:
    """Tasas de un grupo (todas las tareas, o las de un tipo).

    `tasa_memoria` solo considera tareas de tipo 'memoria': en el resto la
    métrica es trivialmente cierta, y promediarla con las demás inflaría el
    indicador hasta hacerlo inútil.

    `tasa_confirmacion` solo promedia tareas con escritura intentada y conteo
    verificado; sin ninguna, es `None` (n/d) y no 0 %.

    `tasa_plan_valido` excluye los `None`: en modo reactivo no hay plan y la
    métrica no aplica.
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

    confirmaciones = [
        valor
        for valor in (_confirmacion_de(p) for p in puntajes)
        if valor is not None
    ]
    valores_plan = [
        bool(p.get("plan_ok")) for p in puntajes if p.get("plan_ok") is not None
    ]
    plan_aplica = not (modo is not None and _normalizar(modo) == "reactivo")
    costo = _costo_estimado(tokens, tarifas)

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
        "tasa_confirmacion": _tasa(confirmaciones),
        "tareas_con_escritura": sum(
            1 for p in puntajes if p.get("escritura_intentada")
        ),
        "plan_aplica": plan_aplica,
        "tasa_plan_valido": _tasa(valores_plan),
        "promedio_replanes": round(sum(replanes) / total, 2) if total else 0.0,
        "tokens_totales": tokens,
        "latencia_total_seg": round(latencia, 2),
        "costo_configurado": tarifas["configurada"],
        "costo_estimado_usd": costo,
        "costo_texto": _texto_costo(costo),
    }


def agregar_metricas(
    puntajes: list[dict],
    *,
    modo: str | None = None,
    tarifas: dict | None = None,
) -> dict:
    """Agrega los puntajes por tipo de tarea y en global.

    `modo` decide si la métrica de plan aplica (en reactivo no) y `tarifas`
    permite fijarlas explícitamente; por defecto salen de los globales del
    módulo que `main` configura desde la CLI.
    """
    tarifas = dict(tarifas) if tarifas is not None else _tarifas_actuales()
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
            por_tipo[tipo] = _metricas_de_grupo(grupo, tarifas, modo)

    return {
        "global": _metricas_de_grupo(puntajes, tarifas, modo),
        "por_tipo": por_tipo,
        "modo": modo,
        "tarifas": tarifas,
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
                    modo=modo,
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
                "metricas": agregar_metricas(puntajes, modo=modo),
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


def _pct_plan(metricas: dict) -> str:
    """Plan como `n/a` cuando el modo no lo produce (reactivo)."""
    if not metricas.get("plan_aplica", True):
        return "n/a"
    return _pct(metricas.get("tasa_plan_valido"))


def _lineas_de_tarifas(resumen: dict) -> list[str]:
    """Encabezado del resumen: qué tarifas se usaron y qué base tiene el costo."""
    tarifas = resumen.get("tarifas") or _tarifas_actuales()
    if not tarifas.get("configurada"):
        return [
            f"- Costo: **{COSTO_NO_CONFIGURADO}** (sin tarifa no hay estimación;",
            "  una tarifa ausente no es un costo cero).",
        ]
    return [
        "- Tarifas de referencia: "
        f"US$ {tarifas['in_usd_por_millon']:g}/millón de tokens de entrada y "
        f"US$ {tarifas['out_usd_por_millon']:g}/millón de salida.",
        "  Son referencias configurables, no una facturación del proveedor.",
        "- Base del costo: **tokens totales**, sin desglose entrada/salida "
        "(la traza no lo expone).",
        "  Se aplica la tarifa de entrada al total: es una estimación, no una "
        "medición facturada.",
    ]


def _markdown(resumen: dict) -> str:
    """Resumen legible: una tabla global por modo y el detalle de fallos."""
    lineas = [
        "# Evaluación del agente de TecnoApoyo",
        "",
        f"- Fecha: {resumen['fecha']}",
        f"- Modelo del agente: {resumen['modelo_agente']}",
        f"- Dataset: {resumen['dataset']} "
        f"({resumen['verificacion_dataset']['tareas']} tareas)",
        *_lineas_de_tarifas(resumen),
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
            "| Tasa de confirmación de escritura | "
            f"{_pct(global_['tasa_confirmacion'])} "
            f"({global_.get('tareas_con_escritura', 0)} tareas con escritura "
            "intentada) |",
            f"| Tasa de plan válido | {_pct_plan(global_)} |",
            f"| Promedio de re-planes | {global_['promedio_replanes']} |",
            f"| Tokens totales | {global_['tokens_totales']} |",
            f"| Latencia total (s) | {global_['latencia_total_seg']} |",
            f"| Costo estimado | {global_.get('costo_texto') or COSTO_NO_CONFIGURADO} |",
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
                f"{_pct_plan(metricas)} |"
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
        "--tarifa-in-usd-por-millon",
        type=float,
        default=TARIFA_IN_USD_POR_MILLON,
        help=(
            "USD por millón de tokens de entrada, tarifa de referencia "
            f"(default: {TARIFA_IN_USD_POR_MILLON}). Se aplica a los tokens "
            "totales: la traza no trae desglose entrada/salida."
        ),
    )
    parser.add_argument(
        "--tarifa-out-usd-por-millon",
        type=float,
        default=TARIFA_OUT_USD_POR_MILLON,
        help=(
            "USD por millón de tokens de salida, tarifa de referencia "
            f"(default: {TARIFA_OUT_USD_POR_MILLON}). Se registra en el "
            "artefacto; no se aplica porque no hay desglose de salida."
        ),
    )
    parser.add_argument(
        "--sin-tarifa",
        action="store_true",
        help=(
            "No estima costo: anula ambas tarifas y el resumen dice "
            f"'{COSTO_NO_CONFIGURADO}'."
        ),
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


def _tarifas_desde_args(args: argparse.Namespace) -> tuple[float, float]:
    """Resuelve (tarifa_in, tarifa_out) desde la CLI. Puro y testeable.

    `--sin-tarifa` gana sobre las tarifas explícitas: pedir "sin tarifa" y
    además precios es contradictorio, y la lectura segura es no estimar.
    """
    if getattr(args, "sin_tarifa", False):
        return 0.0, 0.0
    return (
        float(args.tarifa_in_usd_por_millon),
        float(args.tarifa_out_usd_por_millon),
    )


def _texto_tarifas(tarifas: dict) -> str:
    """Línea de encabezado con las tarifas o el aviso de costo ausente."""
    if not tarifas["configurada"]:
        return f"tarifas: {COSTO_NO_CONFIGURADO}"
    return (
        f"tarifas de referencia: US$ {tarifas['in_usd_por_millon']:g}/M in, "
        f"US$ {tarifas['out_usd_por_millon']:g}/M out"
    )


def main(argv: list[str] | None = None) -> int:
    global TARIFA_IN_USD_POR_MILLON, TARIFA_OUT_USD_POR_MILLON

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

    tarifa_in, tarifa_out = _tarifas_desde_args(args)
    if tarifa_in < 0.0 or tarifa_out < 0.0:
        print("[ERROR] Las tarifas no pueden ser negativas: un costo estimado")
        print("        negativo no significa nada. Usa 0 o --sin-tarifa para")
        print(f"        reportar '{COSTO_NO_CONFIGURADO}'.")
        return 1
    TARIFA_IN_USD_POR_MILLON = tarifa_in
    TARIFA_OUT_USD_POR_MILLON = tarifa_out
    tarifas = _tarifas_actuales()

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
          f"{_texto_tarifas(tarifas)}")
    if tarifas["configurada"]:
        print("Costo estimado sobre tokens totales (sin desglose entrada/salida):")
        print("es una referencia, no una facturación.")
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
        "tarifas": tarifas,
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
            f"plan={_pct_plan(global_)} "
            f"replanes={global_['promedio_replanes']} "
            f"tokens={global_['tokens_totales']} "
            f"latencia={global_['latencia_total_seg']}s "
            f"costo={global_['costo_texto']}"
        )

    print(f"\n[OK] Resultados -> {ruta_json}")
    print(f"[OK] Resumen    -> {ruta_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
