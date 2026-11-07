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
  - cumplimiento del efecto declarado (`esperado.efecto`): la única prueba de
    que una escritura APROBADA se completó. La traza dice qué herramienta se
    llamó, no qué quedó guardado, así que el efecto se verifica contra el
    repositorio después de la consulta (`efecto_ok`);
  - validez del plan en las tareas de planificación (solo modo planificado:
    en reactivo la métrica se reporta como `n/a`, no como 0 %);
  - alcance de la sesión (`alcance`/`cliente_sesion`): un segundo agente con
    las tools atadas al cliente de la tarea, sobre el mismo repositorio, y con
    `al_identificar` igual que la consola: si el agente identifica a un cliente
    en el turno, las tools siguientes trabajan sobre ese cliente;
  - multi-turno real (`secuencia`): una `consultar()` por mensaje, sin limpiar
    la memoria entre mensajes, y se puntúa el último turno;
  - datos faltantes (`esperado.pregunta_clave`) sobre la pregunta que el
    agente hizo con `pedir_datos`;
  - capacidades deterministas (`esperado.sin_tokens`), respuestas que deben
    mencionar un término (`esperado.menciona`) y tareas que no deben escribir
    (`esperado.sin_escritura`).

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
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, TypeGuard

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
from src.capacidades import TEXTO_CAPACIDADES  # noqa: E402 - módulo puro

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

# Alcances con los que se puede evaluar una tarea: `global` usa el agente del
# dominio completo y `sesion` el que tiene las tools atadas a un cliente.
ALCANCES_TAREA = ("global", "sesion")

# Efectos verificables después de la consulta: el tipo y las claves que ese
# tipo necesita para poder comprobarse contra el repositorio. Un efecto sin sus
# claves no se puede verificar, y darlo por cumplido sería inventar evidencia.
TIPOS_EFECTO = {
    "caso_creado": ("cliente", "estado"),
    "estado_actualizado": ("id_caso", "estado"),
    "nota_agregada": ("id_caso", "nota"),
}

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
    try:
        with destino.open(encoding="utf-8") as archivo:
            datos = json.load(archivo)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"No se pudo leer el dataset {destino}: {exc}") from exc
    return datos


def _es_entero(valor: Any) -> TypeGuard[int]:
    """True para enteros reales: `bool` es subclase de `int` en Python.

    Es un `TypeGuard` para que los topes numéricos (`min_pasos_plan`,
    `replanes_max`) se comparen como enteros después de la validación y no como
    `Any | None`.
    """
    return isinstance(valor, int) and not isinstance(valor, bool)


def _errores_de_efecto(etiqueta: str, efecto: Any) -> list[str]:
    """Errores del `esperado.efecto` de una tarea; lista vacía = válido.

    Un efecto mal declarado es peor que no declararlo: se verificaría contra el
    repositorio con claves ausentes y el resultado sería un fallo del agente
    que en realidad es un defecto del dataset.
    """
    if not isinstance(efecto, dict):
        return [
            f"Tarea {etiqueta}: 'esperado.efecto' debe ser un objeto con 'tipo' "
            "y las claves de ese tipo."
        ]
    tipo = efecto.get("tipo")
    if tipo not in TIPOS_EFECTO:
        return [
            f"Tarea {etiqueta}: 'esperado.efecto.tipo' desconocido {tipo!r}. "
            f"Validos: {', '.join(TIPOS_EFECTO)}."
        ]
    errores = []
    for clave in TIPOS_EFECTO[tipo]:
        valor = efecto.get(clave)
        if not isinstance(valor, str) or not valor.strip():
            errores.append(
                f"Tarea {etiqueta}: 'esperado.efecto' de tipo '{tipo}' necesita "
                f"'{clave}' (texto no vacio)."
            )
    return errores


def validar_dataset(datos: dict) -> list[str]:
    """Devuelve los errores de esquema del dataset; lista vacía = válido.

    Valida forma, no semántica de negocio: no comprueba que el nombre de una
    herramienta exista en `src.herramientas` porque ese catálogo vive en un
    módulo que no queremos importar en el camino offline, y una lista
    duplicada aquí quedaría desactualizada sin que nadie lo note.

    Las claves nuevas del contrato (`alcance`, `cliente_sesion`, `secuencia` y
    las expectativas `pregunta_clave`, `sin_tokens`, `sin_escritura`,
    `menciona`, `replanes_max`) son OPCIONALES: una tarea que no las declara se
    comporta exactamente como antes. Se rechazan las combinaciones que no
    tienen lectura posible, porque una tarea mal declarada se vería como un
    fallo del agente.
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
        secuencia = tarea.get("secuencia")
        if "secuencia" in tarea:
            if not isinstance(secuencia, list):
                errores.append(
                    f"Tarea {etiqueta}: 'secuencia' debe ser una lista de "
                    "mensajes."
                )
            elif not secuencia:
                errores.append(
                    f"Tarea {etiqueta}: 'secuencia' no puede estar vacía; una "
                    "tarea sin mensajes no se puede evaluar."
                )
            elif any(
                not isinstance(mensaje, str) or not mensaje.strip()
                for mensaje in secuencia
            ):
                errores.append(
                    f"Tarea {etiqueta}: los elementos de 'secuencia' deben ser "
                    "texto no vacio."
                )
            if "pregunta" in tarea:
                errores.append(
                    f"Tarea {etiqueta}: 'secuencia' reemplaza a 'pregunta' y no "
                    "pueden declararse las dos a la vez."
                )
        # `pregunta` es obligatoria salvo cuando la reemplaza `secuencia`.
        if "secuencia" not in tarea and not (
            isinstance(pregunta, str) and pregunta.strip()
        ):
            errores.append(f"Tarea {etiqueta}: falta 'pregunta' (texto no vacio).")

        alcance = tarea.get("alcance")
        if "alcance" in tarea and alcance not in ALCANCES_TAREA:
            errores.append(
                f"Tarea {etiqueta}: 'alcance' desconocido {alcance!r}. "
                f"Validos: {', '.join(ALCANCES_TAREA)}."
            )

        if "cliente_sesion" in tarea:
            cliente = tarea.get("cliente_sesion")
            if not isinstance(cliente, str) or not cliente.strip():
                errores.append(
                    f"Tarea {etiqueta}: 'cliente_sesion' debe ser texto no "
                    "vacío."
                )
            if alcance != "sesion":
                errores.append(
                    f"Tarea {etiqueta}: 'cliente_sesion' exige "
                    "'alcance': 'sesion'."
                )

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

        for booleano in ("abstencion", "requiere_confirmacion",
                         "confirmacion_aprobada", "sin_tokens",
                         "sin_escritura"):
            if booleano in esperado and not isinstance(esperado[booleano], bool):
                errores.append(
                    f"Tarea {etiqueta}: 'esperado.{booleano}' debe ser booleano."
                )

        for texto in ("pregunta_clave", "menciona"):
            if texto in esperado:
                valor = esperado[texto]
                if not isinstance(valor, str) or not valor.strip():
                    errores.append(
                        f"Tarea {etiqueta}: 'esperado.{texto}' debe ser texto "
                        "no vacío."
                    )

        if "replanes_max" in esperado:
            tope = esperado["replanes_max"]
            if not _es_entero(tope) or tope < 0:
                errores.append(
                    f"Tarea {etiqueta}: 'esperado.replanes_max' debe ser un "
                    "entero >= 0."
                )

        if "efecto" in esperado:
            errores.extend(_errores_de_efecto(etiqueta, esperado["efecto"]))

        if "min_pasos_plan" in esperado:
            minimo = esperado["min_pasos_plan"]
            if minimo is not None and (not _es_entero(minimo) or minimo < 1):
                errores.append(
                    f"Tarea {etiqueta}: 'esperado.min_pasos_plan' debe ser un "
                    "entero >= 1 o null."
                )

        # Una respuesta sin modelo no tiene plan ni pasos ni multi-turno: la
        # combinación se rechaza porque no habría nada que medir en la otra
        # expectativa, y el puntaje saldría en rojo por un defecto del dataset.
        if esperado.get("sin_tokens") is True:
            conflictos = []
            if esperado.get("herramienta") is not None:
                conflictos.append("'esperado.herramienta'")
            if esperado.get("min_pasos_plan") is not None:
                conflictos.append("'esperado.min_pasos_plan'")
            if "secuencia" in tarea:
                conflictos.append("'secuencia'")
            if conflictos:
                errores.append(
                    f"Tarea {etiqueta}: 'esperado.sin_tokens' exige una "
                    "respuesta determinista y no puede combinarse con "
                    f"{', '.join(conflictos)}."
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


def _pregunta_pedida(pasos: list[Any]) -> str | None:
    """Pregunta del último paso `pedir_datos` exitoso, o `None`.

    La salida de la tool es un JSON (`{"datos": {"pregunta": ...}}`), pero un
    artefacto releído puede traerla ya como dict: se aceptan las dos formas.
    Se toma el último paso exitoso, no el último paso: si después de pedir el
    dato hubo otra llamada, la pregunta que se mostró al cliente sigue siendo
    la de `pedir_datos`.
    """
    for paso in reversed(pasos):
        if str(_leer(paso, "herramienta", "") or "") != "pedir_datos":
            continue
        if not bool(_leer(paso, "ok", False)):
            continue
        salida = _leer(paso, "salida", None)
        if isinstance(salida, str):
            try:
                salida = json.loads(salida)
            except (TypeError, ValueError):
                return None
        datos = salida.get("datos") if isinstance(salida, dict) else None
        pregunta = datos.get("pregunta") if isinstance(datos, dict) else None
        if isinstance(pregunta, str) and pregunta.strip():
            return pregunta
        return None
    return None


def _resumen_plan(plan: Any) -> dict | None:
    """Plan serializable para el artefacto.

    El planificador devuelve modelos Pydantic, que `json.dumps` no sabe
    escribir: el JSON de resultados se rompería en modo planificado. Se guarda
    lo que hace legible la traza (objetivo y herramientas por paso) y no el
    plan entero, que duplicaría el prompt en el artefacto.
    """
    if plan is None:
        return None
    if isinstance(plan, (str, int, float, bool)):
        return {"objetivo": str(plan), "pasos": []}
    crudos = _leer(plan, "pasos", None)
    pasos = []
    if isinstance(crudos, (list, tuple)):
        for paso in crudos:
            pasos.append({
                "id": str(_leer(paso, "id", "") or ""),
                "herramienta": str(_leer(paso, "herramienta", "") or ""),
                "depende_de": [
                    str(dependencia)
                    for dependencia in (_leer(paso, "depende_de", None) or [])
                ],
            })
    return {"objetivo": str(_leer(plan, "objetivo", "") or ""), "pasos": pasos}


def _pasos_detalle(pasos: list[Any]) -> list[dict]:
    """Resumen serializable de cada paso: qué se llamó, con qué y si funcionó.

    Se guarda junto al puntaje para poder diagnosticar después sin volver a
    ejecutar la corrida. No incluye `salida`/`error` por paso: la traza
    duplicaría todo el contexto del modelo en el artefacto JSON.
    """
    return [
        {
            "herramienta": str(_leer(paso, "herramienta", "") or ""),
            "argumentos": dict(_argumentos_de(paso)),
            "ok": bool(_leer(paso, "ok", False)),
        }
        for paso in pasos
    ]


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
# Efecto declarado de una escritura
# ---------------------------------------------------------------------------
def _hay_caso_creado(efecto: dict, repositorio: Any) -> bool:
    """True si existe un caso con ese cliente y ese estado.

    El límite se calcula con `contar()` para ver todos los casos de la corrida;
    `listar_casos` lo recorta a 100, que alcanza de sobra para una base aislada
    sembrada con 10 casos y las pocas escrituras de una corrida.
    """
    cliente = _normalizar(efecto.get("cliente"))
    estado = _normalizar(efecto.get("estado"))
    if not cliente or not estado:
        return False
    limite = max(1, int(repositorio.contar()))
    casos = repositorio.listar_casos(limite=limite)
    return any(
        _normalizar(caso.get("cliente")) == cliente
        and _normalizar(caso.get("estado")) == estado
        for caso in casos
    )


def _efecto_cumplido(efecto: Any, repositorio: Any) -> bool | None:
    """Comprueba en el repositorio el efecto que la tarea declara.

    Es la única prueba de que una escritura APROBADA se completó: la traza dice
    qué herramienta se llamó, no qué quedó guardado. Devuelve `None` cuando no
    hay efecto que verificar o no hay repositorio con qué verificar, para no
    confundir "no aplica" con "no se cumplió".
    """
    if not isinstance(efecto, dict) or repositorio is None:
        return None
    tipo = efecto.get("tipo")
    try:
        if tipo == "caso_creado":
            return _hay_caso_creado(efecto, repositorio)
        if tipo == "estado_actualizado":
            caso = repositorio.obtener_caso(str(efecto.get("id_caso")))
            return _normalizar(caso.get("estado")) == _normalizar(
                efecto.get("estado")
            )
        if tipo == "nota_agregada":
            caso = repositorio.obtener_caso(str(efecto.get("id_caso")))
            buscada = _normalizar(efecto.get("nota"))
            return bool(buscada) and any(
                buscada in _normalizar(nota) for nota in caso.get("notas") or []
            )
    except Exception:  # noqa: BLE001 - un caso ausente es "no cumplido"
        # Un id que no existe, o una base ilegible, no cumplen el efecto; dejar
        # que la excepción aborte la corrida perdería el resto de las tareas.
        return False
    # Un tipo desconocido no puede darse por cumplido; `validar_dataset` ya lo
    # rechaza antes de correr, así que aquí solo queda la defensa.
    return False


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
    repositorio: Any | None = None,
    sin_escritura: bool | None = None,
) -> dict:
    """Puntúa una traza contra su tarea. Puro: no ejecuta ni llama al modelo.

    `escrituras_antes`/`escrituras_despues` son el conteo de casos del
    repositorio alrededor de la consulta. Son la prueba de que la escritura
    denegada no ocurrió: una tool de escritura puede responder `ok=False` y
    aun así haber escrito, y este conteo es el que lo delata.

    `repositorio` solo se lee para verificar `esperado.efecto` después de la
    consulta. Sin él, `efecto_ok` queda en `None` (no verificado) en vez de
    `False`: un dato ausente no es un incumplimiento.

    `modo` es el modo del agente que produjo la traza. En `reactivo` no hay
    planificador, así que `plan_ok` queda en `None` (no aplica) en vez de
    `False`: una capacidad que el modo no tiene no puede penalizarlo.

    `sin_escritura` es el resultado del invariante de conteo que calcula el
    llamador (`True` = el número de casos no se movió). Es un argumento
    opcional y de solo palabra clave: la puntuación pura no puede consultar el
    repositorio dos veces, y sin el conteo la expectativa `sin_escritura`
    queda sin verificar (`None`), nunca aprobada por defecto.
    """
    esperado = (tarea.get("esperado") if isinstance(tarea, dict) else None) or {}
    tipo = tarea.get("tipo") if isinstance(tarea, dict) else None
    herramienta_esperada = esperado.get("herramienta")
    claves = esperado.get("argumentos_clave") or {}
    if not isinstance(claves, dict):
        claves = {}
    abstencion_esperada = bool(esperado.get("abstencion", False))
    requiere_confirmacion = bool(esperado.get("requiere_confirmacion", False))
    confirmacion_aprobada = esperado.get("confirmacion_aprobada") is True
    efecto = esperado.get("efecto")
    if not isinstance(efecto, dict):
        efecto = None
    minimo_plan = esperado.get("min_pasos_plan")
    if not _es_entero(minimo_plan):
        minimo_plan = None

    pasos = _pasos_de(respuesta)
    nombres = [str(_leer(paso, "herramienta", "") or "") for paso in pasos]
    abstencion_obtenida = bool(_leer(respuesta, "abstencion", False))
    texto_respuesta = str(_leer(respuesta, "respuesta", "") or "")
    tokens_total = int(_leer(respuesta, "tokens_total", 0) or 0)
    replanes_obtenidos = int(_leer(respuesta, "replanes", 0) or 0)

    # La tarea de `secuencia` no declara `pregunta`: la pregunta que se puntúa
    # es la del último mensaje, y es la que documenta el puntaje.
    pregunta_declarada = tarea.get("pregunta") if isinstance(tarea, dict) else None
    pregunta_puntaje = (
        pregunta_declarada
        if isinstance(pregunta_declarada, str) and pregunta_declarada.strip()
        else _leer(respuesta, "pregunta", None)
    )

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
    if confirmacion_aprobada:
        # La tarea espera una escritura APROBADA: aquí no hay nada que
        # denegar, así que el contrato de "no escribir sin permiso" no aplica
        # (`None`, n/d) y lo que prueba el cumplimiento es `efecto_ok`.
        # Marcarlo como fallo haría que una tarea bien resuelta se reporte mal.
        confirmacion_respetada = None
    elif not escritura_intentada or not conteo_conocido:
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

    # -- datos faltantes: la pregunta que el agente hizo -------------------
    # `pregunta_clave` no exige que la tool sea la única del turno, pero sí que
    # la pregunta mostrada al cliente contenga lo que el caso necesitaba: es
    # la diferencia entre pedir un dato útil y pedir "más información".
    pregunta_clave = esperado.get("pregunta_clave")
    pregunta_clave_aplica = (
        isinstance(pregunta_clave, str) and bool(pregunta_clave.strip())
    )
    pregunta_pedida = _pregunta_pedida(pasos)
    if not pregunta_clave_aplica:
        pregunta_clave_ok = None
    else:
        pregunta_clave_ok = pregunta_pedida is not None and _normalizar(
            pregunta_clave
        ) in _normalizar(pregunta_pedida)

    # -- capacidades deterministas ----------------------------------------
    # Una respuesta sin modelo no gasta tokens y es exactamente el texto
    # canónico; se comparan las dos cosas porque cualquiera de las dos sola se
    # cumple por accidente (una abstención también puede tener 0 tokens).
    sin_tokens_aplica = esperado.get("sin_tokens") is True
    sin_tokens_ok = (
        tokens_total == 0 and texto_respuesta == TEXTO_CAPACIDADES
        if sin_tokens_aplica
        else None
    )

    # -- mención en la respuesta ------------------------------------------
    termino = esperado.get("menciona")
    menciona_aplica = isinstance(termino, str) and bool(termino.strip())
    menciona_ok = (
        _normalizar(termino) in _normalizar(texto_respuesta)
        and not abstencion_obtenida
        if menciona_aplica
        else None
    )

    # -- re-planes ---------------------------------------------------------
    tope_replanes = esperado.get("replanes_max")
    if _es_entero(tope_replanes) and tope_replanes >= 0:
        replanes_aplica = True
        replanes_ok = replanes_obtenidos <= tope_replanes
    else:
        # Un tope ausente o ilegible no es un tope cero: no aplica.
        tope_replanes = None
        replanes_aplica = False
        replanes_ok = None

    # -- escritura prohibida ----------------------------------------------
    # Dos pruebas independientes: el conteo de casos no se movió (lo calcula el
    # llamador) y la traza no muestra un `crear_caso` exitoso. Con una sola,
    # una tool que escribe y luego falla pasaría como "no escribió".
    sin_escritura_aplica = esperado.get("sin_escritura") is True
    hubo_creacion = any(
        nombre == "crear_caso" and bool(_leer(paso, "ok", False))
        for paso, nombre in zip(pasos, nombres, strict=True)
    )
    if not sin_escritura_aplica or sin_escritura is None:
        sin_escritura_ok = None
    else:
        sin_escritura_ok = bool(sin_escritura) and not hubo_creacion

    secuencia = tarea.get("secuencia") if isinstance(tarea, dict) else None
    mensajes_secuencia = secuencia if isinstance(secuencia, list) else []
    multiturno = bool(mensajes_secuencia)

    puntaje = {
        "id": tarea.get("id") if isinstance(tarea, dict) else None,
        "tipo": tipo,
        "pregunta": pregunta_puntaje,
        "alcance": (
            _normalizar(tarea.get("alcance") or "global")
            if isinstance(tarea, dict) else "global"
        ),
        "herramienta_esperada": herramienta_esperada,
        "herramienta_ok": bool(herramienta_ok),
        "argumentos_ok": bool(argumentos_ok),
        "abstencion_esperada": abstencion_esperada,
        "abstencion_obtenida": abstencion_obtenida,
        "abstencion_ok": bool(abstencion_ok),
        "memoria_ok": bool(memoria_ok),
        "requiere_confirmacion": requiere_confirmacion,
        "confirmacion_aprobada": confirmacion_aprobada,
        "escritura_intentada": bool(escritura_intentada),
        "confirmacion_respetada": (
            None if confirmacion_respetada is None else bool(confirmacion_respetada)
        ),
        "efecto": efecto,
        "efecto_ok": (
            None if efecto is None else _efecto_cumplido(efecto, repositorio)
        ),
        "min_pasos_plan": minimo_plan,
        "plan_aplica": bool(plan_aplica),
        "plan_ok": None if plan_ok is None else bool(plan_ok),
        # Plan serializable: el modelo Pydantic del planificador no lo es.
        "plan": _resumen_plan(plan),
        "pasos": len(pasos),
        "herramientas": nombres,
        # Expectativas nuevas. Cada una trae su bandera de aplicabilidad: una
        # tarea que no las declara no entra al denominador de su tasa, y así
        # un `None` no se confunde con un fallo.
        "pregunta_pedida": pregunta_pedida,
        "pregunta_clave": pregunta_clave if pregunta_clave_aplica else None,
        "pregunta_clave_aplica": bool(pregunta_clave_aplica),
        "pregunta_clave_ok": (
            None if pregunta_clave_ok is None else bool(pregunta_clave_ok)
        ),
        "sin_tokens_aplica": bool(sin_tokens_aplica),
        "sin_tokens_ok": None if sin_tokens_ok is None else bool(sin_tokens_ok),
        "menciona": termino if menciona_aplica else None,
        "menciona_aplica": bool(menciona_aplica),
        "menciona_ok": None if menciona_ok is None else bool(menciona_ok),
        "replanes_max": tope_replanes,
        "replanes_aplica": bool(replanes_aplica),
        "replanes_ok": None if replanes_ok is None else bool(replanes_ok),
        "sin_escritura_aplica": bool(sin_escritura_aplica),
        "sin_escritura_ok": (
            None if sin_escritura_ok is None else bool(sin_escritura_ok)
        ),
        "multiturno": bool(multiturno),
        "multiturno_ok": None,
        "turnos": len(mensajes_secuencia) if multiturno else 1,
        # Texto y detalle de la traza: sin ellos un fallo de argumento es
        # indiagnosticable (solo se veía el nombre de la herramienta). Se leen
        # con `_leer` porque un camino de error puede devolver un objeto mínimo.
        "respuesta": str(_leer(respuesta, "respuesta", "") or ""),
        "pasos_detalle": _pasos_detalle(pasos),
        "replanes": replanes_obtenidos,
        "tokens": tokens_total,
        "latencia_seg": round(float(_leer(respuesta, "latencia_seg", 0.0) or 0.0), 3),
        "error": _leer(respuesta, "error", None),
        "escrituras_antes": escrituras_antes,
        "escrituras_despues": escrituras_despues,
    }
    puntaje["motivos"] = _motivos_de_fallo(puntaje)
    if multiturno:
        # La tarea multi-turno se considera cumplida cuando la traza del último
        # turno cumple todo su contrato: es el único indicador que mira el
        # turno final en el contexto de los anteriores, no cada check suelto.
        puntaje["multiturno_ok"] = not puntaje["motivos"]
    return puntaje


_METRICAS_LEGIBLES = {
    "herramienta_ok": "herramienta",
    "argumentos_ok": "argumentos",
    "abstencion_ok": "abstencion",
    "memoria_ok": "memoria",
    "confirmacion_respetada": "confirmacion",
    "efecto_ok": "efecto",
    "plan_ok": "plan",
    "pregunta_clave_ok": "pregunta_pedida",
    "sin_tokens_ok": "sin_tokens",
    "menciona_ok": "mencion",
    "replanes_ok": "replanes",
    "sin_escritura_ok": "escritura",
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

    `tasa_resolucion` es la tasa de efectos cumplidos sobre las tareas que
    declaran `esperado.efecto` y que se pudieron verificar. Sin tareas con
    efecto es `None` (n/d), no 0 %: no hay nada que resolver.

    `tasa_plan_valido` excluye los `None`: en modo reactivo no hay plan y la
    métrica no aplica.

    Las cuatro tasas de la ola 1 (`tasa_pedir_datos`, `tasa_capacidades`,
    `tasa_multiturno`, `tasa_sin_escritura`) se calculan solo sobre las tareas
    que declaran esa expectativa, y son `None` cuando ninguna la declara: una
    métrica sin denominador es "no aplica", no 0 %.
    """
    total = len(puntajes)

    def columna(nombre: str) -> list[bool]:
        return [bool(puntaje.get(nombre)) for puntaje in puntajes]

    def aplicables(aplica: str, clave: str) -> list[bool]:
        """Valores de una expectativa opcional: solo donde aplica y se verificó.

        Una expectativa aplicable pero sin verificar (`None`: falta el conteo, la
        traza no trae la salida) no es un fallo; promediarla como `False`
        castigaría al agente por un dato que la corrida no capturó.
        """
        return [
            bool(puntaje.get(clave))
            for puntaje in puntajes
            if puntaje.get(aplica) and puntaje.get(clave) is not None
        ]

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
    # Solo entran al promedio los efectos verificados: uno sin verificar no es
    # un éxito ni un fallo, y promediarlo como cualquiera de los dos mentiría.
    resoluciones = [
        bool(p.get("efecto_ok"))
        for p in puntajes
        if p.get("efecto") is not None and p.get("efecto_ok") is not None
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
        "tareas_con_efecto": sum(
            1 for p in puntajes if p.get("efecto") is not None
        ),
        "resoluciones_ok": sum(1 for valor in resoluciones if valor),
        "tasa_resolucion": _tasa(resoluciones),
        "plan_aplica": plan_aplica,
        "tasa_plan_valido": _tasa(valores_plan),
        "tasa_pedir_datos": _tasa(
            aplicables("pregunta_clave_aplica", "pregunta_clave_ok")
        ),
        "tasa_capacidades": _tasa(aplicables("sin_tokens_aplica", "sin_tokens_ok")),
        "tasa_multiturno": _tasa(aplicables("multiturno", "multiturno_ok")),
        "tasa_sin_escritura": _tasa(
            aplicables("sin_escritura_aplica", "sin_escritura_ok")
        ),
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
    moverse. Es además la política por defecto de una corrida: una tarea sin
    `esperado.confirmacion_aprobada` nunca autoriza una escritura.
    """
    return False


def _confirmador_de_tarea(tarea: dict) -> Callable[[str, dict], bool]:
    """Confirmador que corresponde a una tarea: aprueba solo si lo declara.

    Función pura: la política sale de `esperado.confirmacion_aprobada` y de
    nada más. Una tarea sin la clave —o con `false`— se deniega, que es el
    default seguro: la evaluación no escribe casos salvo que el contrato lo
    pida explícitamente (T15), y T12 debe seguir denegándose.
    """
    esperado = (tarea.get("esperado") if isinstance(tarea, dict) else None) or {}
    aprobada = esperado.get("confirmacion_aprobada") is True

    def confirmar(_accion: str, _resumen: dict) -> bool:
        return aprobada

    return confirmar


class _ConfirmadorDeCorrida:
    """Confirmador único con política intercambiable antes de cada tarea.

    Las herramientas se construyen una sola vez por modo y capturan el
    confirmador en su clausura; reconstruirlas por tarea sería el cambio
    invasivo (todas las tools por tarea y una traza distinta por tarea). En vez de
    eso se cambia la política antes de cada consulta: la tool sigue siendo la
    misma y decide con el criterio de la tarea vigente, así que T12 se deniega
    y T15 se aprueba en la misma corrida.
    """

    def __init__(self) -> None:
        self._confirmador: Callable[[str, dict], bool] = confirmar_denegando

    def usar(self, tarea: dict) -> None:
        """Fija la política de la tarea que se va a evaluar."""
        self._confirmador = _confirmador_de_tarea(tarea)

    def __call__(self, accion: str, resumen: dict) -> bool:
        return self._confirmador(accion, resumen)


def _preparar_memoria(memoria: Any, tarea: dict) -> None:
    """Carga los turnos previos de la tarea y descarta los de la anterior.

    Sin este reinicio, el contexto de una tarea de memoria se filtraría a la
    siguiente y la métrica de memoria mediría arrastre, no memoria.

    En una tarea de `secuencia` se llama **una sola vez**, antes del primer
    mensaje: limpiar entre mensajes convertiría el multi-turno en N preguntas
    sueltas y el agente no tendría que seguir el hilo.
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


class _ClienteDeSesion:
    """Cliente de la sesión, mutable entre tareas de la misma corrida.

    Las tools de `alcance="sesion"` resuelven el cliente en cada llamada, así
    que basta con cambiar este valor antes de cada tarea: no hay que
    reconstruir el agente por tarea, y la comparación entre tareas sigue
    midiendo el mismo agente. Se limpia después de cada tarea para que un
    cliente no se filtre a la siguiente y haga pasar una tarea por sesión.

    Es también el `al_identificar` de las tools de sesión (ver `_correr`): en la
    consola ese papel lo cumple `establecer_cliente` sobre `config`, y aquí el
    mismo gesto escribe sobre este holder mutable. Por eso `usar` acepta un
    nombre `None`/vacío igual que `establecer_cliente`: identificar sin nombre
    borra el alcance en lugar de dejarlo a medias.
    """

    def __init__(self) -> None:
        self._cliente: str | None = None

    def usar(self, cliente: Any) -> None:
        texto = str(cliente).strip() if cliente is not None else ""
        self._cliente = texto or None

    def limpiar(self) -> None:
        self._cliente = None

    def __call__(self) -> str | None:
        return self._cliente


def _mensajes_de_tarea(tarea: dict) -> list[str]:
    """Mensajes a enviar, en orden: la `secuencia` o la `pregunta` única."""
    secuencia = tarea.get("secuencia") if isinstance(tarea, dict) else None
    if isinstance(secuencia, list) and secuencia:
        return [str(mensaje) for mensaje in secuencia]
    pregunta = tarea.get("pregunta") if isinstance(tarea, dict) else None
    return [str(pregunta or "")]


def _traza_agregada(respuestas: list[Any]) -> dict:
    """Traza del último turno con tokens y latencia sumados de la secuencia.

    El resto de la traza (pasos, plan, error, texto) es la del último turno:
    es el que se puntúa. Lo que se suma es el costo, porque un multi-turno
    gasta más que una pregunta suelta y ocultarlo falsearía el informe.
    """
    ultima = respuestas[-1]
    return {
        "pregunta": str(_leer(ultima, "pregunta", "") or ""),
        "respuesta": _leer(ultima, "respuesta", "") or "",
        "abstencion": bool(_leer(ultima, "abstencion", False)),
        "pasos": _pasos_de(ultima),
        "iteraciones": _leer(ultima, "iteraciones", 0),
        "tokens_total": sum(
            int(_leer(respuesta, "tokens_total", 0) or 0)
            for respuesta in respuestas
        ),
        "latencia_seg": sum(
            float(_leer(respuesta, "latencia_seg", 0.0) or 0.0)
            for respuesta in respuestas
        ),
        "error": _leer(ultima, "error", None),
        "replanes": _leer(ultima, "replanes", 0),
        "plan": _leer(ultima, "plan", None),
    }


def _es_limite_de_tasa(exc: Exception) -> bool:
    """429 del proveedor, sin importarlo: el runner sigue siendo offline-safe."""
    return getattr(exc, "status_code", None) == 429 or (
        type(exc).__name__ == "RateLimitError"
    )


def _traza_es_limite(traza: Any) -> bool:
    """True si la traza terminó por un límite de tasa del proveedor.

    El agente convierte el 429 en un `error` de la respuesta en vez de
    lanzarlo, así que el runner tiene que mirar la traza: si no, la corrida
    publica 0 tokens y una tasa del 50 % como si fuera un resultado.
    """
    texto = str(_leer(traza, "error", "") or "")
    return "RateLimitError" in texto or "429" in texto


def _consultar_secuencia(agente: Any, mensajes: list[str]) -> Any:
    """Consulta cada mensaje en orden y devuelve la traza a puntuar.

    Un turno que lanza no aborta la tarea silenciosamente: se construye la
    traza de error de ESE turno y la secuencia se detiene, porque los mensajes
    siguientes ya no tendrían el contexto que el contrato asumía. La excepción
    es el límite de tasa del proveedor: se propaga para reintentarlo, porque
    convertirlo en un fallo de tarea produce un artefacto falso.
    """
    respuestas: list[Any] = []
    for mensaje in mensajes:
        inicio = time.time()
        try:
            respuestas.append(agente.consultar(mensaje))
        except Exception as exc:  # noqa: BLE001 - se puntúa como fallo
            if _es_limite_de_tasa(exc):
                raise
            respuestas.append(_respuesta_de_error(mensaje, exc, inicio))
            break
    if len(respuestas) == 1:
        return respuestas[0]
    return _traza_agregada(respuestas)


_REINTENTOS_TASA = (5, 15, 30)


def _consultar_con_reintento(agente: Any, mensajes: list[str]) -> Any:
    """Consulta con reintentos acotados ante un límite de tasa.

    El 429 puede llegar como excepción o como `error` en la traza (el agente lo
    captura): se reintentan las dos formas. Si persiste, se propaga y la
    corrida aborta sin escribir resultados: un artefacto con 0 tokens y una
    tasa del 50 % sería peor que no publicar.
    """
    for espera in _REINTENTOS_TASA:
        try:
            traza = _consultar_secuencia(agente, mensajes)
        except Exception as exc:  # noqa: BLE001 - solo se reintenta el 429
            if not _es_limite_de_tasa(exc):
                raise
        else:
            if not _traza_es_limite(traza):
                return traza
        print(f"  ... límite de tasa del proveedor; reintento en {espera}s")
        time.sleep(espera)
    traza = _consultar_secuencia(agente, mensajes)
    if _traza_es_limite(traza):
        raise RuntimeError(
            "Límite de tasa del proveedor (429) persistente. La corrida se "
            "aborta para no publicar un resultado inválido; reintenta cuando "
            "la cuota se restablezca."
        )
    return traza


class _TareasInvalidas(ValueError):
    """Uso inválido de `--tareas`: se reporta como error de uso, no de corrida."""


def _ids_de_tareas(valor: str | None) -> list[str] | None:
    """IDs pedidos en `--tareas`, en el orden en que vienen.

    `None` significa "todas las tareas", que es el comportamiento sin el flag.
    Un valor vacío o un ID vacío es error de uso: `--tareas ""` no puede
    significar "ninguna" ni "todas", y callarlo haría que la corrida midiera
    algo distinto de lo pedido.
    """
    if valor is None:
        return None
    ids = [parte.strip() for parte in str(valor).split(",")]
    if not ids or any(not identificador for identificador in ids):
        raise _TareasInvalidas(
            "--tareas no puede estar vacío. Usa IDs del dataset separados por "
            "coma, por ejemplo: --tareas T16,T17"
        )
    return ids


def _filtrar_tareas(tareas: list[dict], ids: list[str] | None) -> list[dict]:
    """Tareas pedidas, en el orden del dataset.

    El orden lo manda el dataset, no la línea de comandos: dos corridas con los
    mismos IDs son comparables. Un ID desconocido es error de uso, no una tarea
    ignorada: ignorarlo publicaría un resultado que no es el pedido.
    """
    if ids is None:
        return list(tareas)
    conocidos = [str(tarea.get("id")) for tarea in tareas]
    desconocidos = [
        identificador for identificador in ids if identificador not in conocidos
    ]
    if desconocidos:
        raise _TareasInvalidas(
            f"ID de tarea desconocido: {', '.join(desconocidos)}. "
            f"IDs del dataset: {', '.join(conocidos)}."
        )
    pedidos = set(ids)
    return [tarea for tarea in tareas if str(tarea.get("id")) in pedidos]


def _tareas_de_la_corrida(
    datos: dict, ids: list[str] | None, limite: int | None
) -> list[dict]:
    """Selección de tareas de una corrida: primero el filtro, después el tope.

    En ese orden: `--tareas T18 --limite 1` mide T18, no la primera tarea del
    dataset recortada antes de filtrar.
    """
    tareas = _filtrar_tareas(list(datos.get("tareas") or []), ids)
    if limite is not None:
        tareas = tareas[:max(1, limite)]
    return tareas


def _correr(
    datos: dict,
    modos: list[str],
    limite: int | None,
    *,
    ids_tareas: list[str] | None = None,
    solicitudes_por_minuto: float | None = None,
) -> dict:
    """Corre el dataset contra el agente real. Requiere red e índice FAISS.

    Las importaciones son locales: `--verificar-dataset` no debe cargar
    LangChain, FAISS ni el proveedor para validar un JSON.

    El repositorio de casos y la memoria viven en un directorio temporal, uno
    por modo, y se borran al terminar: una escritura aprobada (T15) no puede
    dejar un caso en `data/output/casos.db` ni hacer que dos corridas se
    influyan entre sí.
    """
    # Antes de importar nada con red: `--tareas` inválido es un error de uso y
    # la selección se resuelve sin construir el agente. El filtro va antes que
    # el tope de `--limite`.
    tareas = _tareas_de_la_corrida(datos, ids_tareas, limite)

    from src.agente import AgenteTecnoApoyo
    from src.casos import RepositorioCasos
    from src.herramientas import crear_herramientas
    from src.memoria import Memoria, MemoriaLargoPlazo
    from src.rag import RAGEngine

    motor = RAGEngine()

    corridas = []
    with tempfile.TemporaryDirectory(prefix="evaluar_agente_") as dir_temp:
        # Memoria aislada en un directorio temporal: la corrida no debe dejar
        # hechos ni turnos en la memoria real del usuario.
        memoria = Memoria(
            largo=MemoriaLargoPlazo(ruta=Path(dir_temp) / "memoria.db")
        )
        # Un confirmador por corrida con la política de la tarea vigente: las
        # tools se construyen una sola vez por modo (abajo) y no se reconstruyen
        # en cada tarea.
        confirmador = _ConfirmadorDeCorrida()
        cliente_sesion = _ClienteDeSesion()
        for modo in modos:
            # Repositorio aislado y sembrado desde la semilla versionada, uno
            # por modo. Si los modos compartieran base, el segundo vería el caso
            # que creó el primero y T15 "aprobaría" el efecto sin escribir.
            repositorio = RepositorioCasos(
                ruta=Path(dir_temp) / f"casos_{modo}.db",
                semilla=config.RUTA_CASOS_SEMILLA,
            )
            herramientas = crear_herramientas(
                repositorio, motor, confirmador
            )
            agente = AgenteTecnoApoyo(
                herramientas=herramientas, memoria=memoria, modo=modo,
                solicitudes_por_minuto=solicitudes_por_minuto,
            )
            # Segundo agente, mismo repositorio, motor, memoria y confirmador,
            # con las tools atadas al cliente de la sesión. El cliente se
            # cambia antes de cada tarea (`_ClienteDeSesion`), no se reconstruye
            # el agente: la corrida mide un agente, no uno por tarea.
            #
            # `al_identificar` es el mismo que pasa `src/app.py`
            # (`establecer_cliente`), apuntado al holder mutable en vez de a
            # `config`: sin ese callback la tool `identificar_cliente` ni
            # siquiera se publica, y el catálogo de sesión del evaluador no
            # reproduce el de la consola. Una tarea que pide identificar mediría
            # un artefacto del evaluador, no el comportamiento del producto.
            # El `finally` del bucle limpia el cliente de la tarea, venga de
            # `cliente_sesion` o de la identificación, así que identificarse no
            # filtra el alcance a la tarea siguiente.
            herramientas_sesion = crear_herramientas(
                repositorio, motor, confirmador,
                alcance="sesion", cliente_actual=cliente_sesion,
                al_identificar=cliente_sesion.usar,
            )
            agente_sesion = AgenteTecnoApoyo(
                herramientas=herramientas_sesion, memoria=memoria, modo=modo,
                # El mismo holder que leen las tools: el agente lo consulta en
                # cada turno, así que el prompt del sistema y el del
                # planificador muestran el cliente de la tarea vigente y el
                # atajo determinista de identificación puede dispararse cuando
                # la tarea de sesión no trae cliente.
                cliente_actual=cliente_sesion,
                # La cuota del proveedor es por agente: cada uno tiene su
                # limitador y sus dos modelos lo comparten. Solo corre un modo a
                # la vez, así que no se duplica la tasa efectiva.
                solicitudes_por_minuto=solicitudes_por_minuto,
            )
            puntajes = []
            for posicion, tarea in enumerate(tareas, 1):
                mensajes = _mensajes_de_tarea(tarea)
                pregunta = mensajes[-1]
                # Un solo reinicio de memoria antes del primer mensaje: en una
                # secuencia, limpiar entre mensajes borraría el hilo.
                _preparar_memoria(memoria, tarea)
                confirmador.usar(tarea)
                en_sesion = (
                    _normalizar(tarea.get("alcance") or "global") == "sesion"
                )
                agente_actual = agente_sesion if en_sesion else agente
                cliente_sesion.usar(tarea.get("cliente_sesion"))
                antes = repositorio.contar()
                try:
                    respuesta = _consultar_con_reintento(agente_actual, mensajes)
                finally:
                    # El cliente no sobrevive a su tarea: si quedara fijado, la
                    # tarea siguiente vería casos ajenos sin declararlo.
                    cliente_sesion.limpiar()
                if _traza_es_limite(respuesta):
                    raise RuntimeError(
                        "Límite de tasa del proveedor (429) en la tarea "
                        f"{tarea.get('id')}. La corrida se aborta para no "
                        "publicar un resultado inválido; reintenta cuando la "
                        "cuota se restablezca."
                    )
                despues = repositorio.contar()

                puntaje = puntuar_tarea(
                    tarea,
                    respuesta,
                    escrituras_antes=antes,
                    escrituras_despues=despues,
                    modo=modo,
                    repositorio=repositorio,
                    sin_escritura=antes == despues,
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


def _plan_legible(plan: Any) -> str:
    """Herramientas del plan en una línea, o `-` si la traza no trae plan."""
    if not isinstance(plan, dict):
        return "-"
    nombres = [
        str(paso.get("herramienta") or "-")
        for paso in plan.get("pasos") or []
        if isinstance(paso, dict)
    ]
    return "[" + ", ".join(nombres) + "]" if nombres else "-"


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
            "| Tasa de resolución (efecto cumplido) | "
            f"{_pct(global_.get('tasa_resolucion'))} "
            f"({global_.get('resoluciones_ok', 0)} de "
            f"{global_.get('tareas_con_efecto', 0)} tareas con efecto) |",
            f"| Tasa de plan válido | {_pct_plan(global_)} |",
            f"| Tasa de pedir_datos con la pregunta esperada | "
            f"{_pct(global_.get('tasa_pedir_datos'))} |",
            f"| Tasa de capacidades deterministas | "
            f"{_pct(global_.get('tasa_capacidades'))} |",
            f"| Tasa de multi-turno cumplido | "
            f"{_pct(global_.get('tasa_multiturno'))} |",
            f"| Tasa de no-escritura respetada | "
            f"{_pct(global_.get('tasa_sin_escritura'))} |",
            f"| Promedio de re-planes | {global_['promedio_replanes']} |",
            f"| Tokens totales | {global_['tokens_totales']} |",
            f"| Latencia total (s) | {global_['latencia_total_seg']} |",
            f"| Costo estimado | {global_.get('costo_texto') or COSTO_NO_CONFIGURADO} |",
            "",
            "### Por tipo de tarea",
            "",
            "| Tipo | n | Selección | Argumentos | Abstención | Memoria | Confirmación | Plan | Resolución | Pedir datos | Capacidades | Multi-turno | Sin escritura |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for tipo, metricas in corrida["metricas"]["por_tipo"].items():
            lineas.append(
                f"| {tipo} | {metricas['n_tareas']} | "
                f"{_pct(metricas['tasa_seleccion'])} | "
                f"{_pct(metricas['tasa_argumentos'])} | "
                f"{_pct(metricas['tasa_abstencion'])} | "
                f"{_pct(metricas['tasa_memoria'])} | "
                f"{_pct(metricas['tasa_confirmacion'])} | "
                f"{_pct_plan(metricas)} | "
                f"{_pct(metricas.get('tasa_resolucion'))} | "
                f"{_pct(metricas.get('tasa_pedir_datos'))} | "
                f"{_pct(metricas.get('tasa_capacidades'))} | "
                f"{_pct(metricas.get('tasa_multiturno'))} | "
                f"{_pct(metricas.get('tasa_sin_escritura'))} |"
            )

        fallos = [p for p in corrida["puntajes"] if p.get("motivos")]
        lineas += ["", f"### Tareas con fallos ({len(fallos)})", ""]
        if not fallos:
            lineas.append("Sin fallos: todas las tareas cumplieron su contrato.")
        for puntaje in fallos:
            efecto = puntaje.get("efecto")
            sufijo = (
                f" efecto_ok={puntaje.get('efecto_ok')} "
                f"efecto={efecto}"
                if efecto is not None
                else ""
            )
            lineas.append(
                f"- {puntaje.get('id')} ({puntaje.get('tipo')}): "
                f"{', '.join(puntaje.get('motivos') or [])} — "
                f"herramientas={puntaje.get('herramientas')} "
                f"pasos={puntaje.get('pasos')} "
                f"error={puntaje.get('error')}"
                f" pregunta_pedida={puntaje.get('pregunta_pedida')!r}"
                f" tokens={puntaje.get('tokens')}"
                f" plan={_plan_legible(puntaje.get('plan'))}"
                f"{sufijo}"
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
        "--tareas",
        default=None,
        metavar="IDS",
        help=(
            "Evalúa solo estas tareas, por ID separados por coma (ej. "
            "T16,T17). Conserva el orden del dataset y se combina con "
            "--limite: filtra primero y después aplica el tope. Un ID "
            "desconocido es un error de uso."
        ),
    )
    parser.add_argument(
        "--solicitudes-por-minuto",
        type=float,
        default=0.0,
        metavar="N",
        help=(
            "Limita cada agente a N solicitudes por minuto: el planificador y "
            "el modelo que responde comparten el mismo limitador, con bucket "
            "1. 0 (default) = sin límite."
        ),
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


def _describir_efecto(efecto: dict) -> str:
    """Texto corto del efecto esperado, para la verificación offline."""
    tipo = str(efecto.get("tipo"))
    partes = [
        f"{clave}={efecto[clave]!r}"
        for clave in TIPOS_EFECTO.get(tipo, ())
        if clave in efecto
    ]
    detalle = ", ".join(partes) if partes else "sin claves declaradas"
    return f"{tipo}({detalle})"


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
        print(
            f"  [{tarea['id']}] {tarea['tipo']:22s} "
            f"{_describir_pregunta(tarea)}"
        )
        esperado = tarea.get("esperado") or {}
        etiquetas = _etiquetas_de_esperado(tarea, esperado)
        if etiquetas:
            print(f"       {', '.join(etiquetas)}")
        efecto = esperado.get("efecto")
        if isinstance(efecto, dict):
            # Se imprime el efecto porque es lo que después se verifica contra
            # el repositorio: verlo aquí evita descubrir un typo en la corrida.
            politica = (
                "aprobada" if esperado.get("confirmacion_aprobada") else "denegada"
            )
            print(
                f"       efecto esperado: {_describir_efecto(efecto)} "
                f"| confirmación {politica}"
            )
    return 0


def _describir_pregunta(tarea: dict) -> str:
    """Texto de la tarea para el listado offline.

    Una tarea de `secuencia` no declara `pregunta`: se muestra la secuencia
    completa, porque es lo que se va a enviar al agente.
    """
    pregunta = tarea.get("pregunta")
    if isinstance(pregunta, str) and pregunta.strip():
        return pregunta
    secuencia = tarea.get("secuencia")
    if isinstance(secuencia, list) and secuencia:
        return " -> ".join(str(mensaje) for mensaje in secuencia)
    return "(sin pregunta)"


def _etiquetas_de_esperado(tarea: dict, esperado: dict) -> list[str]:
    """Expectativas nuevas declaradas, para verlas sin abrir el JSON."""
    etiquetas = []
    if isinstance(tarea.get("alcance"), str) and tarea["alcance"] != "global":
        etiquetas.append(f"alcance: {tarea['alcance']}")
    if isinstance(tarea.get("cliente_sesion"), str):
        etiquetas.append(f"cliente: {tarea['cliente_sesion']}")
    if esperado.get("sin_tokens") is True:
        etiquetas.append("sin tokens (capacidades)")
    if esperado.get("sin_escritura") is True:
        etiquetas.append("sin escritura")
    if isinstance(esperado.get("pregunta_clave"), str):
        etiquetas.append(f"pregunta pedida: {esperado['pregunta_clave']!r}")
    if isinstance(esperado.get("menciona"), str):
        etiquetas.append(f"menciona: {esperado['menciona']!r}")
    if _es_entero(esperado.get("replanes_max")):
        etiquetas.append(f"replanes_max: {esperado['replanes_max']}")
    return etiquetas


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

    if args.solicitudes_por_minuto < 0:
        print("[ERROR] --solicitudes-por-minuto no puede ser negativo: 0 ya")
        print("        significa sin límite.")
        return 1

    # S8: la selección se resuelve aquí para que un ID desconocido o un valor
    # vacío se reporten como error de uso sin imprimir el encabezado ni
    # construir el agente. `_correr` vuelve a filtrar porque es ahí donde se
    # combina con `--limite` (filtro primero, tope después).
    try:
        ids_tareas = _ids_de_tareas(args.tareas)
        tareas = _tareas_de_la_corrida(datos, ids_tareas, args.limite)
    except _TareasInvalidas as exc:
        print(f"[ERROR] {exc}")
        return 1

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
          f"tareas: {len(tareas)}/{len(datos['tareas'])} | "
          f"{_texto_tarifas(tarifas)}")
    if tarifas["configurada"]:
        print("Costo estimado sobre tokens totales (sin desglose entrada/salida):")
        print("es una referencia, no una facturación.")
    print("=" * 72)

    try:
        corrida = _correr(
            datos, modos, args.limite,
            ids_tareas=ids_tareas,
            solicitudes_por_minuto=args.solicitudes_por_minuto,
        )
    except _TareasInvalidas as exc:
        print(f"\n[ERROR] {exc}")
        return 1
    except RuntimeError as exc:
        print(f"\n[ERROR] {exc}")
        return 1
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
        "tareas": ids_tareas,
        "solicitudes_por_minuto": args.solicitudes_por_minuto,
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
            f"resolución={_pct(global_.get('tasa_resolucion'))} "
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
