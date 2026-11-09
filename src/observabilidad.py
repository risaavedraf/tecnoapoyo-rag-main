"""Observabilidad del asistente TecnoApoyo: trazas por turno y agregados (RA2 / IL2.4).

El módulo es de solo archivos: escribe un JSONL por día y lo vuelve a leer. No
llama al LLM, no abre base de datos y no toca la red, de modo que la
observabilidad se puede probar entera con `tmp_path`.

Cuatro decisiones sostienen el contrato:

  1. **El formato no lo inventa este módulo.** El `turno` llega armado por el
     llamador (pregunta, respuesta, abstención, pasos, modo, tokens, latencia,
     error). Aquí no se agregan campos: una traza que miente sobre lo que el
     agente hizo es peor que una traza incompleta.
  2. **Una traza perdida no puede tumbar una respuesta.** `escribir_traza`
     devuelve `None` ante cualquier fallo (directorio no escribible, disco
     lleno, valor no serializable) en vez de propagar la excepción al bucle del
     agente, donde abortaría la conversación por un artefacto accesorio.
  3. **Saneado antes de serializar.** Se descarta, en cualquier nivel de
     anidamiento, toda clave cuyo nombre sea exactamente `token`,
     `access_token`, `bearer` o `authorization`, o cuyo nombre contenga alguna
     de estas marcas: `api_key`, `apikey`, `token_secret`, `password`,
     `secret`. Los contadores de tokens del turno (`tokens`, `tokens_total`) se
     eximen por nombre: son métricas, no credenciales, y `token` es prefijo de
     `tokens_total`. El módulo además no lee `os.environ` en ningún punto, así
     que ningún valor de entorno llega a la traza por esta vía. La lista es
     enumerativa y no una política de redacción: solo se descarta lo que
     nombra, y las variantes que no nombra —`auth`, `credential`, `clave`,
     `api-key`, `X-Api-Key`, `api_token`— pasan enteras a la traza. Quien arma
     el `turno` no puede delegar aquí la decisión de qué es secreto.
  4. **Lectura tolerante.** Un archivo por día, líneas independientes: una
     línea corrupta o una traza incompleta se ignora sin invalidar el resto del
     archivo. Un archivo que no es un objeto JSON tampoco es una traza.

Un archivo por día (`trazas_YYYY-MM-DD.jsonl`) mantiene las escrituras en modo
append y acota el daño de un archivo ilegible; el nombre ordena
cronológicamente, que es lo que usa `leer_trazas` para devolver las últimas
trazas en orden de lectura natural.
"""
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

from src import config

# Prefijo y sufijo del archivo diario. El nombre completo es
# `trazas_YYYY-MM-DD.jsonl` dentro del directorio de trazas.
PREFIJO_TRAZA = "trazas_"
SUFIJO_TRAZA = ".jsonl"

# Claves que nunca se escriben, comparadas en minúsculas y como subcadena del
# nombre: `api_key`, `apikey`, `token_secret`, `password` y `secret` alcanzan
# sus propias variantes (`LLM_API_KEY`, `apiKey`, `db_password`, ...). La lista
# es enumerativa, no una política de redacción: un nombre que no contiene
# ninguna de estas marcas pasa entero, así que no cubre «cualquier
# credencial». El saneado es recursivo porque los argumentos de una herramienta
# pueden anidar un diccionario con credenciales.
CLAVES_SENSIBLES = ("api_key", "apikey", "token_secret", "password", "secret")

# Credenciales cuyo nombre es exactamente uno de estos (S10): una cabecera
# `authorization`, un `bearer`, un `access_token` y el propio `token` no
# contienen ninguna marca de arriba y viajaban enteros a la traza. Se comparan
# por nombre completo porque `token` como subcadena redactaría `tokens_total`.
CLAVES_SENSIBLES_EXACTAS = frozenset({
    "token",
    "access_token",
    "bearer",
    "authorization",
})

# Contadores de tokens del turno. Son métricas, no credenciales: se eximen por
# nombre antes de aplicar las marcas, de modo que `tokens` y `tokens_total`
# sobreviven con su valor numérico aun cuando `token` sea una marca sensible.
CLAVES_CONTEO_TOKENS = ("tokens_total", "tokens")

# Tope de anidamiento al sanear. Un diccionario autorreferente no puede colgar
# la escritura ni reventar con RecursionError: más allá del tope el valor se
# deja tal cual y `json.dumps(default=str)` lo representa.
MAX_PROFUNDIDAD = 20

# Modo que se atribuye a una traza que no declara `modo`.
MODO_DESCONOCIDO = "desconocido"

# Claves donde el llamador puede traer la cuenta de tokens del turno, en orden
# de preferencia (`tokens_total` es el nombre del campo de `RespuestaAgente`).
CLAVES_TOKENS = ("tokens_total", "tokens")

# Claves donde el llamador puede traer la lista de pasos, en orden de
# preferencia. `herramientas` se acepta porque un turno mínimo puede registrar
# solo los nombres.
CLAVES_PASOS = ("pasos", "herramientas")


def _es_clave_sensible(nombre: Any) -> bool:
    """True si el nombre de la clave nombra material sensible (S10).

    Los contadores de tokens se eximen primero: `token` es una marca de
    credencial, pero `tokens` y `tokens_total` son las cuentas del turno y la
    traza las necesita intactas.
    """
    if not isinstance(nombre, str):
        return False
    minusculas = nombre.lower()
    if minusculas in CLAVES_CONTEO_TOKENS:
        return False
    return minusculas in CLAVES_SENSIBLES_EXACTAS or any(
        marca in minusculas for marca in CLAVES_SENSIBLES
    )


def _sanear(valor: Any, profundidad: int = 0) -> Any:
    """Copia `valor` descartando las claves sensibles de todo dict anidado.

    Devuelve estructuras nuevas: el `turno` del llamador no se muta, porque
    quien lo armó puede seguir usándolo (por ejemplo, para imprimirlo) y no
    espera que la traza le borre campos.
    """
    if profundidad > MAX_PROFUNDIDAD:
        return valor
    if isinstance(valor, dict):
        return {
            clave: _sanear(contenido, profundidad + 1)
            for clave, contenido in valor.items()
            if not _es_clave_sensible(clave)
        }
    if isinstance(valor, (list, tuple)):
        return [_sanear(item, profundidad + 1) for item in valor]
    return valor


def _carpeta(ruta: Path | None) -> Path:
    """Directorio de trazas: el que pasa el llamador o el de `config`."""
    return Path(ruta) if ruta is not None else config.RUTA_TRAZAS


def _archivo_del_dia(carpeta: Path) -> Path:
    """Archivo diario de trazas. La fecha es la local del proceso."""
    return carpeta / f"{PREFIJO_TRAZA}{date.today().isoformat()}{SUFIJO_TRAZA}"


def escribir_traza(turno: dict, ruta: Path | None = None) -> Path | None:
    """Agrega una traza al archivo del día y devuelve la ruta escrita.

    `ruta` es el directorio de trazas (no el archivo). Se crea si falta y la
    línea se agrega al final, de modo que dos procesos que escriben el mismo
    día no se pisan: cada línea JSON es independiente.

    Nunca lanza: ante cualquier fallo —directorio imposible de crear, permisos,
    valor no serializable— devuelve `None`. Perder una traza es aceptable;
    tumbar la respuesta que se estaba atendiendo no lo es.
    """
    try:
        carpeta = _carpeta(ruta)
        archivo = _archivo_del_dia(carpeta)
        carpeta.mkdir(parents=True, exist_ok=True)
        # `default=str` cubre latencias o dataclasses que no son JSON nativo;
        # sin él, un valor exótico perdería la traza completa.
        linea = json.dumps(_sanear(turno), ensure_ascii=False, default=str)
        with archivo.open("a", encoding="utf-8") as manejador:
            manejador.write(linea + "\n")
        return archivo
    except Exception:  # noqa: BLE001 - la traza es accesoria, no bloqueante
        return None


def _leer_archivo(archivo: Path) -> list[dict]:
    """Trazas de un archivo; las líneas corruptas y los no-objetos se ignoran.

    `errors="replace"` evita que un byte inválido invalide el resto del
    archivo: la línea rota no parsea y se descarta, las demás se conservan.
    """
    try:
        texto = archivo.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    trazas: list[dict] = []
    for linea in texto.splitlines():
        limpia = linea.strip()
        if not limpia:
            continue
        try:
            traza = json.loads(limpia)
        except (TypeError, ValueError):
            continue
        # Una lista, un número o un `null` no es una traza: el contrato de
        # lectura es de diccionarios.
        if isinstance(traza, dict):
            trazas.append(traza)
    return trazas


def _leer_todas(carpeta: Path) -> list[dict]:
    """Todas las trazas del directorio en orden cronológico de archivo."""
    if not carpeta.is_dir():
        return []
    try:
        archivos = sorted(carpeta.glob(f"{PREFIJO_TRAZA}*{SUFIJO_TRAZA}"))
    except OSError:
        return []
    trazas: list[dict] = []
    for archivo in archivos:
        trazas.extend(_leer_archivo(archivo))
    return trazas


def leer_trazas(limite: int = 10, ruta: Path | None = None) -> list[dict]:
    """Últimas `limite` trazas, de la más antigua a la más reciente.

    `ruta` es el directorio de trazas. Un directorio inexistente, vacío o sin
    archivos legibles devuelve `[]`. Un `limite` <= 0 también devuelve `[]`: no
    hay "últimas cero trazas" que devolver, y pedirlas todas (ignorar el
    parámetro) sería lo contrario de lo que pidió el llamador.
    """
    tope = int(limite)
    if tope <= 0:
        return []
    trazas = _leer_todas(_carpeta(ruta))
    return trazas[-tope:]


def _pasos_de(traza: dict) -> list:
    """Lista de pasos de una traza, venga como `pasos` o como `herramientas`."""
    for clave in CLAVES_PASOS:
        valor = traza.get(clave)
        if isinstance(valor, list):
            return valor
    return []


def _nombre_herramienta(paso: Any) -> str:
    """Nombre de herramienta de un paso; "" si el paso no lo declara."""
    if isinstance(paso, str):
        return paso.strip()
    if isinstance(paso, dict):
        for clave in ("herramienta", "nombre", "tool"):
            valor = paso.get(clave)
            if isinstance(valor, str) and valor.strip():
                return valor.strip()
    return ""


def _tokens_de(traza: dict) -> int:
    """Tokens del turno; 0 si no los declara o si el valor no es numérico."""
    for clave in CLAVES_TOKENS:
        valor = traza.get(clave)
        # `bool` es subclase de `int`: un True accidental no son 1 token.
        if isinstance(valor, bool):
            return 0
        if isinstance(valor, (int, float)):
            return int(valor)
        if isinstance(valor, str) and valor.strip().isdigit():
            return int(valor.strip())
    return 0


def _tiene_error(traza: dict) -> bool:
    """True si la traza o alguno de sus pasos trae un error explícito."""
    if str(traza.get("error") or "").strip():
        return True
    return any(
        isinstance(paso, dict) and str(paso.get("error") or "").strip()
        for paso in _pasos_de(traza)
    )


def resumen_stats(ruta: Path | None = None) -> dict:
    """Agregados de todas las trazas del directorio (sin red, solo lectura).

    Lee el directorio completo, no las últimas `leer_trazas` trazas: un
    resumen acotado a 10 filas cambiaría en silencio el significado de
    `consultas` en cuanto el directorio creciera.

    Claves devueltas, todas derivadas de lo leído:

      - `consultas`: cantidad de trazas (una por turno).
      - `abstenciones`: trazas con `abstencion` verdadero.
      - `pasos`: total de pasos ejecutados, sumando los de cada traza.
      - `por_herramienta`: conteo de pasos por nombre de herramienta. Un paso
        sin nombre elegible suma en `pasos` pero no aquí: no hay a quién
        atribuirlo.
      - `por_modo`: conteo de trazas por `modo`; una traza sin modo cae en
        `desconocido` en vez de desaparecer del total.
      - `tokens`: suma de `tokens_total` o `tokens`; un valor ausente o no
        numérico aporta 0 en vez de romper la suma.
      - `error`: trazas con error en el turno o en algún paso.
    """
    trazas = _leer_todas(_carpeta(ruta))
    por_herramienta: dict[str, int] = {}
    por_modo: dict[str, int] = {}
    stats = {
        "consultas": len(trazas),
        "abstenciones": 0,
        "pasos": 0,
        "por_herramienta": por_herramienta,
        "por_modo": por_modo,
        "tokens": 0,
        "error": 0,
    }
    for traza in trazas:
        if bool(traza.get("abstencion")):
            stats["abstenciones"] += 1

        pasos = _pasos_de(traza)
        stats["pasos"] += len(pasos)
        for paso in pasos:
            nombre = _nombre_herramienta(paso)
            if nombre:
                por_herramienta[nombre] = por_herramienta.get(nombre, 0) + 1

        modo = str(traza.get("modo") or "").strip().lower() or MODO_DESCONOCIDO
        por_modo[modo] = por_modo.get(modo, 0) + 1

        stats["tokens"] += _tokens_de(traza)

        if _tiene_error(traza):
            stats["error"] += 1
    return stats


# --- Extracción para la consola y armado de la traza ---------------------

def _campo(objeto: Any, nombre: str, defecto: Any = None) -> Any:
    """Lee un campo de un dataclass o de un dict sin exigir una sola forma."""
    if isinstance(objeto, dict):
        return objeto.get(nombre, defecto)
    return getattr(objeto, nombre, defecto)


def fuentes_de(pasos: Any) -> list[dict[str, Any]]:
    """Fuentes documentales deduplicadas a partir de los pasos ejecutados.

    Reemplaza el volcado de JSON en consola: una respuesta fundamentada debe
    poder citar documento, título y relevancia. Cuando el mismo documento
    aparece en varias búsquedas se conserva el mejor score.
    """
    mejores: dict[str, dict[str, Any]] = {}
    for paso in pasos or []:
        if str(_campo(paso, "herramienta", "")) != "buscar_documentacion":
            continue
        if not _campo(paso, "ok", False):
            continue
        salida = _campo(paso, "salida")
        if not isinstance(salida, str):
            continue
        try:
            payload = json.loads(salida)
        except (TypeError, ValueError):
            continue
        datos = payload.get("datos") if isinstance(payload, dict) else None
        fragmentos = datos.get("fragmentos") if isinstance(datos, dict) else None
        for fragmento in fragmentos or []:
            if not isinstance(fragmento, dict):
                continue
            document_id = str(fragmento.get("document_id", "?"))
            anterior = mejores.get(document_id)
            if anterior is None or float(fragmento.get("score", 0)) > float(
                anterior.get("score", 0)
            ):
                mejores[document_id] = fragmento
    return sorted(mejores.values(),
                  key=lambda fuente: float(fuente.get("score", 0)), reverse=True)


def traza_de_turno(respuesta: Any, modo: str = "",
                   estrategia_memoria: str = "",
                   hechos_inyectados: int = 0) -> dict[str, Any]:
    """Arma el turno que se persiste a partir de una respuesta del agente.

    No calcula métricas nuevas: reubica lo que el agente ya midió (pasos,
    tokens, latencia, abstención) y añade el contexto de ejecución que solo la
    consola conoce (modo y estrategia de memoria).
    """
    pasos = []
    for paso in _campo(respuesta, "pasos", []) or []:
        pasos.append({
            "herramienta": str(_campo(paso, "herramienta", "")),
            "argumentos": _campo(paso, "argumentos", {}) or {},
            "ok": bool(_campo(paso, "ok", False)),
            "error": _campo(paso, "error"),
            "duracion_seg": float(_campo(paso, "duracion_seg", 0.0) or 0.0),
        })
    plan = _campo(respuesta, "plan")
    if hasattr(plan, "model_dump"):
        plan = plan.model_dump()
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "modo": modo,
        "pregunta": _campo(respuesta, "pregunta", ""),
        "respuesta": _campo(respuesta, "respuesta", ""),
        "plan": plan,
        "pensamientos": list(_campo(respuesta, "pensamientos", []) or []),
        "pasos": pasos,
        "replanes": int(_campo(respuesta, "replanes", 0) or 0),
        "presupuesto_usado": len(pasos),
        "tokens": int(_campo(respuesta, "tokens_total", 0) or 0),
        "latencia_seg": float(_campo(respuesta, "latencia_seg", 0.0) or 0.0),
        "abstencion": bool(_campo(respuesta, "abstencion", False)),
        "error": _campo(respuesta, "error"),
        "estrategia_memoria": estrategia_memoria,
        "hechos_inyectados": int(hechos_inyectados),
    }
