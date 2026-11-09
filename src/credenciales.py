"""Persistencia de la clave del proveedor en `.env` y validación remota.

La consola construye su cliente HTTP en `preparar_entorno()`, no al importar
`src.app`, así que una clave ausente se descubre al arrancar y no antes. Este
módulo concentra la escritura del archivo
para que el comando de configuración (`python -m src.configurar_clave`) y los
tests compartan una sola implementación, sin arrastrar el motor ni la red.

Decisiones que el código hace explícitas:

- Se reescribe **solo** la línea de la clave: el resto de `.env` (endpoint,
  modelos, comentarios, orden) se conserva tal cual.
- La escritura es atómica: archivo temporal en el mismo directorio + `replace`.
  Si algo falla, el `.env` anterior queda intacto. `mkstemp` crea el temporal
  con permisos 0600 en POSIX y `replace` los conserva.
- La validación remota nunca bloquea salvo que el endpoint **rechace** la
  clave (401/403). Sin red se guarda igual, con advertencia: el usuario puede
  estar configurando antes de conectarse.
"""
import os
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

RAIZ = Path(__file__).resolve().parents[1]
RUTA_ENV = RAIZ / ".env"
RUTA_PLANTILLA = RAIZ / ".env.example"
VARIABLE = "LLM_API_KEY"

# Salida del arranque sin clave. Vive aquí para que la instrucción y el comando
# que la resuelve no se separen con el tiempo.
MENSAJE_FALTA_CLAVE = (
    "No hay LLM_API_KEY configurada: el motor no puede construir el cliente "
    "del proveedor.\n"
    "Guárdala en .env ejecutando:\n"
    "    python -m src.configurar_clave\n"
    "o define LLM_API_KEY en .env (o en tu entorno) con la clave del proveedor."
)


def normalizar(clave: str) -> str:
    """Limpia la entrada del usuario y rechaza lo que no puede ser una clave.

    Un pegado desde el portapapeles suele traer un salto de línea al final; eso
    se descarta. Un espacio interno o un salto en medio no se pueden adivinar:
    se rechazan en vez de guardar una credencial silenciosamente inválida.
    """
    limpia = clave.strip()
    if not limpia:
        raise ValueError("La clave está vacía.")
    if any(caracter.isspace() for caracter in limpia):
        raise ValueError("La clave no puede contener espacios ni saltos de línea.")
    # Una comilla desbalanceada deja el `.env` sin parsear y la variable
    # desaparece en silencio; se rechaza antes de escribir.
    if any(caracter in "\"'" for caracter in limpia):
        raise ValueError("La clave no puede contener comillas.")
    return limpia


def enmascarar(clave: str) -> str:
    """Huella de la clave para mensajes: nunca se imprime completa."""
    if len(clave) <= 8:
        return "*" * len(clave)
    return f"{clave[:4]}…{clave[-4:]}"


def _lineas_base(ruta: Path, plantilla: Path) -> "list[str]":
    """Contenido de partida: el `.env` actual o, si no existe, la plantilla."""
    if ruta.exists():
        return ruta.read_text(encoding="utf-8").splitlines()
    if plantilla.exists():
        return plantilla.read_text(encoding="utf-8").splitlines()
    return []


def _es_la_variable(linea: str) -> bool:
    """True solo para la asignación activa de la variable, no para comentarios."""
    if linea.lstrip().startswith("#"):
        return False
    nombre = linea.split("=", 1)[0].strip()
    return nombre == VARIABLE


def _escribir_atomico(ruta: Path, contenido: str) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporal = tempfile.mkstemp(
        dir=str(ruta.parent), prefix=".env.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as archivo:
            archivo.write(contenido)
        os.replace(temporal, ruta)
    except BaseException:
        Path(temporal).unlink(missing_ok=True)
        raise


def guardar_clave(
    clave: str,
    ruta: Path = RUTA_ENV,
    plantilla: Path = RUTA_PLANTILLA,
) -> str:
    """Guarda la clave en `ruta` y devuelve su huella enmascarada.

    Lanza `ValueError` si la entrada no es una clave y `OSError` si el archivo
    no se puede escribir (en ese caso el original no se modifica).
    """
    limpia = normalizar(clave)
    lineas = _lineas_base(ruta, plantilla)
    nueva = f"{VARIABLE}={limpia}"
    for indice, linea in enumerate(lineas):
        if _es_la_variable(linea):
            lineas[indice] = nueva
            break
    else:
        lineas.append(nueva)
    _escribir_atomico(ruta, "\n".join(lineas) + "\n")
    return enmascarar(limpia)


def validar_clave(clave: str, base_url: str, timeout: float = 10.0) -> str:
    """Consulta `GET {base_url}/models` y clasifica el resultado.

    Devuelve `"ok"` si el endpoint respondió, `"rechazada"` solo cuando la
    clave fue explícitamente rechazada (401/403) y `"sin_red"` para cualquier
    otro fallo (sin red, timeout, endpoint caído). No consume tokens.

    Lanza `ValueError` si `base_url` no es http(s): un endpoint configurado con
    otro esquema no debe llegar a `urlopen`.
    """
    partes = urlparse(base_url)
    if partes.scheme not in ("http", "https"):
        raise ValueError(
            f"LLM_BASE_URL debe usar http o https, no {partes.scheme or 'sin esquema'!r}."
        )
    # Esquema validado arriba: solo http(s) llega aquí.
    peticion = urllib.request.Request(  # noqa: S310
        base_url.rstrip("/") + "/models",
        headers={"Authorization": f"Bearer {clave}"},
    )
    try:
        with urllib.request.urlopen(  # noqa: S310 - esquema restringido a http(s)
            peticion, timeout=timeout
        ) as respuesta:
            return "ok" if respuesta.status < 400 else "sin_red"
    except urllib.error.HTTPError as error:
        return "rechazada" if error.code in (401, 403) else "sin_red"
    except (urllib.error.URLError, OSError, ValueError):
        return "sin_red"
