"""Comando de setup: guarda la clave del proveedor en `.env`.

Uso:
    python -m src.configurar_clave

La clave se pide con `getpass` y nunca como argumento: un `--clave VALOR`
quedaría en el historial de la shell y en la lista de procesos. Antes de
escribir se valida contra el endpoint: un 401/403 la rechaza y no se guarda
nada; un problema de red la guarda con advertencia, para no impedir configurar
sin conexión.
"""
import contextlib
import getpass
import os
import sys

# Reconexión UTF-8 en Windows: mismo bloque que src/app.py, para que los
# acentos del mensaje no salgan como "?" en una consola de cp1252.
for flujo in (sys.stdin, sys.stdout, sys.stderr):
    reconectar = getattr(flujo, "reconfigure", None)
    if reconectar is None:
        continue
    with contextlib.suppress(ValueError):  # entornos no reconectables o no tty
        reconectar(encoding="utf-8")

from src import config, credenciales  # noqa: E402 - tras reconectar la consola


def main() -> int:
    if not sys.stdin.isatty():
        print(
            "Este comando pide la clave por consola. Ejecútalo en una terminal.",
            file=sys.stderr,
        )
        return 2

    print(f"Endpoint: {config.LLM_BASE_URL}")
    print(f"Modelo:   {config.LLM_MODELO}")
    try:
        clave = credenciales.normalizar(
            getpass.getpass("Clave del proveedor (no se muestra): ")
        )
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    try:
        estado = credenciales.validar_clave(clave, config.LLM_BASE_URL)
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    if estado == "rechazada":
        print(
            "El endpoint rechazó la clave (401/403). No se guardó nada.",
            file=sys.stderr,
        )
        return 1
    if estado == "sin_red":
        print(
            "Aviso: no se pudo verificar la clave (sin red o endpoint no "
            "disponible). Se guarda igual.",
            file=sys.stderr,
        )

    try:
        # Los argumentos se leen en la llamada, no del default del módulo, para
        # que apuntar la escritura a otra ruta en pruebas funcione de verdad.
        huella = credenciales.guardar_clave(
            clave,
            ruta=credenciales.RUTA_ENV,
            plantilla=credenciales.RUTA_PLANTILLA,
        )
    except OSError as error:
        print(
            f"No se pudo escribir {credenciales.RUTA_ENV}: {error}",
            file=sys.stderr,
        )
        return 1

    print(f"Clave {huella} guardada en {credenciales.RUTA_ENV}.")
    # `load_dotenv` no sobrescribe una variable que ya existe en el entorno: si
    # hay otra distinta, lo guardado no tendría efecto hasta quitarla.
    en_entorno = os.environ.get("LLM_API_KEY", "")
    if en_entorno and en_entorno != clave:
        print(
            "Aviso: LLM_API_KEY ya está definida en el entorno y tiene prioridad "
            "sobre .env. Quítala o actualízala para usar la clave guardada.",
            file=sys.stderr,
        )
    print("Ya puedes ejecutar: python -m src.app")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
