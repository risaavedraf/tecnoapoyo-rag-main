# -*- coding: utf-8 -*-
"""Arranque compartido de la consola y los scripts de TecnoApoyo.

Dos pasos se repetían copiados en cada entrypoint:

  - reconectar `stdin`/`stdout`/`stderr` en UTF-8, porque en Windows la consola
    usa la codepage del sistema y las tildes y los signos de apertura salen
    rotos;
  - cortar el arranque sin `LLM_API_KEY` con la instrucción que la configura, en
    lugar del traceback del proveedor, que no dice qué comando ejecutar.

Este módulo los define una sola vez. Ninguna de las dos funciones corre al
importar: importar un módulo no puede tocar los flujos del proceso que lo
importa ni exigir credenciales. El módulo no importa ningún cliente del
proveedor; `exigir_clave` solo mira la credencial.
"""
import contextlib
import sys

from src import config, credenciales


def reconectar_flujos(encoding: str = "utf-8") -> None:
    """Reconecta los tres flujos del proceso al `encoding` indicado.

    Es seguro llamarla dos veces (y más): reconfigurar un flujo ya reconectado
    no tiene efecto. Se usa `getattr` en lugar de llamar directo porque el tipo
    declarado es `TextIO`, que no expone `reconfigure`, y algunos entornos de
    test entregan flujos que tampoco lo implementan. `ValueError` cubre los
    flujos no reconectables o que no son una tty.
    """
    for flujo in (sys.stdin, sys.stdout, sys.stderr):
        reconectar = getattr(flujo, "reconfigure", None)
        if reconectar is None:
            continue
        with contextlib.suppress(ValueError):  # entornos no reconectables o no tty
            reconectar(encoding=encoding)


def exigir_clave() -> None:
    """Corta el arranque sin clave con la instrucción que la resuelve.

    Se lanza `SystemExit` con el mensaje ya redactado en `credenciales` para que
    la consola y los scripts impriman exactamente lo mismo. No construye ni
    importa ningún cliente: un cliente sin clave falla más tarde, con un error
    del proveedor que no dice cómo configurarla.
    """
    if not config.LLM_API_KEY:
        raise SystemExit(credenciales.MENSAJE_FALTA_CLAVE)
