"""Plegado de tildes compartido por todo el proyecto.

Siete módulos comparan textos sin tildes: `src.capacidades` (la pregunta de
capacidades), `src.herramientas` y `src.casos` (nombres de cliente),
`src.memoria` (tokens de la consulta y de los hechos), `src.rag` (alias de
productos fuera de alcance), `src.seleccion` (iniciales del menú) y
`scripts.evaluar_agente` (comparación de la traza). Cada uno tenía su copia del
mismo paso —descomponer el texto y descartar las marcas combinantes— con dos
variantes que podían discrepar sin que nadie lo notara: `NFD` en dos copias y
`NFKD` en las otras cinco. Este módulo es la única definición del paso, así que
un arreglo acá llega a los siete llamadores a la vez.

Decisiones de diseño:
  - La forma es `NFKD`. Es la que ya usaban cinco de los siete llamadores y
    pliega además los caracteres de compatibilidad (`ﬁ` → `fi`, `①` → `1`).
    Para los caracteres acentuados del español las dos formas coinciden, y en
    los datos del proyecto —nombres de la semilla y preguntas del corpus— no
    hay un solo carácter donde difieran.
  - El plegado de mayúsculas no vive acá. Tres llamadores usan `casefold` y
    cuatro `lower`, y no son lo mismo: `"Straße".casefold()` es `"strasse"` y
    `"Straße".lower()` es `"straße"`. Forzar una de las dos cambiaría el
    resultado de la otra mitad, así que cada llamador conserva la suya.
  - El colapso de espacios tampoco vive acá: solo tres llamadores lo hacen, y
    con criterios distintos (una regex, `split`, `strip`). El paso compartido
    es el de las tildes y nada más; el resto es política de cada llamador.
  - La función no decide qué es un texto vacío: no reemplaza `None` por `""`.
    Los llamadores que hoy toleran `None` conservan su `(texto or "")`, y los
    que hoy fallan con `None` siguen fallando igual.
"""
import unicodedata


def sin_tildes(texto: str) -> str:
    """Devuelve `texto` sin las marcas combinantes (tildes, diéresis, virgulilla).

    Cambia solo eso: mayúsculas, espacios y signos pasan tal como vienen, y una
    letra sin tilde no se toca. El llamador decide cuándo plegar mayúsculas y
    cuándo colapsar espacios.
    """
    descompuesto = unicodedata.normalize("NFKD", texto)
    return "".join(
        caracter for caracter in descompuesto
        if not unicodedata.combining(caracter)
    )
