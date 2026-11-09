"""Capacidades del agente: fuente única y respuesta determinista.

La pregunta «¿qué puedes hacer?» no pasa por el modelo. Una tool real viajaría
en el catálogo del prompt (tokens en cada turno) y exigiría una llamada para
ser elegida; acá se responde con un texto canónico, sin modelo y con
`tokens_total = 0`. El paso se publica como `capacidades`, no como una tool del
modelo, porque la traza no puede afirmar una llamada que no existió.

`CAPACIDADES` es la declaración única: la unión de sus `herramientas` debe
cubrir todas las claves de `CLAVES_SALIDA`, y un test lo exige cuando se agrega
una tool nueva.
"""
import re
import unicodedata

# (id, título, descripción, tools que la implementan). Los ids son estables;
# el título y la descripción son el vocabulario que ve el operador.
CAPACIDADES: tuple[tuple[str, str, str, tuple[str, ...]], ...] = (
    (
        "documentacion",
        "Documentación",
        "Políticas, procedimientos, precios, garantías y horarios, citando la "
        "documentación oficial. Si la evidencia no alcanza, se abstiene en vez "
        "de inventar.",
        ("buscar_documentacion",),
    ),
    (
        "casos",
        "Casos",
        "Estado de un caso por su número y listado de los casos del cliente "
        "identificado en la sesión.",
        ("consultar_caso", "listar_casos", "identificar_cliente"),
    ),
    (
        "escrituras",
        "Escrituras",
        "Crear un caso, agregar una nota, cambiar su estado o escalarlo a una "
        "persona; siempre con confirmación humana.",
        ("crear_caso", "agregar_nota", "actualizar_estado", "escalar_a_humano"),
    ),
    (
        "plazos",
        "Plazos",
        "Fecha comprometida en días hábiles, calculada sin adivinar.",
        ("calcular_plazo",),
    ),
    (
        "datos_faltantes",
        "Datos faltantes",
        "Pide el dato que falta en vez de suponerlo.",
        ("pedir_datos",),
    ),
    (
        "memoria",
        "Memoria",
        "Guarda hechos duraderos del usuario cuando la memoria está activa.",
        ("recordar",),
    ),
)

# Texto que recibe el cliente. Se mantiene literal, sin markdown de encabezados
# ni datos operativos: es lo que la consola muestra tal cual.
TEXTO_CAPACIDADES = """Puedo ayudarte con esto:
- Documentación: políticas, procedimientos, precios, garantías y horarios, citando la documentación oficial. Si la evidencia no alcanza, te lo digo en vez de inventar.
- Casos: consultar el estado de un caso por su número (CASO-0000) y listar tus casos.
- Escrituras: crear un caso, agregar una nota, cambiar su estado o escalarlo a una persona; siempre te pido confirmación antes de escribir.
- Plazos: calcular la fecha comprometida en días hábiles.
- Datos faltantes: pedirte el dato que me falte, en vez de suponerlo.
- Memoria: guardar hechos que me pidas recordar, cuando la memoria está activa.

No puedo ver casos de otros clientes ni responder sin respaldo documental."""

# Saludos que pueden preceder la pregunta sin cambiar la intención.
_SALUDOS = (
    "hola", "buenas", "buenos dias", "buenas tardes", "buenas noches",
    "hey", "oye", "disculpa", "por favor",
)

# Formas completas de la pregunta. `que puedes hacer` también se acepta al
# final (\"quiero saber qué puedes hacer\"), pero no en medio: \"qué puedes hacer
# con mi caso\" es una consulta operativa y va al camino normal.
_FRASES_CAPACIDADES = frozenset({
    "que puedes hacer",
    "que sabes hacer",
    "que haces",
    "que servicios ofreces",
    "cuales son tus funciones",
    "cuales son tus capacidades",
    "para que sirves",
    "que puede hacer el agente",
    "que hace el agente",
    "en que me puedes ayudar",
    "en que puedes ayudarme",
    "a que te dedicas",
})


def normalizar(texto: str) -> str:
    """Minúsculas, sin tildes, solo letras y números, espacios colapsados."""
    descompuesto = unicodedata.normalize("NFKD", (texto or "").lower())
    sin_tildes = "".join(
        letra for letra in descompuesto if not unicodedata.combining(letra)
    )
    limpio = "".join(letra if letra.isalnum() else " " for letra in sin_tildes)
    return " ".join(limpio.split())


def _sin_saludo(texto: str) -> str:
    """Quita saludos iniciales, incluso repetidos."""
    partes = texto.split()
    while partes:
        for saludo in _SALUDOS:
            palabras = saludo.split()
            if partes[:len(palabras)] == palabras:
                partes = partes[len(palabras):]
                break
        else:
            return " ".join(partes)
    return ""


# Palabras y frases que acompañan la pregunta sin cambiar la intención. Se
# quitan después de normalizar; "en" o "por" sueltos NO se quitan porque son
# parte de frases del contrato ("en qué me puedes ayudar").
_RELLENOS = (
    "en general", "por favor", "para mi", "para nosotros",
    "exactamente", "realmente", "concretamente", "actualmente",
    "hoy", "ahora", "todas", "todos",
)

# Una prohibición ("no me digas", "no quiero saber") no es una pregunta, pero
# "no sé qué puedes hacer" sí lo es: se comparan prefijos completos, no
# cualquier "no" inicial.
_NEGACIONES = (
    "no me digas", "no quiero saber", "no quiero", "no necesito",
    "no me interesa", "nunca me digas", "jamas me digas",
)


def _sin_relleno(texto: str) -> str:
    """Quita muletillas y acotadores que no cambian la intención."""
    for relleno in _RELLENOS:
        texto = re.sub(rf"\b{re.escape(relleno)}\b", " ", texto)
    return " ".join(texto.split())


def es_consulta_de_capacidades(texto: str) -> bool:
    """Verdadero solo si el turno pregunta qué puede hacer el agente.

    El detector es deliberadamente estricto: una frase operativa que menciona
    \"puedes hacer\" en medio no se intercepta y sigue el camino normal.
    """
    limpio = _sin_relleno(_sin_saludo(normalizar(texto)))
    if not limpio:
        return False
    if any(limpio.startswith(negacion) for negacion in _NEGACIONES):
        return False
    if limpio in _FRASES_CAPACIDADES:
        return True
    return limpio.endswith("que puedes hacer")
