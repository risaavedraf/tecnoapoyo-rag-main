# -*- coding: utf-8 -*-
"""Puente determinista entre el arnés y el usuario para pedidos de escritura.

El modo reactivo resolvía todo pedido de escritura con el modelo en el medio:
él decidía si la tarea era una escritura, elegía `crear_caso` y redactaba la
propuesta. Con un modelo pequeño eso falla de dos maneras medidas: inventa los
datos obligatorios que el texto no trae, o —lo que se midió en T21— se niega a
la tarea completa y el turno no propone ni pregunta nada.

Este módulo le quita al modelo esa parte del trabajo y no le quita ninguna
garantía: detecta el pedido con una lista declarada, recolecta del texto los
campos obligatorios de `crear_caso` y dice cuáles faltan. El agente decide con
esa información **antes** de invocar al modelo: si falta un campo, la pregunta
sale de acá y el modelo no se llama; si están todos, la llamada se arma con los
campos recolectados y se ejecuta por el camino de siempre. La escritura sigue
siendo una propuesta: la guarda de confirmación de `src.herramientas` es el
único paso hacia un efecto real, y este módulo no la toca.

El módulo es puro: no importa LangChain ni ningún proveedor, así que se puede
probar y razonar sin red. Solo depende del plegado de tildes compartido.

Qué NO cubre esta frontera —y por qué no capturar en esos casos:
  - Sinónimos no enumerados (`genera`, `añade`, `carga`, `crea un ticket`). La
    lista de `PATRONES_DE_CREACION` es enumerativa y cerrada: lo que no está no
    se captura y el turno sigue por el modelo, que es el camino de siempre.
  - Otros idiomas (`create a case`). La lista está redactada en español.
  - Pedidos de lectura (`¿qué casos tengo?`, `¿en qué estado está el caso?`):
    aunque nombren un verbo de la lista, `INDICIOS_DE_LECTURA` los devuelve al
    modelo. Es una guarda deliberadamente amplia: ante la duda, no se captura.
  - Preguntas de procedimiento (`¿cómo creo un caso?`): preguntan por el
    trámite, no piden crear nada, y se responden por el camino documental.
  - Turnos con más de una acción (`registra un caso y agrégale una nota`,
    `abre un caso y escálalo`). `INDICIOS_DE_OTRA_ACCION` los devuelve al
    modelo: el puente solo sabe crear el caso, y atender únicamente esa mitad
    dejaría la nota o la escalada sin ejecutar, en silencio.
  - Campos obligatorios sin marca explícita. `categoria` y `descripcion` solo
    se recolectan con una etiqueta (`categoría recepcion`, `descripción: …`)
    porque leerlos de la prosa sería adivinar. Un campo que el texto no trae se
    trata como faltante y se pregunta: nunca se rellena.
  - Los campos opcionales (`prioridad`, `procedimiento`). Sus etiquetas solo
    cierran el valor del campo anterior; no se recolectan, así que el puente no
    inventa una prioridad ni un PROC-* que el texto no declare.
"""
import re
from dataclasses import dataclass

from src.texto import sin_tildes

# Pregunta canónica de identificación. Vive acá —y no en `src.agente`— porque
# este módulo es puro y también la usa como pregunta del campo `cliente`:
# `src.agente` la importa y la reexporta, así que el atajo de identificación y
# el puente piden el nombre de una sola forma.
PREGUNTA_CLIENTE = "¿Cuál es su nombre completo para identificar sus casos?"

# Campos obligatorios de `crear_caso`, en el orden en que los declara
# `ArgsCrearCaso` (`src/herramientas.py`). Un test compara esta tupla contra el
# `args_schema` real: si la herramienta suma o reordena un campo requerido, el
# puente no se queda con una lista vieja. Los opcionales (`prioridad`,
# `procedimiento`) no están acá a propósito.
CAMPOS_OBLIGATORIOS_CREAR_CASO: tuple[str, ...] = (
    "cliente",
    "categoria",
    "descripcion",
)

# Pregunta determinista por campo. La del cliente es la canónica del asistente;
# la de la categoría reutiliza el vocabulario del contrato de la herramienta
# («categoría de soporte, por ejemplo recepcion»).
PREGUNTA_POR_CAMPO: dict[str, str] = {
    "cliente": PREGUNTA_CLIENTE,
    "categoria": (
        "¿Qué categoría de soporte corresponde al caso? Por ejemplo: recepcion."
    ),
    "descripcion": (
        "¿Cuál es la descripción del caso? Indique el equipo y la falla que "
        "presenta."
    ),
}

# Verbos de escritura sobre un caso, con sus formas imperativas. Las claves son
# el verbo canónico y los valores las formas que se reconocen en el texto,
# declaradas sin tildes porque la comparación normaliza. La lista es cerrada:
# un verbo que no esté acá no se captura (ver el docstring del módulo). Las
# formas con pronombre enclítico (`créame`, `regístrame`) están incluidas
# porque son la forma natural de pedir un caso.
PATRONES_DE_CREACION: dict[str, tuple[str, ...]] = {
    "crear": ("crea", "creame"),
    "registrar": ("registra", "registrame"),
    "abrir": ("abre", "abreme"),
    "ingresar": ("ingresa", "ingresame"),
    "dar_de_alta": ("da de alta", "dale de alta"),
}

# Infinitivos: no son una orden por sí solos («crear un caso» no pide nada),
# así que solo cuentan detrás de un indicio de pedido.
FORMAS_INFINITIVAS: tuple[str, ...] = (
    "crear", "registrar", "abrir", "ingresar", "dar de alta",
)
INDICIOS_DE_PEDIDO = frozenset({
    "quiero", "necesito", "quisiera", "puedes", "podrias", "favor",
    "ayudame",
})

# Sustantivo que convierte el verbo en un pedido sobre un caso. Sin él,
# «abre la documentación» no es un pedido de escritura de casos.
SUSTANTIVOS_CASO = frozenset({"caso", "casos"})

# Indicios de lectura. Si el turno trae uno, el puente no captura aunque
# encuentre un verbo de escritura: una consulta no puede terminar en una
# pregunta por datos de un caso que el usuario no pidió crear.
INDICIOS_DE_LECTURA = frozenset({
    "listar", "listado", "lista", "listame", "listan",
    "muestra", "muestrame", "mostrar", "ver",
    "consulta", "consultar", "revisar",
    "estado", "situacion", "detalle", "seguimiento",
    "cuantos", "cuantas", "cuales", "tengo", "hay",
    "abierto", "abiertos", "pendiente", "pendientes",
})

# Palabras que convierten el turno en una pregunta de procedimiento, no en un
# pedido. «¿Cómo creo un caso?» se responde con documentación.
PALABRAS_DE_PROCEDIMIENTO = frozenset({"como", "cuando", "donde"})

# Indicios de que el turno pide, además de crear el caso, otra escritura (una
# nota, un cambio de estado, una escalada). El puente solo sabe crear: si
# capturara el turno, la otra mitad del pedido se perdería sin aviso. Ante esa
# duda el turno vuelve al modelo, que sí puede encadenar las herramientas.
INDICIOS_DE_OTRA_ACCION = frozenset({
    "nota", "notas", "agrega", "agregale", "agregar", "agregas",
    "anade", "anadele", "anadir", "actualiza", "actualizale", "actualizar",
    "cambia", "cambiale", "cambiar", "escala", "escalalo", "escalala",
    "escalar", "escalado", "escalada",
})

# Palabras del dominio que, capitalizadas, no son el nombre de un cliente
# («para Recepción», «para Nuevo Caso»). Es la guarda de confianza del puente,
# no la del guard de identidad de `src.agente`: ahí se decide si alguien
# declara un nombre, acá si un texto capitalizado sirve como `cliente`.
PALABRAS_QUE_NO_SON_CLIENTE = frozenset({
    "caso", "casos", "cliente", "clientes", "soporte", "recepcion",
    "reparacion", "diagnostico", "presupuesto", "garantia", "mantenimiento",
    "respaldo", "tecnoapoyo", "alta", "nuevo", "nueva", "urgente",
    "prioridad", "categoria", "descripcion",
})

# Topes de los campos, copiados del `args_schema` de `crear_caso`. Un valor
# fuera de rango no se recorta en silencio: se trata como faltante y se
# pregunta, porque recortar cambiaría el dato que la persona escribió.
LARGO_MINIMO = {"cliente": 3, "categoria": 3, "descripcion": 5}
LARGO_MAXIMO = {"cliente": 120, "categoria": 60, "descripcion": 1000}

_NO_ALFANUMERICO = re.compile(r"[^a-z0-9]+")

# Etiquetas que el texto puede usar para declarar un campo. `prioridad` y
# `procedimiento` están para cerrar el valor del campo anterior; no se
# recolectan.
_ETIQUETA = re.compile(
    r"\b(categor[ií]a|descripci[oó]n|cliente|prioridad|procedimiento)\s*:?\s*",
    re.IGNORECASE,
)

# Introducen el nombre del cliente. «de» no está: «un caso de impresora»
# convertiría una descripción en un cliente. Ante la duda, el campo falta y se
# pregunta.
_INTRODUCTOR_CLIENTE = re.compile(
    r"\b(?:a\s+nombre\s+de|para|cliente)\s+(?:el\s+|la\s+)?",
    re.IGNORECASE,
)

# Una palabra capitalizada: el nombre de una persona, no un id ni una sigla.
_PALABRA_DE_NOMBRE = r"[A-ZÁÉÍÓÚÑÜ][a-záéíóúñü'’\-]*"
_NOMBRE_AL_INICIO = re.compile(
    rf"{_PALABRA_DE_NOMBRE}(?:\s+{_PALABRA_DE_NOMBRE}){{0,3}}"
)

_COMILLAS_Y_SEPARADORES = "\"'«»“”,;.:"


def _normalizar(texto: str) -> str:
    """Minúsculas, sin tildes y con los separadores colapsados en un espacio."""
    return " ".join(_NO_ALFANUMERICO.sub(" ", sin_tildes(texto.lower())).split())


def _limpiar(valor: str) -> str:
    return valor.strip().strip(_COMILLAS_Y_SEPARADORES).strip()


def _nombre_al_inicio(valor: str) -> str | None:
    """Nombre propio con el que empieza `valor`, o None si no lo hay.

    Acepta de una a cuatro palabras capitalizadas. La primera no puede ser una
    palabra del dominio —«para Recepción» no nombra a nadie— y el resultado
    tiene que caber en el contrato de `crear_caso`.
    """
    coincidencia = _NOMBRE_AL_INICIO.match(valor.strip())
    if coincidencia is None:
        return None
    nombre = " ".join(coincidencia.group(0).split())
    if not (LARGO_MINIMO["cliente"] <= len(nombre) <= LARGO_MAXIMO["cliente"]):
        return None
    if _normalizar(nombre.split()[0]) in PALABRAS_QUE_NO_SON_CLIENTE:
        return None
    return nombre


def _valores_etiquetados(texto: str) -> dict[str, str]:
    """Valor crudo de cada campo que el texto declara con etiqueta.

    El valor de una etiqueta llega hasta la etiqueta siguiente o el final del
    texto. Se conserva crudo: cada campo lo limpia según su contrato.
    """
    coincidencias = list(_ETIQUETA.finditer(texto))
    valores: dict[str, str] = {}
    for posicion, coincidencia in enumerate(coincidencias):
        campo = _normalizar(coincidencia.group(1))
        inicio = coincidencia.end()
        if posicion + 1 < len(coincidencias):
            fin = coincidencias[posicion + 1].start()
        else:
            fin = len(texto)
        valor = _limpiar(texto[inicio:fin])
        if valor and campo not in valores:
            valores[campo] = valor
    return valores


def _categoria_de(valor: str) -> str | None:
    """Categoría declarada con etiqueta, o None si no hay certeza.

    La categoría del corpus es una sola palabra (`recepcion`, `respaldo_datos`).
    Un valor con más palabras —«de reparación», «recepción por teléfono»— no es
    un dato, es prosa: se trata como faltante.
    """
    limpio = _limpiar(valor)
    if not limpio or len(limpio.split()) != 1:
        return None
    if not (LARGO_MINIMO["categoria"] <= len(limpio) <= LARGO_MAXIMO["categoria"]):
        return None
    return limpio


def _descripcion_de(valor: str) -> str | None:
    limpio = _limpiar(valor)
    if not (LARGO_MINIMO["descripcion"] <= len(limpio)
            <= LARGO_MAXIMO["descripcion"]):
        return None
    return limpio


def _cliente_de(texto: str, valores: dict[str, str]) -> str | None:
    """Cliente del pedido: etiqueta `cliente`, o un nombre detrás de un introductor.

    Se prueban los dos caminos en orden. Nada se completa si ninguno da un
    nombre con forma de nombre: el campo queda faltante y el arnés lo pregunta.
    """
    etiquetado = valores.get("cliente")
    if etiquetado:
        nombre = _nombre_al_inicio(etiquetado)
        if nombre is not None:
            return nombre
    for coincidencia in _INTRODUCTOR_CLIENTE.finditer(texto):
        nombre = _nombre_al_inicio(texto[coincidencia.end():])
        if nombre is not None:
            return nombre
    return None


def recolectar_campos(texto: str) -> dict[str, str]:
    """Campos obligatorios de `crear_caso` que el texto trae, y solo esos.

    No usa valores por defecto ni deriva un campo de otro: lo que no está no
    aparece en el diccionario y el llamador lo trata como faltante.
    """
    valores = _valores_etiquetados(texto or "")
    campos: dict[str, str] = {}
    cliente = _cliente_de(texto or "", valores)
    if cliente is not None:
        campos["cliente"] = cliente
    categoria = _categoria_de(valores.get("categoria", ""))
    if categoria is not None:
        campos["categoria"] = categoria
    descripcion = _descripcion_de(valores.get("descripcion", ""))
    if descripcion is not None:
        campos["descripcion"] = descripcion
    return campos


def campos_faltantes(campos: dict[str, str]) -> tuple[str, ...]:
    """Campos obligatorios que no están, en el orden del contrato."""
    return tuple(
        campo for campo in CAMPOS_OBLIGATORIOS_CREAR_CASO if campo not in campos
    )


def pregunta_por_el_primero_que_falta(campos: dict[str, str]) -> str:
    """Pregunta determinista por el primer campo obligatorio que falta, o ""."""
    faltantes = campos_faltantes(campos)
    return PREGUNTA_POR_CAMPO[faltantes[0]] if faltantes else ""


def _hay_patron_de_escritura(normalizado: str, palabras: set[str]) -> bool:
    """Verdadero si el texto trae una forma imperativa declarada."""
    texto = f" {normalizado} "
    for formas in PATRONES_DE_CREACION.values():
        for forma in formas:
            if " " in forma:
                if f" {forma} " in texto:
                    return True
            elif forma in palabras:
                return True
    return False


def _hay_infinitivo_con_pedido(normalizado: str, palabras: set[str]) -> bool:
    """Verdadero si un infinitivo declarado va detrás de un indicio de pedido."""
    if not palabras & INDICIOS_DE_PEDIDO:
        return False
    texto = f" {normalizado} "
    return any(f" {forma} " in texto for forma in FORMAS_INFINITIVAS)


@dataclass(frozen=True)
class PedidoDeCrearCaso:
    """Pedido de creación detectado: lo que el texto trae y lo que falta."""

    campos: dict[str, str]
    faltantes: tuple[str, ...]

    @property
    def completo(self) -> bool:
        return not self.faltantes

    @property
    def argumentos(self) -> dict[str, str]:
        """Argumentos de `crear_caso`: solo los obligatorios recolectados.

        No agrega `prioridad` —la herramienta tiene su propio valor por
        defecto— ni `procedimiento`, y tampoco claves de más: el `args_schema`
        declara `extra="forbid"`.
        """
        return dict(self.campos)


def detectar_pedido_de_crear_caso(texto: str) -> PedidoDeCrearCaso | None:
    """Pedido de creación de caso detectado, o None si el turno no lo es.

    Devuelve None en todos los casos dudosos: pedidos de lectura, preguntas de
    procedimiento, verbos u objetos fuera de la lista declarada, y turnos sin
    la forma verbo + «caso». El llamador interpreta None como «este turno no es
    mío» y el modelo lo recibe tal como llegó.
    """
    if not isinstance(texto, str) or not texto.strip():
        return None
    normalizado = _normalizar(texto)
    palabras = set(normalizado.split())
    if not palabras & SUSTANTIVOS_CASO:
        return None
    # Ante la duda, no se captura: una consulta o una pregunta de trámite no
    # puede terminar en una pregunta por los datos de un caso.
    if palabras & INDICIOS_DE_LECTURA:
        return None
    if palabras & PALABRAS_DE_PROCEDIMIENTO:
        return None
    if palabras & INDICIOS_DE_OTRA_ACCION:
        return None
    if not (
        _hay_patron_de_escritura(normalizado, palabras)
        or _hay_infinitivo_con_pedido(normalizado, palabras)
    ):
        return None
    campos = recolectar_campos(texto)
    return PedidoDeCrearCaso(campos=campos, faltantes=campos_faltantes(campos))
