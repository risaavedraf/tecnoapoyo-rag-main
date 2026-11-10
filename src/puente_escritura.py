# -*- coding: utf-8 -*-
"""Puente determinista entre el arnés y el usuario para pedidos de escritura.

El reparto de responsabilidades de este módulo es uno solo (S25), y es el que
ordena todo lo demás:

  - El arnés hace la MECÁNICA: detecta el pedido de creación, extrae del texto
    los datos que solo la persona puede aportar y pregunta determinísticamente
    lo que el texto no trae, sin llamar al modelo.
  - El modelo hace el JUICIO: elige la **categoría** del caso y emite la
    propuesta. El puente no la elige, no la supone, no la completa y no lleva
    ningún valor por defecto: `ArgsCrearCaso` la sigue exigiendo
    (`src/herramientas.py`), porque un campo obligatorio que pasa a opcional deja
    un hueco por el que un llamador omite el dato y el sistema lo rellena en
    silencio.

Cómo se reparte el turno: si falta un dato del usuario, la pregunta sale de
acá y el modelo no se invoca. Cuando los datos están completos, el arnés le
entrega al modelo esos datos ya recolectados con una instrucción estrecha
(`instruccion_para_el_modelo`): no extraer, no preguntar, no elegir la
herramienta —es `crear_caso`— y decidir solo la categoría. La propuesta del
modelo entra por el mismo camino de ejecución y confirmación de siempre: la
guarda de `src.herramientas` sigue siendo el único paso hacia un efecto real, y
este módulo no la toca.

Datos que el puente extrae, con su forma declarada:
  - `cliente`: el nombre detrás de «el cliente <Nombre>», «la clienta <Nombre>»,
    «para <Nombre>» o «a nombre de <Nombre>», o el valor de la etiqueta
    `cliente:`. Tiene que empezar con mayúscula, no puede ser una palabra del
    dominio («para Recepción») y tiene que caber en el contrato (3-120).
  - `descripcion`: la falla enunciada en prosa detrás de «por» («por un equipo
    que se apaga solo»), de «porque», de «:» o de la coma que abre la cláusula
    siguiente al nombre (o el valor de la etiqueta `descripción:`). Se lee desde
    el final del nombre del cliente, o en todo el mensaje cuando el nombre no
    está; solo se acepta si cumple el contrato de `descripcion` (5-1000).

Qué NO se extrae, y por qué:
  - `categoria`, nunca. Ni con etiqueta (`categoría <valor>`): aunque el texto
    la declare, el puente no la lee. Es la decisión del modelo y el campo solo
    viaja en la llamada que emite el modelo.
  - Instrucciones sobre la confirmación o la autoridad del turno («sin pedirme
    confirmación», «no me pidas confirmación», «sin autorización»). Acotan el
    candidato de descripción; no son la falla del equipo ni se propagan.
  - Cortesías y muletillas («por favor», «gracias»): no son una descripción.
  - `prioridad` y `procedimiento`: son opcionales en el esquema, así que el
    puente no los fabrica. Sus etiquetas solo cierran el valor anterior.
  - Un fragmento dudoso de cualquiera de los dos campos: si no tiene la forma
    declarada o no cabe en el contrato, el campo se trata como faltante y se
    pregunta. Nunca se rellena ni se recorta.

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

El módulo es puro: no importa LangChain ni ningún proveedor, así que se puede
probar y razonar sin red. Solo depende del plegado de tildes compartido.
"""
import re
from dataclasses import dataclass

from src.texto import sin_tildes

# Pregunta canónica de identificación. Vive acá —y no en `src.agente`— porque
# este módulo es puro y también la usa como pregunta del campo `cliente`:
# `src.agente` la importa y la reexporta, así que el atajo de identificación y
# el puente piden el nombre de una sola forma.
PREGUNTA_CLIENTE = "¿Cuál es su nombre completo para identificar sus casos?"

# Campos que el puente completa: los dos datos que solo aporta la persona, en el
# orden en que los declara `ArgsCrearCaso`. `categoria` no está y no puede
# estar: es la única decisión que conserva el modelo, así que no se recolecta,
# no se pregunta y no se completa con ningún valor. Los opcionales (`prioridad`,
# `procedimiento`) tampoco: tienen su propio default en la herramienta.
CAMPOS_DEL_PUENTE: tuple[str, ...] = ("cliente", "descripcion")

# Pregunta determinista por campo. La del cliente es la canónica del asistente;
# la de la descripción pide el equipo y la falla, que es lo que el contrato
# necesita. No hay pregunta por la categoría: no es un dato del usuario.
PREGUNTA_POR_CAMPO: dict[str, str] = {
    "cliente": PREGUNTA_CLIENTE,
    "descripcion": (
        "¿Cuál es la descripción del caso? Indique el equipo y la falla que "
        "presenta."
    ),
}

# Herramienta que el modelo debe emitir cuando el arnés ya tiene los datos. La
# nombra la instrucción de entrega: elegir la herramienta no es una decisión que
# quede abierta en ese turno.
HERRAMIENTA_CREAR_CASO = "crear_caso"

# Instrucción estrecha que acompaña al turno entregado al modelo (S25). Los
# datos van tal como los escribió la persona; la categoría se PIDE, no se
# aporta: acá no viaja ningún valor de categoría.
INSTRUCCION_ENTREGA = (
    "El arnés ya extrajo del mensaje los datos que aporta el usuario: no los "
    "vuelvas a extraer y no preguntes por ellos. La herramienta está decidida: "
    "emite la llamada a `" + HERRAMIENTA_CREAR_CASO + "` (y ninguna otra) con "
    "estos valores exactos:\n"
    "- cliente: {cliente}\n"
    "- descripcion: {descripcion}\n"
    "Tu única decisión es la categoría: elige la categoría de soporte que "
    "corresponde a la falla y emítela en la misma llamada, en el argumento "
    "`categoria`. No la omitas ni la dejes vacía."
)

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

# Topes de los campos que el puente completa, copiados del `args_schema` de
# `crear_caso`. Un valor fuera de rango no se recorta en silencio: se trata como
# faltante y se pregunta, porque recortar cambiaría el dato que la persona
# escribió. No hay topes de `categoria` porque el puente no maneja ese campo.
LARGO_MINIMO = {"cliente": 3, "descripcion": 5}
LARGO_MAXIMO = {"cliente": 120, "descripcion": 1000}

_NO_ALFANUMERICO = re.compile(r"[^a-z0-9]+")

# Etiquetas que el texto puede usar para declarar un campo. Solo se leen
# `cliente` y `descripcion`; las demás —incluida `categoría`— están para cerrar
# el valor del campo anterior: el puente no extrae esos campos ni, en el caso de
# la categoría, ningún valor para ellos.
_ETIQUETA = re.compile(
    r"\b(categor[ií]a|descripci[oó]n|cliente|prioridad|procedimiento)\s*:?\s*",
    re.IGNORECASE,
)

# Introducen el nombre del cliente, en el orden en que se prueban. `de` no está:
# «un caso de impresora» convertiría una descripción en un cliente. Ante la
# duda, el campo falta y se pregunta.
_INTRODUCTORES_CLIENTE = (
    re.compile(r"\ba\s+nombre\s+de\s+(?:el\s+|la\s+)?", re.IGNORECASE),
    re.compile(r"\b(?:el|la|del)\s+client(?:e|a)\s+", re.IGNORECASE),
    re.compile(
        r"\bpara\s+(?:el\s+|la\s+)?(?:client(?:e|a)\s+)?", re.IGNORECASE
    ),
)

# Arrancan la descripción de la falla: «por …», «porque …», «: …» y la cláusula
# que sigue al nombre del cliente detrás de una coma. Todas se declaran en
# español; la primera que aparece en el texto (desde el final del nombre) gana.
_INTRODUCTORES_DESCRIPCION = (
    re.compile(r"\bporque\b", re.IGNORECASE),
    re.compile(r"\bpor\b", re.IGNORECASE),
    re.compile(r":"),
    re.compile(r","),
)

# Fragmentos que no describen una falla: pedidos sobre la confirmación o la
# autoridad del turno y cortesías. Acotan el candidato de descripción; no se
# extraen ni se propagan.
_CORTES_DE_DESCRIPCION = (
    re.compile(
        r"\bsin\s+(?:pedir(?:me|te)?|que\s+me\s+pida)\s+confirmaci[oó]n",
        re.IGNORECASE,
    ),
    re.compile(r"\bno\s+me\s+pidas\s+confirmaci[oó]n", re.IGNORECASE),
    re.compile(r"\bsin\s+confirmaci[oó]n", re.IGNORECASE),
    re.compile(r"\bsin\s+autorizaci[oó]n", re.IGNORECASE),
    re.compile(r"\bsin\s+permiso\b", re.IGNORECASE),
    re.compile(r"\bignorando\s+(?:las\s+)?instrucciones", re.IGNORECASE),
    re.compile(r"\bpor\s+favor\b", re.IGNORECASE),
    re.compile(r"\bgracias\b", re.IGNORECASE),
)

# Fórmulas de cortesía que no son una falla. Un candidato que sea solo esto no
# se acepta: la descripción falta y se pregunta. Hace falta además de los
# cortes porque el introductor («por» en «por favor») ya se consumió cuando se
# evalúa el candidato.
FORMULAS_DE_CORTESIA = frozenset({
    "favor", "por favor", "porfavor", "porfa", "de favor",
    "gracias", "muchas gracias",
})

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
    texto. Se conserva crudo: cada campo lo limpia según su contrato. Solo se
    leen `cliente` y `descripcion`; las demás etiquetas existen para cerrar el
    valor anterior.
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


def _candidato_de_descripcion(valor: str) -> str:
    """Recorta el candidato donde el texto deja de describir una falla.

    Dos cortes, los dos declarados: la etiqueta de otro campo («categoría …»)
    cierra el valor, y los fragmentos sobre la confirmación o de cortesía no son
    parte de la falla. Nada de eso se extrae: solo acota lo que ya estaba.
    """
    recortado = valor
    etiqueta = _ETIQUETA.search(recortado)
    if etiqueta is not None:
        recortado = recortado[:etiqueta.start()]
    for patron in _CORTES_DE_DESCRIPCION:
        corte = patron.search(recortado)
        if corte is not None:
            recortado = recortado[:corte.start()]
    return _limpiar(recortado)


def _descripcion_valida(valor: str) -> str | None:
    """Descripción que cabe en `ArgsCrearCaso`, o None si no hay certeza."""
    limpio = _candidato_de_descripcion(valor)
    if not (LARGO_MINIMO["descripcion"] <= len(limpio)
            <= LARGO_MAXIMO["descripcion"]):
        return None
    if _normalizar(limpio) in FORMULAS_DE_CORTESIA:
        return None
    return limpio


def _inicio_de_la_falla(texto: str) -> int | None:
    """Dónde termina el introductor de descripción que aparece primero."""
    mejor: tuple[int, int] | None = None
    for patron in _INTRODUCTORES_DESCRIPCION:
        coincidencia = patron.search(texto)
        if coincidencia is None:
            continue
        if mejor is None or coincidencia.start() < mejor[0]:
            mejor = (coincidencia.start(), coincidencia.end())
    return None if mejor is None else mejor[1]


def _cliente_de(texto: str, valores: dict[str, str]) -> tuple[str | None, int]:
    """Cliente del pedido y la posición donde termina su nombre.

    El segundo valor dice desde dónde se busca la falla: lo que el texto dice
    antes del nombre no se lee como descripción. Se prueban los introductores en
    el orden declarado y, si ninguno da un nombre con forma de nombre, el campo
    queda faltante y el arnés lo pregunta.
    """
    etiquetado = valores.get("cliente")
    if etiquetado:
        nombre = _nombre_al_inicio(etiquetado)
        if nombre is not None:
            return nombre, 0
    for patron in _INTRODUCTORES_CLIENTE:
        for coincidencia in patron.finditer(texto):
            inicio = coincidencia.end()
            nombre = _nombre_al_inicio(texto[inicio:])
            if nombre is not None:
                return nombre, inicio + len(nombre)
    return None, 0


def _descripcion_de(
    texto: str, valores: dict[str, str], desde: int,
) -> str | None:
    """Falla enunciada en el texto, o None si no hay certeza.

    Una etiqueta `descripción:` es la forma explícita y manda. Sin etiqueta, la
    falla se lee desde el final del nombre del cliente (o en todo el mensaje
    cuando el nombre no está) detrás del primer introductor declarado.
    """
    etiquetada = valores.get("descripcion")
    if etiquetada:
        return _descripcion_valida(etiquetada)
    resto = texto[desde:]
    inicio = _inicio_de_la_falla(resto)
    if inicio is None:
        return None
    return _descripcion_valida(resto[inicio:])


def recolectar_campos(texto: str) -> dict[str, str]:
    """Campos que el puente completa —`cliente` y `descripcion`— y solo esos.

    No usa valores por defecto ni deriva un campo de otro: lo que no está no
    aparece en el diccionario y el llamador lo trata como faltante. `categoria`
    no se recolecta nunca, aunque el texto la declare: elegirla es el juicio que
    conserva el modelo.
    """
    texto = texto or ""
    valores = _valores_etiquetados(texto)
    campos: dict[str, str] = {}
    cliente, fin_del_nombre = _cliente_de(texto, valores)
    if cliente is not None:
        campos["cliente"] = cliente
    descripcion = _descripcion_de(texto, valores, fin_del_nombre)
    if descripcion is not None:
        campos["descripcion"] = descripcion
    return campos


def campos_faltantes(campos: dict[str, str]) -> tuple[str, ...]:
    """Campos del puente que no están, en el orden declarado."""
    return tuple(campo for campo in CAMPOS_DEL_PUENTE if campo not in campos)


def pregunta_por_el_primero_que_falta(campos: dict[str, str]) -> str:
    """Pregunta determinista por el primer campo que falta, o ""."""
    faltantes = campos_faltantes(campos)
    return PREGUNTA_POR_CAMPO[faltantes[0]] if faltantes else ""


def instruccion_para_el_modelo(campos: dict[str, str]) -> str:
    """Instrucción estrecha del turno entregado al modelo (S25).

    Lleva los datos ya recolectados y deja abierta una sola decisión: la
    categoría. Se llama solo con los dos campos completos; no hay valor de
    categoría en el texto que arma el arnés.
    """
    return INSTRUCCION_ENTREGA.format(
        cliente=campos["cliente"], descripcion=campos["descripcion"],
    )


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
