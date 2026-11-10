"""Bucle del agente de TecnoApoyo (RA2 / IL2.1).

El agente no genera la respuesta de una sola pasada: usa *function calling*
nativo del proveedor compatible con OpenAI. El modelo decide qué herramientas
llamar, el bucle las ejecuta, devuelve el resultado como `ToolMessage` y el
modelo continúa hasta que deja de pedir herramientas.

Tres invariantes sostienen este módulo, y las tres son del ejecutor, no del
modelo:

  1. Tope duro de pasos (`max_pasos`). Sin él, un modelo que encadena
     llamadas puede girar indefinidamente. Al agotarse el tope el resultado es
     vacío con un error explícito: nunca se inventa una respuesta final.
  2. Guard de abstención. `buscar_documentacion` ya decide si la evidencia
     alcanza el umbral del motor RAG; si la última búsqueda dijo que no
     alcanza, el ejecutor impone `FRASE_ABSTENCION` aunque el modelo insista.
     La abstención queda así por debajo de la generación, no a su merced.
  3. Errores contenidos. Una herramienta inexistente, argumentos que no pasan
     la validación Pydantic o una excepción dentro de la herramienta se
     registran en el paso y el bucle sigue: el error se le informa al modelo,
     que puede corregir el rumbo, en vez de abortar la conversación.
  4. Detección de repetición. El presupuesto de pasos se gasta en avanzar, no
     en repetir lo mismo. Una llamada idéntica a otra ya ejecutada en el turno
     (mismo nombre y mismos argumentos canónicos) no se vuelve a ejecutar: se
     responde al modelo con un error que lo invita a cambiar de estrategia; a
     la segunda repetición idéntica el bucle se corta sin respuesta final.

El modelo se inyecta (`llm`), de modo que todo el bucle —`bind_tools`,
ejecución, `ToolMessage`, tope— se puede probar sin red. `ChatOpenAI` solo se
construye si hace falta de verdad, en la primera consulta.

La consola no espera el turno completo: `consultar` acepta un aviso por paso
(`al_paso`) y un emisor de trozos (`al_emitir`) para la respuesta final, en los
dos modos. En el planificado transmite la síntesis, salvo cuando el turno puede
terminar en otro texto —un listado que el guard corregiría o evidencia
documental deficiente que impone la abstención—: ahí la emisión se difiere
hasta después de la decisión, para que lo transmitido sea el texto final. En el
reactivo transmite el turno del modelo que redacta la respuesta. Sin
`al_emitir` todo pasa por `invoke`, que es la ruta de la evaluación: silenciosa,
sin stream y con la contabilidad de tokens del callback.
"""
import json
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from pydantic import SecretStr

from src import config
from src.capacidades import (
    TEXTO_CAPACIDADES,
    es_consulta_de_capacidades,
    normalizar,
)
from src.herramientas import (
    crear_herramienta_recordar,
    mensaje_del_turno,
    por_nombre,
)
from src.puente_escritura import (
    PREGUNTA_CLIENTE,
    PedidoDeCrearCaso,
    detectar_pedido_de_crear_caso,
    instruccion_para_el_modelo,
    pregunta_por_el_primero_que_falta,
)
from src.rag import ColectorTokens, FRASE_ABSTENCION, PREFIJO_ABSTENCION

# `FRASE_ABSTENCION` y `PREFIJO_ABSTENCION` se importan de `src.rag`, que es
# quien produce la abstención: una sola definición en todo el asistente para que
# la frase sea reconocible y evaluable, y para que el guard compare contra la
# misma cadena que el motor devuelve. Se reexportan con su nombre público; lo
# que no existe es una copia que pueda editarse por separado.

# Nombre de la herramienta cuyo veredicto de evidencia controla el guard.
HERRAMIENTA_BUSQUEDA = "buscar_documentacion"

# Herramientas que aportan datos de un caso concreto. Un caso recuperado con
# éxito es evidencia operativa válida aunque la documentación no cubra la
# pregunta, así que limpia un veredicto documental previo negativo.
HERRAMIENTAS_CASO = ("consultar_caso", "listar_casos")

MENSAJE_SIN_RESPUESTA = "El modelo no devolvió una respuesta final."

# Aviso que recibe el modelo cuando insiste con una llamada ya ejecutada. Es
# un `ToolMessage` más: el bucle no se detiene, pero el paso queda en False y la
# herramienta no se ejecuta por segunda vez.
MENSAJE_REPETICION = (
    "Ya llamaste a '{herramienta}' con estos mismos argumentos en este turno. "
    "No la ejecuto de nuevo: responde con lo que ya tienes o cambia de "
    "estrategia (otros argumentos u otra herramienta)."
)

# Corte por insistencia: tres llamadas idénticas en el mismo turno. No hay
# respuesta final ni abstención, solo el error explícito en la traza.
ERROR_REPETICION = (
    "Corte por repetición: '{herramienta}' se llamó tres veces con los mismos "
    "argumentos en este turno; no hay respuesta final del modelo."
)

# Encabezado del bloque de historial que viaja en el prompt. El de hechos vive
# en `src.memoria`, que es quien lo redacta: una sola definición por bloque,
# porque dos redacciones distintas serían dos contratos distintos.
ENCABEZADO_HISTORIAL = "Historial reciente:"

# --- guard de la síntesis de `listar_casos` (S1/S2) ----------------------
# La síntesis redacta sobre lo que devolvieron las herramientas, pero puede
# negar lo que la herramienta acaba de probar: el defecto real fue
# `listar_casos | ok=sí` y enseguida "No tengo casos abiertos asociados a tu
# nombre". El guard es determinista: no vuelve a llamar al modelo.

# Frases con las que una respuesta afirma que no hay casos. Se buscan sobre el
# texto normalizado (minúsculas y sin tildes), así que "No se encontraron" y
# "ningún caso" también cuentan.
AFIRMACIONES_DE_AUSENCIA_CASOS = (
    "no hay",
    "no tengo",
    "no se encontr",
    "sin casos",
    "ningun caso",
)

# Tope de casos que enumera el resumen: la respuesta es un resumen, no el
# listado completo, y un cliente con decenas de casos no puede desbordar la
# consola. El resto se cuenta en una línea.
TOPE_CASOS_RESUMEN = 5

ENCABEZADO_RESUMEN_CASOS = "Encontré {total} caso{plural}:"
LINEA_RESUMEN_CASO = "- {id} · {estado} · {categoria}"
LINEA_RESUMEN_RESTANTES = "- y {restantes} caso{plural} más."

# Tope del bloque de historial. Una sesión larga no puede desplazar a los hechos
# del usuario ni al contexto recuperado; el recorte conserva la cola.
LIMITE_HISTORIAL = 1500

# Primera línea del contexto del planificador: qué alcance tienen las
# herramientas de este turno. Sin ella el modelo no sabía si listar o consultar
# casos exigía un cliente identificado y respondía con `pedir_datos` una
# pregunta que el alcance global no necesita (o que la memoria ya contestaba).
ALCANCE_GLOBAL = (
    "Alcance de las herramientas: global; no requiere cliente identificado."
)
ALCANCE_SESION_CON_CLIENTE = (
    "Alcance de las herramientas: sesión; cliente identificado: {cliente}."
)
ALCANCE_SESION_SIN_CLIENTE = (
    "Alcance de las herramientas: sesión; sin cliente identificado: para "
    "listar o consultar casos hay que pedir el nombre con pedir_datos."
)
# Atributos del módulo `config` donde puede vivir el cliente de la sesión. No
# se declaran en `src/config.py`: son estado de la consola, no configuración del
# proyecto, y por eso se leen con `getattr`. El primero es el nombre del
# contrato; el segundo es el que escribe `src/app.py`.
CLAVES_CLIENTE_SESION = ("CLIENTE_SESION", "CLIENTE_ACTUAL")

# La pregunta canónica de identificación vive en `src.puente_escritura`, que
# también la usa como pregunta del campo `cliente` de `crear_caso`. Se importa y
# se reexporta con su nombre público: una sola definición para los dos atajos, y
# `src.planificador` y la consola siguen leyendo `agente.PREGUNTA_CLIENTE`.

# Línea única con la que el cliente de la sesión entra a los dos prompts (el de
# la conversación y el del planificador). El nombre va saneado: ver
# `sanear_nombre_cliente`.
ETIQUETA_CLIENTE_SESION = "Cliente de la sesión: {cliente}"

# Forma explícita de la ausencia de cliente. Nunca se escribe un nombre que no
# exista: el prompt declara que falta y el planificador pide el dato.
ETIQUETA_CLIENTE_SIN_IDENTIFICAR = "Cliente de la sesión: sin cliente identificado"

# Tope del nombre que viaja en el prompt. Un nombre real no llega a 120
# caracteres; lo que sí llega es un texto que intenta instruir al modelo, y el
# tope acota cuánto de eso entra al sistema.
TOPE_NOMBRE_CLIENTE = 120

# --- detector determinista de consulta de casos (S2) ----------------------
# Sustantivos que nombran un caso de soporte. Sin uno de ellos no hay consulta
# de casos, por más que el turno pregunte algo.
_SUSTANTIVOS_CASO = frozenset({"caso", "casos"})

# Indicios de que el turno pide datos de esos casos (estado, listado,
# seguimiento) y no otra cosa que los menciona de pasada: "¿cómo creo un caso?"
# no pide datos de casos existentes y no se intercepta.
_INDICIOS_CONSULTA_CASO = frozenset({
    "que", "cual", "cuales", "cuanto", "cuantos", "cuanta", "cuantas",
    "estado", "situacion", "listar", "listado", "lista", "listas",
    "listame", "listan", "mis", "tengo", "hay", "ver", "mostrar",
    "muestra", "muestrame", "dame", "detalle", "seguimiento", "abierto",
    "abiertos", "pendiente", "pendientes", "consultar", "revisar",
})

# Temas que pertenecen a la documentación, a los plazos o a las capacidades.
# Una pregunta que los menciona no se intercepta aunque diga "caso": la
# respuesta correcta está en el camino normal (RAG o cálculo de plazo).
_TEMAS_AJENOS_A_CASOS = frozenset({
    "garantia", "garantias", "politica", "politicas", "procedimiento",
    "procedimientos", "documentacion", "documento", "documentos", "precio",
    "precios", "horario", "horarios", "plazo", "plazos", "demora", "demoras",
    "capacidad", "capacidades", "funcion", "funciones",
})

# Formas que introducen el nombre de una persona en un turno: «los casos
# abiertos de Valentina Soto», «soy Valentina Soto», «a nombre de». La palabra
# siguiente solo cuenta como nombre si va capitalizada.
_INTRODUCTORES_DE_NOMBRE = frozenset({"de", "del", "para", "soy", "nombre"})

# Palabras que una persona capitaliza sin que sean su nombre («el estado del
# Caso 0007», «el Equipo»). Sin esta lista, cualquier sustantivo del dominio
# detrás de un introductor se leería como nombre declarado.
_PALABRAS_QUE_NO_SON_NOMBRE = frozenset({
    "caso", "casos", "cliente", "clientes", "equipo", "equipos",
    "diagnostico", "reparacion", "repuesto", "soporte", "tecnicoapoyo",
}) | _TEMAS_AJENOS_A_CASOS | _SUSTANTIVOS_CASO

# --- puente determinista de escritura (S23/S25) --------------------------
# El reparto es el de S25: el puente hace la mecánica y el modelo el juicio.
# Las notas van a `pensamientos`, el campo de la traza que
# `observabilidad.traza_de_turno` persiste además de los pasos, para que el
# operador vea qué hizo el arnés y qué el modelo. El texto lo dice: no es
# razonamiento del modelo.
NOTA_PUENTE_ESCRITURA = (
    "Puente determinista de escritura: el arnés resolvió este turno antes de "
    "llamar al modelo; cualquier escritura pasó por la guarda de confirmación "
    "humana."
)

# Nota del turno que sí se le entrega al modelo: los datos los extrajo el
# arnés, la categoría la decide el modelo y la escritura sigue pasando por la
# guarda de confirmación.
NOTA_ENTREGA_AL_MODELO = (
    "Puente determinista de escritura: el arnés extrajo del mensaje los datos "
    "del usuario (cliente y descripción) y se los entrega al modelo, que "
    "decide solo la categoría; cualquier escritura pasó por la guarda de "
    "confirmación humana."
)

# Reglas de dominio del asistente. Viven en una sola constante porque valen
# para los dos modos: el reactivo las recibe en su prompt de sistema y el
# planificador las interpola en su propio prompt. Tenerlas duplicadas fue el
# defecto: el modo planificado planificaba sin saber que el dominio no tiene
# correo, que no se enumeran casos de otros clientes ni que los PROC-* no se
# inventan. `src/planificador.py` importa esta constante; no se copia el texto.
REGLAS_DOMINIO = """Reglas de dominio (idénticas en los dos modos, reactivo y planificado):
1. El dominio de casos NO tiene correo electrónico: el único dato del cliente es su nombre. Nunca afirmes que buscas, filtras o asocias casos por correo; si el usuario menciona uno, dile que no se puede usar como filtro y pide el nombre del cliente o el id del caso.
2. Nunca enumeres casos, nombres, descripciones ni notas de otros clientes. Solo muestras los casos del cliente identificado en la sesión.
3. Si no hay un cliente identificado, no consultes ni listes casos: pide al operador que identifique al cliente con /cliente <nombre> antes de seguir.
4. Nunca inventes un procedimiento PROC-*: si la documentación no confirma el procedimiento asociado, deja procedimiento vacío en vez de suponerlo.
5. Cuando el usuario diga su nombre para consultar sus casos, identifícalo primero con identificar_cliente: sin cliente identificado, listar_casos, consultar_caso y las escrituras se niegan. Si no hay casos a ese nombre, pide el nombre completo tal como figura en el caso o el id CASO-NNNN.
6. La identidad solo se toma del nombre que la persona declara en el mensaje del turno actual. Nunca identifiques a un cliente con datos de memoria (un correo guardado, un nombre recordado) ni con el historial: nada de eso prueba quién es la persona que está escribiendo.
7. En `calcular_plazo`, una consulta que menciona un repuesto usa `tipo_servicio="repuesto"` (7 días hábiles); `reparacion` es la reparación sin repuesto (5 días). No uses `reparacion` cuando el mensaje dice "con repuesto"."""

# Reglas del bucle y de las herramientas: cómo se comporta el agente, no qué
# sabe del dominio. Las de dominio se agregan desde `REGLAS_DOMINIO`.
_PROMPT_SISTEMA_BASE = f"""Eres el agente de soporte de TecnoApoyo, una pyme de reparación de computadores en Santiago de Chile.

Atiendes consultas de clientes y de recepción usando las herramientas disponibles. Nunca inventes datos, precios, plazos ni políticas.

Herramientas disponibles:
- Consulta: buscar_documentacion (documentación oficial), consultar_caso y listar_casos (casos de soporte).
- Escritura: crear_caso, agregar_nota, actualizar_estado y escalar_a_humano. Todas requieren confirmación de una persona: tú solo propones la acción, no puedes autorizarla.
- Razonamiento: calcular_plazo, que calcula fechas comprometidas en días hábiles sin adivinar.
- Identificación: identificar_cliente, que ata la sesión al cliente que declara el usuario.
- Datos faltantes: pedir_datos, que pregunta al usuario el dato que falta en vez de inventarlo.

Reglas estrictas:
1. Ante preguntas sobre políticas, procedimientos, horarios, precios o garantías, busca primero en la documentación con buscar_documentacion.
2. Si buscar_documentacion devuelve evidencia_suficiente en False, debes abstenerte: responde exactamente "{FRASE_ABSTENCION}" y no completes con conocimiento general.
3. Cuando la evidencia alcance, responde en español, breve y claro, citando los datos de los documentos.
4. Usa calcular_plazo para las fechas comprometidas en días hábiles; no calcules plazos por tu cuenta.
5. Si una herramienta responde ok en False, explica el problema al usuario en vez de repetir la misma llamada."""

PROMPT_SISTEMA = f"{_PROMPT_SISTEMA_BASE}\n\n{REGLAS_DOMINIO}\n"

# Centinela interno: distingue "no hubo veredicto de evidencia" de un
# veredicto explícito False, que es el que dispara el guard.
_SIN_VEREDICTO = object()

# Síntesis del modo planificado: el texto final se redacta a partir de lo que
# devolvieron las herramientas, no de lo que el modelo recuerde. La frase de
# abstención entra como argumento, igual que en `PROMPT_CONVERSACION`: ninguna
# de las dos plantillas guarda una copia de la redacción canónica.
PROMPT_SINTESIS = """Responde al cliente de TecnoApoyo usando SOLO los resultados de las herramientas ejecutadas.

Si los resultados no contienen información suficiente, responde exactamente: "{abstencion}"

Objetivo del cliente: {objetivo}

Resultados de las herramientas:
{pasos}

Respuesta final, breve y en español, sin explicar tu proceso.
"""

# Plan vacío: el modelo decidió que la consulta no necesita herramientas. La
# síntesis normal respondería con la abstención ante "(sin resultados)", así
# que la conversación tiene su propia instrucción y, si lo que pide sí exigía
# evidencia, deriva a la misma frase canónica.
PROMPT_CONVERSACION = """No se ejecutó ninguna herramienta: la consulta no necesita consultar documentación ni casos.

Responde al cliente de TecnoApoyo en español, breve y cordial, sin inventar datos operativos (casos, plazos, políticas, precios ni estados).
Si el historial o los hechos recordados en el contexto ya contienen el dato que pide, úsalo y no pidas un nombre ni un identificador para acceder a él.
Si lo que pide sí requiere consultar la documentación o un caso, responde exactamente: "{abstencion}"

Objetivo del cliente: {objetivo}

Respuesta final, breve y en español, sin explicar tu proceso.
"""


def una_sola_linea(texto: Any) -> str:
    """Colapsa `texto` a una sola línea: sin control, sin saltos, sin dobles espacios.

    Se usa para el nombre del cliente antes de que entre al prompt. Un salto de
    línea dentro de un nombre abriría una línea nueva en el prompt del sistema,
    que es exactamente el vector de inyección de instrucciones: el texto del
    usuario dejaría de ser un dato y pasaría a ser una directiva. Los
    caracteres de control (incluidos `\n`, `\r` y `\t`), los separadores de
    línea Unicode y los formatos invisibles se reemplazan por un espacio.
    """
    if not isinstance(texto, str):
        return ""
    sin_control = "".join(
        " " if unicodedata.category(caracter) in ("Cc", "Cf", "Zl", "Zp")
        else caracter
        for caracter in texto
    )
    return " ".join(sin_control.split())


def sanear_nombre_cliente(nombre: Any) -> str:
    """Nombre del cliente en una línea y con tope de longitud (S3).

    Devuelve "" cuando no hay un texto usable: un valor vacío o de otro tipo
    cuenta como «sin cliente» y nunca se convierte en un nombre inventado.
    """
    limpio = una_sola_linea(nombre)
    if len(limpio) > TOPE_NOMBRE_CLIENTE:
        limpio = limpio[:TOPE_NOMBRE_CLIENTE].rstrip()
    return limpio


def linea_de_cliente(nombre: Any) -> str:
    """Línea canónica `Cliente de la sesión: <nombre>`, o "" si no hay nombre.

    Es la única redacción del cliente de la sesión: la usan el prompt de la
    conversación y el del planificador, y por eso el nombre se sanea en un solo
    lugar.
    """
    limpio = sanear_nombre_cliente(nombre)
    if not limpio:
        return ""
    return ETIQUETA_CLIENTE_SESION.format(cliente=limpio)


def _palabras_de(texto: str) -> list[str]:
    """Palabras del texto conservando mayúsculas: solo letras, sin dígitos.

    Los ids pierden la parte numérica («CASO-0004» queda como «CASO») y las
    mayúsculas se conservan, que es justo lo que distingue un nombre de una
    palabra cualquiera.
    """
    return "".join(
        letra if letra.isalpha() else " " for letra in (texto or "")
    ).split()


def _es_nombre_de_persona(palabra: str) -> bool:
    """Verdadero si la palabra se escribe como el nombre de una persona.

    Exige mayúscula inicial y descarta dos casos: las siglas o ids todo en
    mayúsculas («CASO») y los sustantivos del dominio que alguien capitaliza
    sin querer decir un nombre («el estado del Caso 0007»).
    """
    if not palabra[:1].isupper():
        return False
    if len(palabra) > 1 and palabra.isupper():
        return False
    return normalizar(palabra) not in _PALABRAS_QUE_NO_SON_NOMBRE


def _declara_nombre_de_cliente(texto: str) -> bool:
    """Verdadero si el turno ya escribe el nombre del cliente (S4).

    Determinista y sin base de datos: la detección es del texto. Cuenta un
    nombre propio introducido por «de», «del», «para», «soy» o «nombre»
    («los casos abiertos de Valentina Soto»). La mayúscula sola no basta: la
    prosa que capitaliza cada palabra —«Necesito Ver Los Casos Abiertos»,
    «Hola, Muéstrame Los Casos Abiertos»— no declara a nadie, y leerla como
    nombre suprimía la pregunta de identificación de una sesión sin cliente
    (R3-001). Sin esta guarda la pregunta canónica de identificación se dispara
    aunque el nombre ya esté en el mensaje, y el turno gasta el atajo en pedir
    un dato que el usuario acaba de dar.
    """
    palabras = _palabras_de(texto)
    for anterior, palabra in zip(palabras, palabras[1:]):
        if (
            _es_nombre_de_persona(palabra)
            and normalizar(anterior) in _INTRODUCTORES_DE_NOMBRE
        ):
            return True
    return False


def es_consulta_de_casos(texto: str) -> bool:
    """Verdadero solo si el turno pide datos de casos (estado o listado).

    Determinista y sin modelo: normaliza acentos y mayúsculas y decide por
    vocabulario. Exige un sustantivo de caso Y un indicio de consulta, y se
    descarta cuando el tema es documentación, plazo o capacidades: esas
    preguntas se responden por el camino normal aunque mencionen un caso.
    También se descarta cuando el turno ya declara el nombre del cliente: no hay
    nada que preguntar y el camino normal identifica con ese nombre (S4).
    """
    if es_consulta_de_capacidades(texto):
        return False
    palabras = set(normalizar(texto).split())
    if not palabras & _SUSTANTIVOS_CASO:
        return False
    if palabras & _TEMAS_AJENOS_A_CASOS:
        return False
    if not palabras & _INDICIOS_CONSULTA_CASO:
        return False
    return not _declara_nombre_de_cliente(texto)


def _razonamiento_de(mensaje: Any) -> str:
    """`reasoning_content` del proveedor, si lo entrega (R1 y similares)."""
    for fuente in (
        getattr(mensaje, "additional_kwargs", None),
        getattr(mensaje, "response_metadata", None),
    ):
        if isinstance(fuente, dict):
            valor = fuente.get("reasoning_content")
            if isinstance(valor, str) and valor.strip():
                return valor.strip()
    return ""


def afirma_ausencia_de_casos(texto: str) -> bool:
    """Verdadero si la respuesta afirma que no hay casos (S1).

    Normaliza minúsculas y tildes antes de comparar: el modelo escribe la
    misma negación de varias formas ("No se encontraron casos", "sin casos")
    y la comparación literal dejaría pasar el defecto con una tilde distinta.
    """
    normalizado = normalizar(texto)
    return any(frase in normalizado for frase in AFIRMACIONES_DE_AUSENCIA_CASOS)


def _total_y_casos_de(paso: "PasoAgente") -> tuple[int, list[dict]] | None:
    """(total, casos) de la salida de un `listar_casos`, o None si no se lee.

    La salida del paso es el mismo JSON que vio el modelo, pero un doble de
    test puede entregarla ya como dict: se aceptan las dos formas y cualquier
    otra cosa se trata como «no hay nada que verificar».
    """
    salida = paso.salida
    if isinstance(salida, str):
        try:
            salida = json.loads(salida)
        except (TypeError, ValueError):
            return None
    if not isinstance(salida, dict):
        return None
    datos = salida.get("datos")
    if not isinstance(datos, dict):
        return None
    casos = datos.get("casos")
    casos = casos if isinstance(casos, list) else []
    total = datos.get("total")
    if not isinstance(total, int) or isinstance(total, bool):
        total = len(casos)
    return total, [caso for caso in casos if isinstance(caso, dict)]


def ultimo_listado_exitoso(
    pasos: list["PasoAgente"],
) -> tuple[int, list[dict]] | None:
    """(total, casos) del último `listar_casos` exitoso del turno, o None.

    Solo el listado: un `consultar_caso` exitoso prueba un caso por id, no un
    listado, y la documentación no habla de casos del cliente. Un listado
    posterior sí reemplaza al anterior, porque es el que sostiene la respuesta.
    """
    for paso in reversed(pasos):
        if paso.herramienta != "listar_casos" or not paso.ok:
            continue
        return _total_y_casos_de(paso)
    return None


def resumen_de_casos(total: int, casos: list[dict]) -> str:
    """Resumen determinista de los casos listados: id, estado y categoría.

    Es la respuesta que reemplaza a una síntesis que negó el listado. Se
    muestran hasta `TOPE_CASOS_RESUMEN` casos y el resto se cuenta en una
    línea: el texto es corto, estable y no depende de otra llamada al modelo.
    """
    plural = "s" if total != 1 else ""
    lineas = [ENCABEZADO_RESUMEN_CASOS.format(total=total, plural=plural)]
    for caso in casos[:TOPE_CASOS_RESUMEN]:
        lineas.append(LINEA_RESUMEN_CASO.format(
            id=caso.get("id") or "sin id",
            estado=caso.get("estado") or "sin estado",
            categoria=caso.get("categoria") or "sin categoría",
        ))
    restantes = total - len(casos[:TOPE_CASOS_RESUMEN])
    if restantes > 0:
        lineas.append(LINEA_RESUMEN_RESTANTES.format(
            restantes=restantes, plural="s" if restantes != 1 else "",
        ))
    return "\n".join(lineas)


def corregir_sintesis_de_listado(
    pasos: list["PasoAgente"], respuesta: str,
) -> str:
    """S1/S2: la síntesis no puede negar un listado exitoso con casos.

    Devuelve el texto del modelo sin cambios cuando no hay listado exitoso,
    cuando el listado trae `total == 0` (ahí "no hay casos" es la respuesta
    correcta) o cuando la síntesis no afirma ausencia. Si se contradicen, la
    respuesta pasa a ser el resumen determinista de lo que la herramienta sí
    devolvió y el modelo no se vuelve a invocar.
    """
    listado = ultimo_listado_exitoso(pasos)
    if listado is None:
        return respuesta
    total, casos = listado
    if total <= 0 or not afirma_ausencia_de_casos(respuesta):
        return respuesta
    return resumen_de_casos(total, casos)


def _pregunta_de_pedir_datos(pasos: list["PasoAgente"]) -> str | None:
    """Pregunta del último paso, si fue un `pedir_datos` exitoso.

    Solo se consulta el último paso: un plan validado que pide datos tiene ese
    ÚNICO paso, así que si la pregunta no cerró el turno no hay pregunta que
    mostrar. Se devuelve el texto exacto, sin recortes ni reformateo.
    """
    if not pasos:
        return None
    paso = pasos[-1]
    if paso.herramienta != "pedir_datos" or not paso.ok:
        return None
    try:
        payload = json.loads(paso.salida or "")
    except (TypeError, ValueError):
        return None
    datos = payload.get("datos") if isinstance(payload, dict) else None
    pregunta = datos.get("pregunta") if isinstance(datos, dict) else None
    if isinstance(pregunta, str) and pregunta.strip():
        return pregunta
    return None


@dataclass
class PasoAgente:
    """Una ejecución de herramienta, tal como quedó en la traza.

    `salida` es el JSON serializable que devolvió la herramienta (lo mismo que
    vio el modelo en el `ToolMessage`) y `error` explica por qué falló. El
    default de `ok` es False porque un paso sin ejecución confirmada no debe
    leerse como exitoso: el bucle siempre lo escribe de forma explícita.
    """

    numero: int
    herramienta: str
    argumentos: dict
    salida: str | None = None
    error: str | None = None
    ok: bool = False
    # Id del paso del plan que originó la ejecución (`p1`, `p2`, ...). La
    # consola cruza plan y traza por esta identidad: los argumentos de la traza
    # vienen resueltos y no coinciden con los placeholders del plan. Vale None
    # en las trazas del bucle reactivo, que no tienen plan.
    paso_id: str | None = None
    # Duración medida de la ejecución de la herramienta, en segundos: la traza
    # necesita latencia por paso, no solo la del turno completo. `None`
    # significa «no se midió» y es el único valor honesto para un paso que no
    # ejecutó ninguna herramienta (atajo determinista, herramienta
    # desconocida, llamada repetida rechazada): un 0.0 se lee como una
    # ejecución instantánea y es indistinguible de un paso sin medir.
    duracion_seg: float | None = None


@dataclass
class RespuestaAgente:
    """Resultado de una consulta al agente, con su traza de pasos."""

    pregunta: str
    respuesta: str
    abstencion: bool
    pasos: list[PasoAgente]
    iteraciones: int
    tokens_total: int
    # Latencia del turno en segundos, medida con el reloj monótono de
    # `_duracion_desde` y redondeada a microsegundos. No es la resta de dos
    # lecturas de `time.time()`: ese reloj avanza a saltos de milisegundo y el
    # redondeo a centésimas publicaba 0.0 en un turno que tardó microsegundos,
    # un valor indistinguible de «no se midió».
    latencia_seg: float
    error: str | None = None
    # Modo planificado: plan ejecutado y cuántas veces se re-planificó.
    replanes: int = 0
    plan: Any = None
    # Texto que el modelo emitió antes de cada herramienta, más el
    # `reasoning_content` cuando el proveedor lo entrega. No es una cadena de
    # pensamiento forzada: si el proveedor no emite razonamiento, queda vacío.
    # Un atajo determinista escribe aquí su propia nota, porque es el único
    # campo de la traza que `observabilidad.traza_de_turno` persiste además de
    # los pasos: así el operador ve que el turno no salió del modelo.
    pensamientos: list[str] = field(default_factory=list)


def _texto_de(mensaje: Any, *, recortar: bool = True) -> str:
    """Extrae el texto de un `AIMessage`, que puede venir en bloques.

    `recortar` es la única diferencia entre los dos usos: la respuesta final se
    recorta; un trozo de streaming puede ser solo un espacio y recortarlo
    perdería la separación entre palabras al ensamblar el texto completo.
    """
    contenido = getattr(mensaje, "content", "") or ""
    if isinstance(contenido, str):
        texto = contenido
    elif isinstance(contenido, list):
        partes = []
        for bloque in contenido:
            if isinstance(bloque, str):
                partes.append(bloque)
            elif isinstance(bloque, dict) and bloque.get("type") == "text":
                partes.append(str(bloque.get("text", "")))
        texto = "".join(partes)
    else:
        texto = str(contenido)
    return texto.strip() if recortar else texto


def _json_error(mensaje: str) -> str:
    """Payload de error que recibe el modelo cuando el paso no se ejecutó."""
    return json.dumps({"ok": False, "datos": None, "error": mensaje}, ensure_ascii=False)


def _sumar_uso(colector: ColectorTokens, mensaje: Any) -> None:
    """Suma el `usage_metadata` de un trozo al colector de tokens.

    Un `.stream()` directo no dispara `on_llm_end`, así que el callback de
    LangChain no ve nada (medido: 0/0/0). Con `stream_usage=True` el proveedor
    adjunta el uso en el último trozo y aquí se suma a mano. Los campos que
    falten se tratan como 0 para no romper la contabilidad con otro proveedor.
    """
    uso = getattr(mensaje, "usage_metadata", None)
    if not isinstance(uso, dict):
        return
    colector.prompt_tokens += uso.get("input_tokens", 0) or 0
    colector.completion_tokens += uso.get("output_tokens", 0) or 0
    colector.total_tokens += uso.get("total_tokens", 0) or 0


def _duracion_desde(inicio: float) -> float:
    """Segundos transcurridos desde `inicio`, medidos con el reloj monótono.

    `time.time()` no sirve para medir una ejecución: no es monótono —un ajuste
    del reloj devuelve duraciones negativas— y en Windows avanza a saltos de
    milisegundo, así que dos lecturas alrededor de una llamada rápida
    devuelven el mismo valor. Medido en este equipo: 15 valores distintos en
    200.000 lecturas, y 200 de 200 llamadas de 6 µs registradas como 0.0 al
    redondear a tres decimales. Por eso el reloj es `perf_counter` y la
    precisión, de microsegundos: un paso que llamó a una herramienta nunca
    escribe un 0.0 que se confunda con un paso sin medir.

    La duración de un paso y la latencia del turno entero se miden igual: quien
    llama toma `time.perf_counter()` al empezar el intervalo y pasa ese valor
    aquí. Así un turno resuelto sin modelo —los atajos deterministas tardan
    microsegundos— tampoco publica un 0.0.
    """
    return round(time.perf_counter() - inicio, 6)


def _avisar_paso(al_paso: Any | None, paso: PasoAgente) -> None:
    """Avisa un paso ejecutado sin que un fallo del aviso tumbe el turno.

    La consola es accesoria: si el callback lanza (un flujo cerrado, por
    ejemplo), la consulta sigue y la traza queda completa.
    """
    if al_paso is None:
        return
    try:
        al_paso(paso)
    except Exception:  # noqa: BLE001 - el progreso no puede romper la consulta
        return


def _campo_de(llamada: Any, campo: str) -> str:
    """Lee `name` o `id` de una llamada venga como objeto o como dict."""
    valor = getattr(llamada, campo, None)
    if valor is None and isinstance(llamada, dict):
        valor = llamada.get(campo)
    return str(valor) if valor is not None else ""


def _args_de(llamada: Any) -> dict[str, Any]:
    """Normaliza los argumentos de una llamada a herramienta."""
    argumentos = getattr(llamada, "args", None)
    if argumentos is None and isinstance(llamada, dict):
        argumentos = llamada.get("args")
    return argumentos if isinstance(argumentos, dict) else {}


def clave_de_argumentos(argumentos: Any) -> str:
    """Serialización canónica de los argumentos de una herramienta.

    Una sola definición para el guard de repetición del agente y para la card
    del plan en la consola: los dos cruzan pasos del plan con pasos de la traza
    comparando esta cadena. Si los separadores o el respaldo difirieran, el
    cruce de veredictos y la detección de llamadas repetidas fallarían en
    silencio.
    """
    try:
        return json.dumps(
            argumentos or {}, sort_keys=True, ensure_ascii=False, default=str
        )
    except (TypeError, ValueError):
        # Un argumento no serializable no puede tumbar el turno: se compara por
        # su representación, que sigue siendo estable para el mismo objeto.
        return repr(argumentos)


def _clave_llamada(llamada: Any) -> tuple[str, str]:
    """Identidad canónica de una llamada: nombre + argumentos serializados.

    La comparación es sobre los argumentos, no sobre el texto que redactó el
    modelo: dos llamadas con el mismo contenido lógico son la misma llamada
    aunque el modelo las escriba en otro orden de claves o con otro `id`.
    """
    return _campo_de(llamada, "name"), clave_de_argumentos(_args_de(llamada))


class _EjecutorConProgreso:
    """Envoltorio del ejecutor del plan que avisa los pasos ya ejecutados.

    `src/planificador.py` no acepta un callback por paso, así que el aviso se
    emite al terminar cada ejecución de plan, en el mismo orden de la traza.
    """

    def __init__(self, ejecutor: Any, al_paso: Any):
        self.ejecutor = ejecutor
        self.al_paso = al_paso

    def ejecutar(self, plan: Any, salidas_previas: Any = None) -> Any:
        resultado = self.ejecutor.ejecutar(plan, salidas_previas=salidas_previas)
        for paso in resultado.pasos:
            _avisar_paso(self.al_paso, paso)
        return resultado


class AgenteTecnoApoyo:
    """Ejecutor del ciclo de herramientas sobre las tools de `src.herramientas`."""

    def __init__(
        self,
        herramientas: list[BaseTool],
        llm: Any | None = None,
        max_pasos: int | None = None,
        prompt_sistema: str | None = None,
        memoria: Any | None = None,
        modo: str | None = None,
        cliente_actual: Any | None = None,
        *,
        solicitudes_por_minuto: float | None = None,
    ):
        self.memoria = memoria
        # Lector del cliente de la sesión. Es el mismo callable que reciben las
        # tools de alcance `sesion`, y su sola presencia distingue al agente de
        # sesión del global: sin él no hay atajo determinista de identificación
        # ni línea de cliente en el prompt. Se lee en cada turno, no aquí: en el
        # evaluador el cliente cambia entre tareas sin reconstruir el agente.
        self.cliente_actual = cliente_actual
        # Cuota del proveedor (S9): techo de solicitudes por minuto, con bucket
        # 1 para que no haya dos peticiones en vuelo. `None` o 0 es el
        # comportamiento de siempre (sin limitador); el evaluador pasa el valor
        # por aquí para no tocar la configuración global ni el agente de la
        # consola, que no lo necesita.
        self.solicitudes_por_minuto = (
            0.0 if solicitudes_por_minuto is None else float(solicitudes_por_minuto)
        )
        self._limitador: Any | None = None
        self.herramientas = list(herramientas)
        # La memoria no es solo un bloque de prompt: también aporta una action
        # propia (`recordar`). Se agrega solo si hay memoria y si el llamador no
        # trajo ya una tool con ese nombre.
        if memoria is not None and "recordar" not in {
            herramienta.name for herramienta in self.herramientas
        }:
            self.herramientas.append(crear_herramienta_recordar(memoria))
        self.indice = por_nombre(self.herramientas)
        # Modo de resolución: 'reactivo' es el bucle de herramientas de siempre;
        # 'planificado' pasa por src/planificador.py. Un valor no reconocido no
        # detiene la sesión: un argumento inválido cae a reactivo. La variable
        # de entorno inválida, en cambio, usa el predeterminado de config.
        elegido = (modo or config.AGENTE_MODO).strip().lower()
        self.modo = elegido if elegido in config.MODOS_AGENTE else "reactivo"
        # Un tope <= 0 dejaría al agente sin poder ejecutar ninguna herramienta.
        self.max_pasos = max(
            1, config.AGENTE_MAX_PASOS if max_pasos is None else int(max_pasos)
        )
        self.prompt_sistema = prompt_sistema or PROMPT_SISTEMA
        self.colector_tokens = ColectorTokens()
        self._llm = llm
        # Un modelo inyectado (tests) también produce los planes: no tiene
        # sentido construir un segundo cliente cuando no hay red de por medio.
        self._llm_inyectado = llm is not None
        # Cliente separado para planificar: permite usar un modelo más capaz solo
        # en el plan sin tocar el que redacta las respuestas.
        self._llm_plan = None
        # Con un modelo inyectado el agente queda armado desde el minuto uno;
        # el `ChatOpenAI` real se construye más abajo, solo al consultar, para
        # que los tests nunca toquen la red.
        self.llm_con_tools = llm.bind_tools(self.herramientas) if llm is not None else None

    # -- modelo -----------------------------------------------------------
    def _limitador_de_solicitudes(self) -> Any:
        """Limitador de la cuota del proveedor, compartido por los dos modelos.

        Se construye en la primera consulta y se cachea: uno nuevo por llamada
        no limitaría nada, porque cada uno arranca con el bucket lleno. Es el
        mismo objeto para el planificador y para el que responde, así que la
        cuota es del agente y no de cada cliente por separado.
        """
        if self.solicitudes_por_minuto <= 0:
            return None
        if self._limitador is None:
            from langchain_core.rate_limiters import InMemoryRateLimiter

            self._limitador = InMemoryRateLimiter(
                requests_per_second=self.solicitudes_por_minuto / 60.0,
                check_every_n_seconds=0.1,
                max_bucket_size=1,
            )
        return self._limitador

    def _cliente_de_chat(self, modelo: str) -> Any:
        """Cliente `ChatOpenAI` con la configuración compartida del agente.

        El que planifica y el que responde se construyen con la misma base,
        credenciales, temperatura, timeouts y limitador: un parámetro nuevo se
        agrega acá una vez y no puede quedar en una sola de las dos rutas.
        """
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=modelo,
            base_url=config.LLM_BASE_URL,
            api_key=SecretStr(config.LLM_API_KEY),
            temperature=config.TEMPERATURA,
            timeout=config.LLM_TIMEOUT_SEG,
            max_retries=config.LLM_MAX_REINTENTOS,
            stream_usage=True,
            rate_limiter=self._limitador_de_solicitudes(),
        )

    def _modelo_de_planificacion(self) -> Any:
        """Modelo que produce los planes, según `config.PLAN_MODELO`.

        Se construye aparte del modelo del agente y se cachea: el plan exige
        JSON estricto y puede convenir un modelo más capaz, pero no tiene
        sentido instanciar un cliente nuevo por consulta.
        """
        if self._llm_plan is None:
            self._llm_plan = self._cliente_de_chat(config.PLAN_MODELO)
        return self._llm_plan

    def _modelo(self) -> Any:
        """Devuelve el modelo con las tools enlazadas, construyéndolo si falta."""
        if self.llm_con_tools is None:
            self._llm = self._cliente_de_chat(config.AGENTE_MODELO)
            self.llm_con_tools = self._llm.bind_tools(self.herramientas)
        return self.llm_con_tools

    # -- ejecución de una herramienta -------------------------------------
    def _ejecutar(self, llamada: Any, numero: int) -> tuple[PasoAgente, str, object]:
        """Ejecuta una llamada sin propagar errores.

        Devuelve el paso registrado, el contenido que irá en el `ToolMessage` y
        el veredicto de evidencia (`True`, `False` o el centinela cuando la
        herramienta no opina sobre la suficiencia documental).
        """
        nombre = _campo_de(llamada, "name")
        argumentos = _args_de(llamada)

        herramienta = self.indice.get(nombre)
        if herramienta is None:
            disponibles = ", ".join(sorted(self.indice)) or "ninguna"
            mensaje = (
                f"Herramienta desconocida: '{nombre}'. "
                f"Herramientas disponibles: {disponibles}."
            )
            paso = PasoAgente(
                numero=numero,
                herramienta=nombre,
                argumentos=argumentos,
                salida=None,
                error=mensaje,
                ok=False,
            )
            return paso, _json_error(mensaje), _SIN_VEREDICTO

        inicio_paso = time.perf_counter()
        try:
            # `invoke` valida con el `args_schema` de la tool antes de ejecutar
            # (Pydantic, `extra="forbid"`): argumentos inválidos no escriben.
            salida = herramienta.invoke(argumentos)
        except Exception as exc:  # noqa: BLE001 - informar al modelo, no abortar
            mensaje = f"{type(exc).__name__}: {exc}"
            paso = PasoAgente(
                numero=numero,
                herramienta=nombre,
                argumentos=argumentos,
                salida=None,
                error=mensaje,
                ok=False,
                duracion_seg=_duracion_desde(inicio_paso),
            )
            return paso, _json_error(mensaje), _SIN_VEREDICTO

        contenido = (
            salida
            if isinstance(salida, str)
            else json.dumps(salida, ensure_ascii=False, default=str)
        )
        try:
            payload = json.loads(contenido)
        except (TypeError, ValueError):
            payload = None

        if isinstance(payload, dict):
            ok = bool(payload.get("ok", True))
            error = payload.get("error")
            datos = payload.get("datos")
        else:
            # Una tool que devuelve texto libre no trae veredicto propio.
            ok, error, datos = True, None, None

        veredicto: object = _SIN_VEREDICTO
        if (
            nombre == HERRAMIENTA_BUSQUEDA
            and ok
            and isinstance(datos, dict)
            and "evidencia_suficiente" in datos
        ):
            veredicto = bool(datos["evidencia_suficiente"])

        paso = PasoAgente(
            numero=numero,
            herramienta=nombre,
            argumentos=argumentos,
            salida=contenido,
            error=None if ok else (str(error) if error else "La herramienta falló."),
            ok=ok,
            duracion_seg=_duracion_desde(inicio_paso),
        )
        return paso, contenido, veredicto

    # -- memoria ----------------------------------------------------------
    def _historial_acotado(self) -> str:
        """`corto.texto()` recortado a `LIMITE_HISTORIAL`, conservando la cola.

        Las estrategias (buffer/ventana/resumen) ya están aplicadas dentro de
        `texto()`: aquí solo se pone el tope de tamaño. Se corta por el final
        porque los turnos más nuevos son los que sostienen un seguimiento del
        tipo "¿y en qué estado está?".
        """
        if self.memoria is None:
            return ""
        texto = (self.memoria.corto.texto() or "").strip()
        if len(texto) > LIMITE_HISTORIAL:
            texto = "…" + texto[-(LIMITE_HISTORIAL - 1):]
        return texto

    def _bloque_de_memoria(self, pregunta: str) -> str:
        """Historial reciente y hechos relevantes, en ese orden.

        Devuelve "" cuando no hay memoria, cuando no hay nada que inyectar o
        cuando leerla falla. El historial llega hasta aquí sin el turno en
        curso: `_registrar_turnos` corre después de responder, así que la
        pregunta actual no se duplica en el prompt.
        """
        if self.memoria is None:
            return ""
        partes: list[str] = []
        # Cada mitad se protege por separado: un historial ilegible no debe
        # llevarse por delante los hechos del usuario, que sí son recuperables.
        try:
            historial = self._historial_acotado()
        except Exception:  # noqa: BLE001 - sin memoria el agente sigue operando
            historial = ""
        if historial:
            partes.append(f"{ENCABEZADO_HISTORIAL}\n{historial}")
        try:
            hechos = self.memoria.bloque_para(pregunta)
        except Exception:  # noqa: BLE001 - sin memoria el agente sigue operando
            hechos = ""
        if hechos:
            partes.append(hechos)
        return "\n\n".join(partes)

    def _cliente_de_sesion(self) -> str:
        """Cliente identificado en la sesión, saneado, o "" si no hay.

        La fuente preferida es el callable que recibió el agente
        (`cliente_actual`, el mismo que leen las tools): en el evaluador es un
        holder mutable que cambia entre tareas, así que hay que leerlo en cada
        turno. Si el agente recibió un callable, su respuesta es la autoridad:
        un vacío significa «sesión sin cliente» y no se pisa con `config`. El
        respaldo por `config` solo existe para el agente que no recibió
        callable, porque la consola histórica guarda ahí su estado. Un valor
        vacío o de otro tipo cuenta como «sin cliente»: nunca se inventa uno.
        """
        if self.cliente_actual is not None:
            try:
                valor = self.cliente_actual()
            except Exception:  # noqa: BLE001 - sin cliente el turno sigue
                return ""
            return sanear_nombre_cliente(valor)
        for clave in CLAVES_CLIENTE_SESION:
            valor = getattr(config, clave, None)
            nombre = sanear_nombre_cliente(valor)
            if nombre:
                return nombre
        return ""

    def _linea_de_cliente(self) -> str:
        """Línea del cliente de la sesión para los prompts (S3).

        El agente de sesión sin cliente lo declara explícitamente: el
        planificador necesita saber que falta el nombre para pedirlo en vez de
        inventarlo. El agente global no imprime nada, porque su alcance no
        requiere cliente identificado (T05 lista casos sin cliente por diseño).
        """
        if self.cliente_actual is None:
            return ""
        return linea_de_cliente(self._cliente_de_sesion()) or (
            ETIQUETA_CLIENTE_SIN_IDENTIFICAR
        )

    def _requiere_identificacion(self, pregunta: str) -> bool:
        """S1/S2: sesión sin cliente y una consulta que necesita datos de casos.

        El atajo es exclusivo del agente de sesión (`cliente_actual` presente) y
        solo aplica mientras el cliente está vacío: con cliente fijado la
        consulta se responde normalmente (S4) y en alcance global listar casos
        no exige identificación.
        """
        if self.cliente_actual is None:
            return False
        if self._cliente_de_sesion():
            return False
        return es_consulta_de_casos(pregunta)

    def _linea_de_alcance(self) -> str:
        """Declara el alcance de las herramientas para el planificador.

        Los nombres de las tools son el alcance: `identificar_cliente` solo se
        publica en alcance de sesión, así que su ausencia significa alcance
        global, donde listar y consultar casos no exigen cliente. La línea va
        primera en el contexto para que el modelo decida si puede preguntar
        antes de leer el historial y los hechos.
        """
        if "identificar_cliente" not in self.indice:
            return ALCANCE_GLOBAL
        cliente = self._cliente_de_sesion()
        if cliente:
            return ALCANCE_SESION_CON_CLIENTE.format(cliente=cliente)
        return ALCANCE_SESION_SIN_CLIENTE

    def _contexto_de_memoria(self, pregunta: str) -> str:
        """Contexto para el planificador: alcance, historial y hechos.

        La línea de alcance se antepone al bloque de memoria porque es la única
        que nombra al cliente: el nombre no se repite en otro lugar del contexto
        y el modelo no lo confunde con un dato pendiente de preguntar.
        """
        bloque = self._bloque_de_memoria(pregunta)
        linea = self._linea_de_alcance()
        return f"{linea}\n\n{bloque}" if bloque else linea

    def _prompt_sistema_del_turno(self) -> str:
        """Prompt del sistema con la línea del cliente de la sesión.

        Se compone en cada turno y no en el constructor: el cliente puede
        cambiar sin reconstruir el agente, y el prompt tiene que mostrar el del
        turno en curso, no el del turno en que se construyó.
        """
        linea = self._linea_de_cliente()
        if not linea:
            return self.prompt_sistema
        return f"{self.prompt_sistema.rstrip()}\n\n{linea}"

    def _prompt_con_memoria(self, pregunta: str) -> str:
        """Prompt del sistema con la línea del cliente, historial y hechos."""
        bloque = self._bloque_de_memoria(pregunta)
        base = self._prompt_sistema_del_turno()
        return f"{base}\n\n{bloque}" if bloque else base

    def _registrar_turnos(self, pregunta: str, respuesta: str) -> None:
        """Persiste el turno. Una falla de memoria no tumba la conversación.

        La respuesta vacía no se guarda: en un corte por tope o por error del
        proveedor no hubo turno del asistente, y guardarla dejaría un hueco que
        la recarga leería como respuesta real.
        """
        if self.memoria is None:
            return
        try:
            self.memoria.registrar("usuario", pregunta)
            if respuesta:
                self.memoria.registrar("asistente", respuesta)
        except Exception:  # noqa: BLE001 - la memoria es accesoria, no bloqueante
            return

    # -- ciclo principal --------------------------------------------------
    def _transmitir_turno(
        self, modelo: Any, mensajes: list[Any], al_emitir: Any,
    ) -> tuple[Any, str, str | None]:
        """Un turno del bucle reactivo por `stream`, ensamblado trozo a trozo.

        Los trozos se suman con la aritmética de LangChain: el contenido de
        texto se concatena y los `tool_call_chunks` de una misma llamada se
        funden en una sola entrada con nombre, argumentos e id completos. Así el
        turno se lee igual que un `AIMessage` de `invoke` (`content` y
        `tool_calls`) y el resto del bucle no cambia. Cada trozo con texto se
        entrega al emisor tal como llega.

        Sin `config` ni callbacks: un `.stream()` directo no dispara
        `on_llm_end`, así que el uso se suma a mano desde el último trozo que lo
        traiga y la contabilidad no queda en 0/0/0.

        Devuelve el turno ensamblado (None si el stream no entregó nada), el
        texto emitido en el turno y el error del proveedor cuando el stream se
        cortó.
        """
        ensamblado: Any = None
        ultimo_con_uso: Any = None
        emitido: list[str] = []
        try:
            for trozo in modelo.stream(mensajes):
                ensamblado = trozo if ensamblado is None else ensamblado + trozo
                if getattr(trozo, "usage_metadata", None) is not None:
                    ultimo_con_uso = trozo
                texto = _texto_de(trozo, recortar=False)
                if not texto:
                    continue
                emitido.append(texto)
                al_emitir(texto)
        except Exception as exc:  # noqa: BLE001 - el corte no borra lo mostrado
            # Lo ya emitido se conserva: el usuario no puede quedarse con media
            # respuesta y sin saber qué alcanzó a llegar.
            _sumar_uso(self.colector_tokens, ultimo_con_uso)
            return ensamblado, "".join(emitido), f"{type(exc).__name__}: {exc}"
        _sumar_uso(self.colector_tokens, ultimo_con_uso)
        return ensamblado, "".join(emitido), None

    def consultar(
        self, pregunta: str, return_intermediate_steps: bool = True,
        al_paso: Any | None = None, al_emitir: Any | None = None,
        al_pensamiento: Any | None = None,
    ) -> RespuestaAgente:
        """Resuelve `pregunta` publicando el mensaje del turno para las tools.

        El mensaje del turno es el único texto del que se puede afirmar que la
        persona lo escribió: lo consulta `identificar_cliente` antes de atar la
        sesión a un nombre. La publicación cubre los dos modos y se restablece
        al terminar el turno, de modo que una mención de un turno anterior no
        sirva para identificar en el siguiente. La memoria y el historial no se
        publican como mensaje del turno a propósito: no identifican a nadie.
        """
        with mensaje_del_turno(pregunta):
            return self._ejecutar_turno(
                pregunta, return_intermediate_steps,
                al_paso=al_paso, al_emitir=al_emitir,
                al_pensamiento=al_pensamiento,
            )

    def _responder_capacidades(
        self, pregunta: str, return_intermediate_steps: bool,
        al_paso: Any | None, al_emitir: Any | None,
    ) -> RespuestaAgente:
        """Responde qué puede hacer el agente sin llamar al modelo.

        El paso se publica como `capacidades` y no como una tool del modelo: la
        traza no puede afirmar una llamada que no existió. `tokens_total` queda
        en cero por la misma razón.
        """
        inicio = time.perf_counter()
        paso = PasoAgente(
            numero=1, herramienta="capacidades", argumentos={},
            salida="resolución determinista, sin modelo", ok=True,
        )
        if al_paso is not None:
            al_paso(paso)
        if al_emitir is not None:
            al_emitir(TEXTO_CAPACIDADES)
        # No se registra en la memoria de corto plazo: el texto canónico no es
        # contexto conversacional y, si se guardara, entraría en el prompt de
        # turnos futuros. La traza y el historial de la consola sí lo registran.
        return RespuestaAgente(
            pregunta=pregunta,
            respuesta=TEXTO_CAPACIDADES,
            abstencion=False,
            pasos=[paso] if return_intermediate_steps else [],
            iteraciones=1,
            tokens_total=0,
            latencia_seg=_duracion_desde(inicio),
            error=None,
            replanes=0,
            plan=None,
        )

    def _responder_identificacion(
        self, pregunta: str, return_intermediate_steps: bool,
        al_paso: Any | None, al_emitir: Any | None,
    ) -> RespuestaAgente:
        """Pide el nombre del cliente sin llamar al modelo (S1).

        La pregunta canónica ya la conoce el dominio, así que el turno no gasta
        tokens ni depende de que el modelo elija `pedir_datos`. El paso se
        publica como `pedir_datos` con la misma forma que devolvería la tool
        —`datos.pregunta`— para que la traza, la consola y el evaluador lean el
        turno igual que uno planificado.
        """
        inicio = time.perf_counter()
        salida = json.dumps(
            {"ok": True, "datos": {"pregunta": PREGUNTA_CLIENTE}, "error": None},
            ensure_ascii=False,
        )
        paso = PasoAgente(
            numero=1, herramienta="pedir_datos",
            argumentos={"pregunta": PREGUNTA_CLIENTE},
            salida=salida, ok=True,
        )
        _avisar_paso(al_paso, paso)
        if al_emitir is not None:
            al_emitir(PREGUNTA_CLIENTE)
        # Igual que el atajo de capacidades: no se registra en la memoria de
        # corto plazo, porque la pregunta canónica no es contexto
        # conversacional y entraría en el prompt de los turnos siguientes.
        return RespuestaAgente(
            pregunta=pregunta,
            respuesta=PREGUNTA_CLIENTE,
            abstencion=False,
            pasos=[paso] if return_intermediate_steps else [],
            iteraciones=1,
            tokens_total=0,
            latencia_seg=_duracion_desde(inicio),
            error=None,
            replanes=0,
            plan=None,
        )

    def _pregunta_del_puente_de_escritura(
        self, pedido: PedidoDeCrearCaso, pregunta: str,
        return_intermediate_steps: bool,
        al_paso: Any | None, al_emitir: Any | None,
        al_pensamiento: Any | None,
    ) -> RespuestaAgente:
        """S25: pregunta determinista por el dato del usuario que falta.

        El puente solo completa `cliente` y `descripcion`; la categoría no se
        pregunta porque no es un dato del usuario: es la decisión del modelo,
        que la elige en la llamada cuando el turno se le entrega con los datos
        completos. Acá falta al menos un dato, así que el turno se cierra sin
        modelo: `tokens_total` queda en cero porque el proveedor no se invocó.
        """
        self.colector_tokens.reset()
        inicio = time.perf_counter()
        if al_pensamiento is not None:
            al_pensamiento(NOTA_PUENTE_ESCRITURA)
        # La pregunta canónica no se sintetiza: se muestra tal cual, con la
        # misma forma de salida que la tool `pedir_datos`.
        texto = pregunta_por_el_primero_que_falta(pedido.campos)
        salida = json.dumps(
            {"ok": True, "datos": {"pregunta": texto}, "error": None},
            ensure_ascii=False,
        )
        paso = PasoAgente(
            numero=1, herramienta="pedir_datos",
            argumentos={"pregunta": texto}, salida=salida, ok=True,
        )
        # El aviso va después de registrar el paso: la consola y la traza nunca
        # discrepan sobre qué se ejecutó.
        _avisar_paso(al_paso, paso)
        if al_emitir is not None:
            al_emitir(texto)
        # A diferencia del atajo de identificación, este turno sí entra a la
        # memoria: la pregunta del puente es el contexto que el turno siguiente
        # necesita para entender la respuesta.
        if self.memoria is not None:
            self._registrar_turnos(pregunta, texto)
        return RespuestaAgente(
            pregunta=pregunta,
            respuesta=texto,
            abstencion=False,
            pasos=[paso] if return_intermediate_steps else [],
            iteraciones=1,
            tokens_total=0,
            latencia_seg=_duracion_desde(inicio),
            error=None,
            replanes=0,
            plan=None,
            pensamientos=[NOTA_PUENTE_ESCRITURA],
        )

    def _ejecutar_turno(
        self, pregunta: str, return_intermediate_steps: bool = True,
        al_paso: Any | None = None, al_emitir: Any | None = None,
        al_pensamiento: Any | None = None,
    ) -> RespuestaAgente:
        """Resuelve `pregunta` con hasta `max_pasos` iteraciones de herramientas.

        `al_paso` se invoca después de cada paso ejecutado, con éxito o con
        error, para que la consola muestre el progreso mientras corre.
        `al_emitir` recibe la respuesta final por trozos en los dos modos: en el
        planificado la síntesis, en el reactivo el turno del modelo que redacta.
        Sin `al_emitir` el bucle usa `invoke` y no abre ningún stream; el
        reactivo toma esa misma ruta silenciosa en el turno que el guard de
        evidencia va a reemplazar, para no mostrar un texto que el turno
        contradice al cerrar.

        En el reactivo, un pedido de creación de caso se resuelve antes del
        bucle (S25): si el texto no trae los datos del usuario —`cliente` y
        `descripcion`— el puente pregunta el primero que falta y el modelo no se
        invoca. Con los datos completos, el turno se le entrega al modelo con
        una instrucción estrecha: la única decisión que le queda es la
        categoría.
        """
        if es_consulta_de_capacidades(pregunta):
            return self._responder_capacidades(
                pregunta, return_intermediate_steps,
                al_paso=al_paso, al_emitir=al_emitir,
            )
        if self._requiere_identificacion(pregunta):
            return self._responder_identificacion(
                pregunta, return_intermediate_steps,
                al_paso=al_paso, al_emitir=al_emitir,
            )
        if self.modo == "planificado":
            return self._consultar_planificado(
                pregunta, return_intermediate_steps,
                al_paso=al_paso, al_emitir=al_emitir,
            )
        # S25: el puente hace la mecánica antes de que el modelo exista en el
        # turno. Si el texto no trae los datos del usuario, el turno se cierra
        # con una pregunta determinista y sin proveedor. Si los trae, el turno
        # sigue al bucle con los datos ya recolectados y una instrucción que
        # deja al modelo una sola decisión: la categoría.
        pedido = detectar_pedido_de_crear_caso(pregunta)
        if pedido is not None and pedido.faltantes:
            return self._pregunta_del_puente_de_escritura(
                pedido, pregunta, return_intermediate_steps,
                al_paso=al_paso, al_emitir=al_emitir,
                al_pensamiento=al_pensamiento,
            )
        inicio = time.perf_counter()
        self.colector_tokens.reset()

        mensajes: list[Any] = [
            SystemMessage(content=self._prompt_con_memoria(pregunta)),
            HumanMessage(content=pregunta),
        ]
        # Los datos recolectados viajan como un segundo mensaje de sistema,
        # antes del pedido original: el texto de la persona no se reescribe y el
        # bloque queda fuera del historial. La categoría no está en el texto que
        # arma el arnés; la pone el modelo en su llamada.
        pensamientos: list[str] = []
        if pedido is not None:
            mensajes.insert(
                1,
                SystemMessage(
                    content=instruccion_para_el_modelo(pedido.campos)
                ),
            )
            pensamientos.append(NOTA_ENTREGA_AL_MODELO)
            if al_pensamiento is not None:
                al_pensamiento(NOTA_ENTREGA_AL_MODELO)
        modelo = self._modelo()
        configuracion = {"callbacks": [self.colector_tokens]}

        pasos: list[PasoAgente] = []
        iteraciones = 0
        respuesta = ""
        error: str | None = None
        ultima_evidencia: object = _SIN_VEREDICTO
        finalizado = False
        # Veces que se vio cada llamada canónica en este turno. El contador vive
        # aquí y no en el agente: dos turnos distintos pueden repetir la misma
        # consulta sin que eso sea un bucle.
        vistas: dict[tuple[str, str], int] = {}
        corte_por_repeticion = False

        for _ in range(self.max_pasos):
            iteraciones += 1
            # No se transmite lo que el guard puede reemplazar. El veredicto ya
            # se conoce antes de este turno: `ultima_evidencia` se actualiza con
            # cada resultado de herramienta y un caso exitoso lo limpia. Si la
            # última búsqueda quedó bajo el umbral, el turno va por la ruta
            # silenciosa: transmitir su texto dejaría en pantalla una respuesta
            # que la abstención contradice al cerrar. El modelo se llama igual,
            # así que tokens y traza siguen midiéndose.
            transmitir = al_emitir is not None and ultima_evidencia is not False
            if not transmitir:
                # Ruta silenciosa (evaluación, o turno que el guard reemplaza):
                # `invoke` con los callbacks, tal como siempre. No abre stream
                # ni construye emisores.
                try:
                    mensaje = modelo.invoke(mensajes, config=configuracion)
                except Exception as exc:  # noqa: BLE001 - el REPL no debe morir
                    # Mismo criterio que el motor RAG: un 401/429/timeout es un
                    # error del proveedor, no una abstención ni un cierre.
                    error = f"{type(exc).__name__}: {exc}"
                    break
            else:
                mensaje, parcial, fallo = self._transmitir_turno(
                    modelo, mensajes, al_emitir,
                )
                if fallo is not None:
                    # El turno se reporta como error, no como abstención, y el
                    # texto ya visible se devuelve como respuesta parcial.
                    respuesta = parcial.strip()
                    error = fallo
                    break
                if mensaje is None:
                    # Un stream sin trozos no dejó turno que leer.
                    error = MENSAJE_SIN_RESPUESTA
                    break
            mensajes.append(mensaje)

            llamadas = list(getattr(mensaje, "tool_calls", None) or [])
            if not llamadas:
                # Sin llamadas pendientes, este texto es la respuesta final.
                respuesta = _texto_de(mensaje)
                if respuesta:
                    finalizado = True
                else:
                    # Una respuesta vacía no es un éxito: no dice nada.
                    error = MENSAJE_SIN_RESPUESTA
                break

            # Lo que el modelo dijo antes de actuar es su razonamiento visible.
            # En streaming ya se transmitió; acá se registra igual para la traza
            # y para la ruta silenciosa, que no lo muestra.
            for fuente in (_texto_de(mensaje), _razonamiento_de(mensaje)):
                if fuente:
                    pensamientos.append(fuente)
                    if al_pensamiento is not None:
                        al_pensamiento(fuente)

            for llamada in llamadas:
                clave = _clave_llamada(llamada)
                vistas[clave] = vistas.get(clave, 0) + 1
                veces = vistas[clave]
                # Solo las herramientas que existen pueden estar "ya ejecutadas":
                # repetir un nombre inexistente sigue reportándose como
                # herramienta desconocida, que es el error útil en ese caso.
                if veces >= 2 and clave[0] in self.indice:
                    if veces >= 3:
                        error = ERROR_REPETICION.format(herramienta=clave[0])
                        corte_por_repeticion = True
                        break
                    aviso = MENSAJE_REPETICION.format(herramienta=clave[0])
                    paso = PasoAgente(
                        numero=len(pasos) + 1,
                        herramienta=clave[0],
                        argumentos=_args_de(llamada),
                        salida=None,
                        error=aviso,
                        ok=False,
                    )
                    pasos.append(paso)
                    mensajes.append(
                        ToolMessage(
                            content=_json_error(aviso),
                            tool_call_id=(
                                _campo_de(llamada, "id") or f"call_{paso.numero}"
                            ),
                        )
                    )
                    continue
                paso, contenido, veredicto = self._ejecutar(llamada, len(pasos) + 1)
                pasos.append(paso)
                # El aviso va después de registrar el paso: la consola y la
                # traza nunca discrepan sobre qué se ejecutó.
                _avisar_paso(al_paso, paso)
                if veredicto is not _SIN_VEREDICTO:
                    ultima_evidencia = veredicto
                elif paso.ok and paso.herramienta in HERRAMIENTAS_CASO:
                    # El caso trajo datos: el veredicto documental anterior ya
                    # no decide la respuesta.
                    ultima_evidencia = _SIN_VEREDICTO
                mensajes.append(
                    ToolMessage(
                        content=contenido,
                        tool_call_id=_campo_de(llamada, "id") or f"call_{paso.numero}",
                    )
                )

            if corte_por_repeticion:
                break

        if not finalizado and error is None:
            # Tope duro: se corta sin respuesta inventada.
            respuesta = ""
            error = (
                f"Se alcanzó el tope de {self.max_pasos} pasos del agente "
                "sin una respuesta final del modelo."
            )

        if error is not None:
            # Un fallo del proveedor, una respuesta vacía o un corte por
            # repetición no son abstención: marcarlas así falsearía las
            # métricas de abstención.
            abstencion = False
        else:
            abstencion = respuesta.startswith(PREFIJO_ABSTENCION)
            if ultima_evidencia is False:
                # Guard duro: la última búsqueda no alcanzó el umbral del RAG,
                # así que la abstención se impone sobre lo que diga el modelo.
                respuesta = FRASE_ABSTENCION
                abstencion = True

        if self.memoria is not None:
            self._registrar_turnos(pregunta, respuesta)

        return RespuestaAgente(
            pregunta=pregunta,
            respuesta=respuesta,
            abstencion=abstencion,
            pasos=pasos if return_intermediate_steps else [],
            iteraciones=iteraciones,
            tokens_total=self.colector_tokens.total_tokens,
            latencia_seg=_duracion_desde(inicio),
            error=error,
            pensamientos=pensamientos,
        )

    # -- modo planificado --------------------------------------------------
    def _consultar_planificado(
        self, pregunta: str, return_intermediate_steps: bool = True,
        al_paso: Any | None = None, al_emitir: Any | None = None,
    ) -> RespuestaAgente:
        """Resuelve con planificación explícita y síntesis sobre los resultados.

        La diferencia con el modo reactivo no es cosmética: aquí el modelo
        produce un plan validable antes de tocar ninguna herramienta, y la
        respuesta final se redacta sobre lo que esas herramientas devolvieron.
        """
        from src.planificador import (
            EjecutorPlan,
            OrquestadorPlanificado,
            Planificador,
        )

        inicio = time.perf_counter()
        self.colector_tokens.reset()

        modelo = self._llm
        if modelo is None:
            self._modelo()
            modelo = self._llm
        modelo_plan = modelo if self._llm_inyectado else self._modelo_de_planificacion()

        pasos: list[PasoAgente] = []
        plan = None
        replanes = 0
        error: str | None = None
        respuesta = ""
        evidencia_deficiente = False
        diferir = False
        # La pregunta de `pedir_datos` no es una síntesis: es un texto fijo que
        # el plan ordenó mostrar. El guard del listado solo corrige lo que
        # redactó el modelo, nunca la pregunta que el usuario debe recibir
        # literal (podría empezar con "No hay..." sin afirmar nada).
        desde_pedir_datos = False

        try:
            # El plan se acota con PLAN_MAX_PASOS, no con AGENTE_MAX_PASOS: el
            # primero gobierna el plan y el segundo el bucle reactivo. Pasar el
            # tope del bucle dejaba PLAN_MAX_PASOS sin efecto real.
            planificador = Planificador(
                modelo_plan, self.herramientas,
                # Sin los callbacks, el consumo del modo planificado se
                # subcuenta: el planificador es quien hace la mayoría de las
                # llamadas al modelo en ese modo.
                callbacks=[self.colector_tokens],
                # La línea del cliente viaja al prompt del planificador: sin
                # ella el plan pedía identificar a un cliente ya identificado
                # (S5) y proponía pedir_datos para una pregunta de lectura (S4).
                linea_cliente=self._linea_de_cliente(),
            )
            ejecutor: Any = EjecutorPlan(self.herramientas)
            # El aviso por paso se envuelve aquí: el planificador ejecuta el
            # plan completo y no acepta un callback propio.
            if al_paso is not None:
                ejecutor = _EjecutorConProgreso(ejecutor, al_paso)
            orquestador = OrquestadorPlanificado(planificador, ejecutor)
            resultado = orquestador.resolver(
                pregunta, self._contexto_de_memoria(pregunta)
            )
        except Exception as exc:  # noqa: BLE001 - la sesión no debe caer
            error = f"{type(exc).__name__}: {exc}"
        else:
            pasos = list(resultado.pasos)
            plan = resultado.plan
            replanes = resultado.replanes
            evidencia_deficiente = bool(resultado.evidencia_deficiente)
            error = resultado.error
            if error is None:
                # `pedir_datos` no se sintetiza: la pregunta se muestra tal cual
                # y el turno no gasta una segunda llamada al modelo.
                pregunta_pedida = _pregunta_de_pedir_datos(pasos)
                if pregunta_pedida is not None:
                    respuesta = pregunta_pedida
                    desde_pedir_datos = True
                else:
                    listado = ultimo_listado_exitoso(pasos)
                    # La emisión se difiere cuando la respuesta del turno puede
                    # dejar de ser la síntesis: un listado exitoso con casos
                    # (guard del listado) o evidencia documental deficiente
                    # (guard de abstención). Sin diferir, la consola muestra un
                    # texto que el turno contradice al cerrar.
                    diferir = al_emitir is not None and (
                        evidencia_deficiente
                        or (listado is not None and listado[0] > 0)
                    )
                    respuesta, error = self._sintetizar(
                        modelo, pregunta, resultado, al_emitir=al_emitir,
                        diferir=diferir,
                    )

        if error is None and respuesta and not desde_pedir_datos:
            # S1/S2: la síntesis puede negar lo que un `listar_casos` exitoso
            # acaba de devolver. El reemplazo es determinista: no hay otra
            # llamada al modelo.
            respuesta = corregir_sintesis_de_listado(pasos, respuesta)

        abstencion = False
        if error is None:
            if evidencia_deficiente:
                # Mismo guard que el modo reactivo: sin evidencia documental no
                # se responde, aunque la síntesis haya redactado algo.
                respuesta = FRASE_ABSTENCION
                abstencion = True
            elif respuesta.startswith(PREFIJO_ABSTENCION):
                abstencion = True

        # La emisión va después de la decisión: lo que la consola muestra no
        # puede ser un texto que el turno reemplace después. Solo se emite aquí
        # lo retenido; el resto ya se transmitió trozo a trozo.
        if al_emitir is not None and error is None and respuesta and (
            diferir or desde_pedir_datos
        ):
            al_emitir(respuesta)

        if self.memoria is not None:
            self._registrar_turnos(pregunta, respuesta)

        return RespuestaAgente(
            pregunta=pregunta,
            respuesta=respuesta,
            abstencion=abstencion,
            pasos=pasos if return_intermediate_steps else [],
            iteraciones=len(pasos),
            tokens_total=self.colector_tokens.total_tokens,
            latencia_seg=_duracion_desde(inicio),
            error=error,
            replanes=replanes,
            plan=plan,
        )

    def _sintetizar(self, modelo: Any, pregunta: str, resultado: Any,
                    al_emitir: Any | None = None,
                    diferir: bool = False) -> tuple[str, str | None]:
        """Redacta la respuesta final a partir de las salidas de las tools.

        Sin `al_emitir` usa `invoke` y devuelve el texto completo: es la ruta
        de evaluación, que debe ser silenciosa y no abrir un stream. Con
        `al_emitir`, itera `modelo.stream()` y entrega cada trozo con contenido;
        el uso de tokens se suma a mano desde el último trozo porque un stream
        directo no dispara los callbacks de LangChain y la contabilidad
        quedaría en 0/0/0.

        Un plan vacío no tiene salidas que sintetizar: usa
        `PROMPT_CONVERSACION` en lugar de `PROMPT_SINTESIS` para no responder
        con la abstención por defecto ante una consulta conversacional.

        `diferir` retiene los trozos hasta el final: el llamador aplica los
        guards —el del listado y el de abstención— y emite el texto definitivo
        una sola vez. Es la única forma de que la consola no muestre una
        contradicción que luego se corrige.
        """
        lineas = []
        for paso in resultado.pasos:
            salida = paso.salida if paso.ok else (paso.error or "sin salida")
            # Tope por paso: un payload enorme desplazaría al objetivo del
            # prompt y encarecería la síntesis sin aportar.
            lineas.append(f"- {paso.herramienta}: {str(salida)[:800]}")
        if lineas:
            instruccion = PROMPT_SINTESIS.format(
                objetivo=pregunta, pasos="\n".join(lineas),
                abstencion=FRASE_ABSTENCION,
            )
        else:
            instruccion = PROMPT_CONVERSACION.format(
                objetivo=pregunta, abstencion=FRASE_ABSTENCION
            )
        mensajes = [
            SystemMessage(content=self._prompt_con_memoria(pregunta)),
            HumanMessage(content=instruccion),
        ]

        if al_emitir is None:
            try:
                mensaje = modelo.invoke(
                    mensajes, config={"callbacks": [self.colector_tokens]}
                )
            except Exception as exc:  # noqa: BLE001 - error del proveedor
                return "", f"{type(exc).__name__}: {exc}"
            texto = _texto_de(mensaje)
            if not texto:
                return "", MENSAJE_SIN_RESPUESTA
            return texto, None

        partes: list[str] = []
        ultimo: Any = None
        try:
            # Sin `config` ni callbacks: el uso se contabiliza abajo.
            for trozo in modelo.stream(mensajes):
                ultimo = trozo
                texto_trozo = _texto_de(trozo, recortar=False)
                if not texto_trozo:
                    continue
                partes.append(texto_trozo)
                if not diferir:
                    al_emitir(texto_trozo)
        except Exception as exc:  # noqa: BLE001 - el corte no borra lo mostrado
            # El texto ya visible se conserva: el usuario no puede quedarse con
            # media respuesta y sin saber qué alcanzó a llegar. Lo diferido
            # tampoco se pierde: se emite ahora, porque no habrá una segunda
            # oportunidad de mostrarlo.
            _sumar_uso(self.colector_tokens, ultimo)
            parcial = "".join(partes).strip()
            if diferir and parcial:
                al_emitir(parcial)
            return parcial, f"{type(exc).__name__}: {exc}"
        _sumar_uso(self.colector_tokens, ultimo)
        texto = "".join(partes).strip()
        if not texto:
            return "", MENSAJE_SIN_RESPUESTA
        return texto, None
