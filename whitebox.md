# Declaración de uso de inteligencia artificial

Este documento declara de forma transparente cómo se utilizó inteligencia artificial durante el desarrollo y revisión del asistente RAG de TecnoApoyo. La IA se empleó como apoyo técnico; las decisiones finales, la configuración de credenciales y la responsabilidad académica corresponden al equipo humano.

## Herramientas y modelos

| Herramienta o modelo | Uso declarado |
|---|---|
| OpenCode | Entorno de trabajo y coordinación de agentes auxiliares. |
| Gentle AI | Asistente de análisis, implementación acotada y revisión de resultados. |
| GPT-6 Luna | Modelo principal en esta sesión de trabajo. |
| Space Bunny (stealth model free) | Segundo modelo utilizado en esta sesión de trabajo, |
| Gentle Shell (Pi) | Entorno de trabajo actual: consola de agente, ejecución de comandos, edición de archivos y verificación sobre el repositorio. |
| DeepSeek V4.1 Flash (`deepseek-v4.1-flash`) | Modelo del entorno de trabajo en la etapa RA2 y en esta actualización del documento. |

Desde la etapa RA2 el trabajo se realiza en Gentle Shell con `deepseek-v4.1-flash`
(proveedor `opencode-go`). Ese modelo pertenece al entorno de desarrollo y no al
producto: el asistente en ejecución usa `ministral-8b-latest` para chat y
`mistral-embed` para embeddings, ambos configurables por variables de entorno. La
distinción importa porque los resultados de evaluación reportados en este
documento se obtuvieron con el proveedor del producto, no con el modelo del
entorno de trabajo.

Los agentes auxiliares de OpenCode colaboraron en exploración y verificación. No se atribuyen a un modelo subyacente concreto cuando esa identidad no está disponible en el registro.

## Trabajo apoyado por IA

- Revisión estática del código del asistente y análisis de los notebooks y materiales de estudio de RA1.
- Identificación y corrección de problemas en la configuración del proveedor, el manejo de errores, la evaluación, la trazabilidad de consultas y la exportación de sesiones.
- Análisis de los resultados de evaluación ejecutados por el equipo con la API de Mistral.
- Apoyo de Gentle AI/OpenCode en la actualización del `README.md` para documentar la configuración con Mistral, la instalación y los comandos, el flujo de ejecución y evaluación, los costos y sus limitaciones, y las mejoras futuras priorizadas.
- Actualización de `informe/presentacion.html` para reflejar el perfil Mistral documentado y los resultados medidos, aclarar que el umbral 0,45 no se activó y que hubo 2 falsos positivos y 2 falsos negativos, y ampliar el diseño a 1600 px con ajustes responsivos.
- Edición de una copia de trabajo del informe técnico sobre la migración a Mistral y sus resultados; el informe original se preservó.
- Preparación de esta declaración de uso de IA.

## Participación y validación humana

El equipo configuró localmente las variables del proveedor, regeneró el corpus y el índice FAISS, y ejecutó la evaluación con sus propias credenciales de Mistral. La IA no incorpora claves ni otros valores secretos en este documento. Los cambios de código se verificaron sintácticamente; la ejecución de la aplicación y de la evaluación con la API fue realizada por el equipo.

La presentación se validó estructuralmente: el HTML parsea correctamente, conserva sus 14 diapositivas y navegación, y declara ajustes de ancho y diseño responsivo. No se hizo una comprobación visual en un navegador.

Durante el análisis se produjo una interpretación incorrecta de un posible barrido del umbral de abstención. Al contrastarla con los puntajes registrados, se detectó y corrigió: los datos disponibles demostraban que el umbral no se activó en esa ejecución. La corrección se reflejó en la copia del informe. Este caso muestra por qué las conclusiones generadas por IA deben contrastarse con la evidencia original.

## Nota de Gentle AI

> Fue un gusto ayudar a revisar el proyecto, encontrar errores y convertir los resultados en una explicación más clara. También me equivoqué en una inferencia y tuve que corregirla al volver a los datos: esa revisión forma parte del trabajo, no se debe ocultar. La IA puede acelerar el análisis, pero el criterio, la verificación y la autoría final siguen siendo del equipo humano. ¡Éxito con la entrega y la presentación!

## RA2: capa de agentes, memoria y planificación

Esta sección documenta el trabajo de RA2 sobre el asistente existente: qué se
construyó, cómo se verificó y qué queda sin verificar. No reemplaza las
secciones anteriores; las complementa.

### Qué se construyó

| Componente | Archivo | Contenido |
|---|---|---|
| Agente con *function calling* | `src/agente.py` | `AgenteTecnoApoyo` con `bind_tools`, tope `AGENTE_MAX_PASOS`, `ToolMessage` por paso, errores de herramienta contenidos, guard de abstención, despacho al modo planificado y, en el reactivo, puente determinista del pedido de creación de caso: pregunta lo que el texto no trae antes de invocar al modelo y, con los datos completos, le entrega el turno para que elija la categoría. |
| Puente determinista de escritura | `src/puente_escritura.py` | Puente del modo reactivo: lista enumerativa de verbos de creación de caso en español, extracción de los dos datos que solo aporta la persona —el nombre detrás de «el cliente», «la clienta», «para <Nombre>» o «a nombre de <Nombre>», y la falla en prosa detrás de «por», «porque», «:» o la cláusula que sigue al nombre—, pregunta determinista por el primero que falta sin llamar al modelo y entrega de los datos al modelo con la instrucción de emitir `crear_caso` y elegir la categoría, que el puente no decide. Módulo puro: no importa LangChain ni ningún proveedor. |
| Herramientas y contratos | `src/herramientas.py` | Ocho tools de consulta, escritura y razonamiento, más `recordar` cuando hay memoria; contrato `ResultadoHerramienta`; `Confirmador` inyectado; `calcular_plazo` determinista. |
| Casos persistidos | `src/casos.py` | `RepositorioCasos` sobre SQLite con allowlist de estados y prioridades, ids estables, siembra idempotente y `ErrorCaso`. |
| Memoria | `src/memoria.py` | Corto plazo con `buffer`, `ventana` y `resumen`; largo plazo con hechos, sesiones y turnos; resumidor LLM con respaldo local. |
| Planificación | `src/planificador.py` | Contrato `Plan`/`PasoPlan`, `validar_plan`, `EjecutorPlan` con orden topológico y placeholders, `OrquestadorPlanificado` con presupuesto y re-plan, `ErrorPlan`. |
| Consola | `src/app.py` | Comandos `/modo`, `/memoria`, `/sesiones`, `/cargar`, `/guardar`, `/stats`, `/trazas` y `/trazo`; confirmación `s/n` de escrituras; tabla de pasos y fuentes. |
| Observabilidad | `src/observabilidad.py` | Traza por turno en JSONL con archivo diario, saneado recursivo de claves sensibles, lectura tolerante, agregados para `/stats` y extracción de fuentes deduplicadas. |
| Comparación de modos | `scripts/comparar_modos.py` | Ejecuta las mismas consultas en reactivo y planificado y guarda pasos, re-planes, tokens y latencia. |
| Configuración | `src/config.py` | Variables de la capa de agentes con default y marcas de valor inválido. |

La regla de diseño fue exponer el motor RAG como herramienta, no reescribirlo:
`buscar_documentacion` llama a `RAGEngine.recuperar()` y el veredicto de
evidencia usa el mismo `UMBRAL_ABSTENCION`. La frase canónica de abstención es
una sola en todo el asistente.

### Cómo se verificó

**Pruebas sin red.** La suite se ejecuta con `python -m pytest -q` desde la raíz
y no llama al proveedor. Los tests del agente y del planificador inyectan un
`ModeloFalso` con guion y un `MotorFalso`; los de memoria y casos usan bases
SQLite en `tmp_path`. La cobertura por módulo es:

| Archivo | Qué verifica |
|---|---|
| `tests/test_herramientas.py` | Contratos de las tools, validación Pydantic, rechazo de campos extra, denegación de escrituras sin confirmador, bordes de `calcular_plazo`, salida JSON serializable. |
| `tests/test_agente.py` | Ejecución de tools, `ToolMessage`, tope de pasos, guard de abstención, errores de herramienta y del proveedor, respuesta vacía, pasos intermedios, y los dos turnos del puente de escritura: la pregunta sin llamada al modelo y el turno entregado, con la nota del arnés en la traza y la categoría que devolvió el modelo. |
| `tests/test_puente_escritura.py` | Frontera declarada del puente (lo que captura y lo que deja al modelo), extracción del nombre y de la falla en prosa, campos que no se inventan, pregunta por el primer campo faltante, categoría que no se extrae ni viaja en la entrega al modelo, y contrato contra `ArgsCrearCaso`, que la sigue exigiendo. |
| `tests/test_casos.py` | Siembra idempotente, ids secuenciales, persistencia tras reinicio, estados y prioridades inválidos. |
| `tests/test_memoria.py` | Las tres estrategias de corto plazo, hechos que sobreviven al reinicio, sesiones y recarga. |
| `tests/test_agente_memoria.py` | Bloque de memoria en el prompt, tool `recordar`, turnos persistidos, falla de memoria no bloqueante. |
| `tests/test_planificador.py` | Validación determinista, placeholders, orden topológico, re-plan, presupuesto y evidencia deficiente. |
| `tests/test_agente_planificado.py` | Plan y síntesis, encadenamiento por placeholder, abstención forzada, plan inválido sin ejecución, error del proveedor y re-plan. |
| `tests/test_rag_abstencion.py` | Regresión del motor: abstención por umbral, umbral explícito, error del proveedor y `TOP_K`. |
| `tests/test_observabilidad.py` | Escritura y lectura de trazas, archivo diario, saneado de claves sensibles, línea corrupta, límite de lectura y agregados. |
| `tests/test_observabilidad_agente.py` | Armado de la traza desde una respuesta del agente, serialización del plan, fuentes deduplicadas y fallo de escritura que no rompe el turno. |

**Sonda real de tool calling.** Antes de apoyarse en el modelo real se hizo una
llamada con `AGENTE_MODELO=ministral-8b-latest` (registrada en
`odd/tasks/ra2-agente.md`, L9): llegaron dos `tool_calls` nativas,
`calcular_plazo` y `listar_casos`, con argumentos correctos. El soporte de
*function calling* del proveedor quedó confirmado, no inferido.

**Verificación de consola y trazas.** La consola muestra por turno los pasos
ejecutados, su estado `ok`, los errores, las fuentes documentales con
relevancia, el modo, los re-planes, los tokens y la latencia. Cada turno se
escribe como una línea JSON en `data/output/trazas/trazas_AAAA-MM-DD.jsonl`.
`/trazas` lee las últimas 10 y `/stats` agrega las trazas persistidas:
consultas, abstenciones, pasos, herramientas y modos.

### Límites de verificación

- **El motor y el agente se miden con instrumentos distintos.**
  `scripts/evaluate.py` llama a `RAGEngine.consultar()` sobre las 41 consultas
  de `evaluation/dataset_evaluacion.json`; no evalúa selección de
  herramientas, calidad de argumentos, escrituras confirmadas ni planes. El
  agente se mide con `scripts/evaluar_agente.py` sobre las 23 tareas de
  `evaluation/dataset_agente.json`, y esa corrida requiere red y una clave del
  proveedor, así que sus resultados no se reproducen sin costo.
- **El puente determinista de escritura no está medido.** No se ha ejecutado
  ninguna corrida contra el proveedor desde que el puente existe, así que su
  efecto sobre los puntajes de la evaluación es desconocido. Lo verificado son
  sus contratos sin red: la detección por su lista declarada, la extracción del
  nombre y de la falla en prosa, las preguntas por el dato faltante y que ese
  turno de pregunta no invoca al modelo. El turno con los datos completos sí
  invoca al modelo y solo se prueba con dobles, así que el acierto de la
  categoría y la obediencia a la instrucción de entrega no están medidos.
  Además, la expectativa de T21 cambió en `d13237a` y su `pregunta_clave` quedó
  en `descrip` en `8cb791b`: su puntaje ya no es comparable con las corridas del
  9 de noviembre de 2026 y ninguna comparación futura puede tratarlo como si
  midiera lo mismo.
- **La comparación de modos no se ejecutó.** `scripts/comparar_modos.py`
  requiere red e índice FAISS, y no hay archivo
  `evaluation/results/comparacion_modos_*.json` en el repositorio. La capacidad
  está implementada y medida por contrato, pero no hay resultado empírico.
- **La sonda de tool calling fue una sola llamada.** Confirma que el proveedor
  emite llamadas nativas, no que la selección de herramienta sea estable en
  todas las consultas.
- **Los tests usan dobles.** Verifican el ejecutor, la validación y los guards,
  no el comportamiento de un modelo real. Un modelo puede elegir una
  herramienta válida pero equivocada y los tests no lo detectarían.
- **La salida de consola no tiene una prueba de interfaz.** Los tests verifican
  el armado de la traza, el saneado, la lectura y los agregados, no el texto
  que la consola imprime ni la interacción con `rich`.
- **Esta redacción no reejecutó la verificación contra el proveedor.** La
  sección describe la verificación registrada en el repositorio y en el
  historial de trabajo; la suite se ejecutó al cerrar esta actualización
  (908 passed, 1 skipped), pero ninguna corrida de evaluación se repitió.

### Riesgos abiertos

- El umbral de abstención no está calibrado: en la corrida documentada los
  puntajes superiores quedaron entre `0.5746` y `0.8673` y ninguno activó el
  corte de `0.45`.
- `calcular_plazo` no considera feriados.
- La memoria de largo plazo selecciona hechos por coincidencia léxica, no
  semántica.
- El resumidor local es un respaldo que concatena y recorta; no condensa.
- Cada `/guardar` crea una sesión nueva con el historial acumulado, sin
  actualizar la anterior.
- El puente de escritura es una lista cerrada: un sinónimo no enumerado
  (`genera`, `añade`, `carga`), un `ticket`, otro idioma, un pedido de lectura,
  una pregunta de procedimiento o un turno con más de una acción vuelven al
  modelo, que es el camino anterior. La extracción también es cerrada: solo
  reconoce los introductores declarados —«el cliente», «la clienta», «para
  <Nombre>», «a nombre de <Nombre>»— y una falla que quepa en el contrato de
  `descripcion`; lo que no reconoce lo pregunta en vez de rellenarlo, así que un
  texto que trae el dato en otra forma recibe una pregunta de más.
- Con los datos del usuario completos, el turno vuelve al modelo y depende de
  que siga la instrucción de entrega: el arnés no fija la herramienta ni
  inspecciona la llamada antes de ejecutarla, así que si el modelo elige otra
  herramienta o no elige ninguna, el turno no se corrige. La categoría es del
  modelo y el arnés no la revisa más allá de lo que exige `ArgsCrearCaso`.
- T21 dejó de medir la propuesta bajo una orden que pide escribir sin
  confirmación: ahora mide la pregunta por la descripción que falta. El caso
  adversarial de propuesta ya no tiene cobertura en el dataset.

### Entorno de trabajo de esta etapa

La sección de RA2 se elaboró en Gentle Shell con `deepseek-v4.1-flash`
(proveedor `opencode-go`): lectura del repositorio para localizar los contratos
existentes, redacción de esta sección, revisión de consistencia entre el README,
los informes y el código, y verificación estructural del HTML de la presentación
(parseo correcto, número de diapositivas y navegación).

Ese entorno no ejecuta el asistente ni reemplaza las comprobaciones del equipo:
la suite de pruebas, las corridas de evaluación y el uso de la API de Mistral los
realiza el equipo con sus propias credenciales. Lo que aquí se declara como
medido proviene del registro del repositorio, no de una corrida hecha por el
modelo del entorno.

