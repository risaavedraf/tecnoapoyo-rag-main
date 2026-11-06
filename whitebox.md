# Declaración de uso de inteligencia artificial

Este documento declara de forma transparente cómo se utilizó inteligencia artificial durante el desarrollo y revisión del asistente RAG de TecnoApoyo. La IA se empleó como apoyo técnico; las decisiones finales, la configuración de credenciales y la responsabilidad académica corresponden al equipo humano.

## Herramientas y modelos

| Herramienta o modelo | Uso declarado |
|---|---|
| OpenCode | Entorno de trabajo y coordinación de agentes auxiliares. |
| Gentle AI | Asistente de análisis, implementación acotada y revisión de resultados. |
| GPT-6 Luna | Modelo principal en esta sesión de trabajo. |
| Space Bunny (stealth model free) | Segundo modelo utilizado en esta sesión de trabajo, |

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
| Agente con *function calling* | `src/agente.py` | `AgenteTecnoApoyo` con `bind_tools`, tope `AGENTE_MAX_PASOS`, `ToolMessage` por paso, errores de herramienta contenidos, guard de abstención y despacho al modo planificado. |
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
| `tests/test_agente.py` | Ejecución de tools, `ToolMessage`, tope de pasos, guard de abstención, errores de herramienta y del proveedor, respuesta vacía, pasos intermedios. |
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

- **La evaluación existente mide el motor RAG, no al agente.**
  `scripts/evaluate.py` llama a `RAGEngine.consultar()` sobre 41 consultas; no
  evalúa selección de herramientas, calidad de argumentos, escrituras
  confirmadas ni planes. No existe un dataset de evaluación del agente en el
  repositorio.
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
- **Esta redacción no reejecutó la suite.** La sección describe la verificación
  registrada en el repositorio y en el historial de trabajo, no una corrida
  nueva de `pytest` durante la escritura de este documento.

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

