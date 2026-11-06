# Asistente documental de TecnoApoyo (RAG)

Proyecto EP1 - Ingeniería de Soluciones con IA (ISY0101). Asistente de atención
al cliente para la pyme ficticia **TecnoApoyo**, dedicada a la reparación de
computadores. Recupera información de la documentación de la empresa y se
abstiene cuando no encuentra evidencia suficiente.

La configuración documentada usa la API de Mistral para chat
(`ministral-8b-latest`) y embeddings (`mistral-embed`). El proveedor, el endpoint
y los modelos se pueden configurar mediante variables de entorno.

## Requisitos

- Python 3.10 o superior.
- Windows PowerShell para los comandos de instalación de este ejemplo.
- Acceso de red y una clave válida de Mistral para construir el índice, usar la
  consola y ejecutar la evaluación. La generación del corpus es local y no llama
  al proveedor.

## Instalación y configuración

Desde la raíz del proyecto, crea y activa un entorno virtual e instala las
dependencias:

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

El ejemplo crea `.venv`; si el entorno ya existe con otro nombre, reemplaza
`.venv` en la ruta de activación por el nombre real de esa carpeta. Por ejemplo,
para un entorno llamado `venv`, usa `.\venv\Scripts\Activate.ps1`.

Crea o edita `.env` en la raíz del proyecto con esta plantilla. Reemplaza el
marcador por tu clave local; no compartas ni confirmes en Git el valor real:

```dotenv
OPENAI_API_KEY=
LANGCHAIN_TRACING_V2=
LANGCHAIN_API_KEY=
LANGCHAIN_PROJECT=tecnoapoyo-rag
LLM_BASE_URL=
LLM_API_KEY=
LLM_MODEL="ministral-8b-latest"
LLM_MODEL_SMALL="ministral-8b-latest"
EMBEDDING_MODELO="mistral-embed"
```

`LLM_MODELO` también configura el modelo de chat y tiene prioridad si se definen
tanto `LLM_MODELO` como `LLM_MODEL`. El ejemplo usa `LLM_MODEL` por
compatibilidad. Para reproducir la configuración Mistral no omitas `.env`: el
proveedor y los modelos se leen desde estas variables.

## Flujo de ejecución

Ejecuta los pasos desde la raíz del proyecto y en este orden:

1. Genera el corpus y el dataset de evaluación. Es un paso local, sin llamadas a
   la API:

   ```powershell
   python scripts/generar_corpus.py
   ```

   Salida esperada: 10 procedimientos, 30 FAQ y 41 consultas (31 respondibles y
   10 de abstención), además de `data/entidades.json`.

2. Construye el índice FAISS. Este paso solicita embeddings y ejecuta una
   consulta de prueba al modelo de chat; puede generar cargos reales de API:

   ```powershell
   python scripts/build_index.py
   ```

   Se guardan el índice y sus metadatos en `data/output/`, junto con el reporte
   de comparación de chunking. En la ejecución documentada se seleccionaron
   chunks de 200 caracteres con overlap de 20 y se generaron 78 chunks. Debes
   reconstruir el índice si cambia `EMBEDDING_MODELO` o el corpus.

3. Inicia la aplicación de consola. La invocación como módulo debe ejecutarse
   desde la raíz y requiere el índice del paso anterior:

   ```powershell
   python -m src.app
   ```

   Cada consulta puede llamar a Mistral y generar cargos. Escribe `/salir` para
   terminar la sesión antes de continuar.

4. Ejecuta la evaluación completa de 41 consultas. Llama al proveedor y puede
   generar cargos reales de API; escribe los resultados con fecha y hora en
   `evaluation/results/` como archivos `.json` y `.txt`:

   ```powershell
   python scripts/evaluate.py
   ```

## Consola interactiva

| Comando | Acción |
|---|---|
| `/ayuda` | Muestra los comandos disponibles. |
| `/trazo on` | Muestra los documentos recuperados y sus puntajes. |
| `/trazo off` | Oculta la trazabilidad de los documentos recuperados. |
| `/guardar [nombre]` | Exporta la sesión a `data/output/sesiones/`. |
| `/stats` | Resume el número de consultas y abstenciones de la sesión. |
| `/salir` | Termina la aplicación. |

Para una prueba rápida, consulta `¿Cuánto cuesta el diagnóstico básico?`, que
está dentro del alcance documentado. Como prueba fuera de alcance, usa
`¿Cuál es el estado de mi orden de servicio 4590?`; la aplicación no consulta
estados de órdenes y debería abstenerse.

## Índice y abstención

- El índice debe construirse y consultarse con el mismo modelo de embeddings.
  En esta configuración, `mistral-embed` debe mantenerse tanto al crear el
  índice como al consultar; reconstruye el índice si cambia `EMBEDDING_MODELO` o
  el corpus.
- `TOP_K` controla cuántos resultados se recuperan (valor usado: `3`).
- `UMBRAL_ABSTENCION` es configurable (valor usado: `0.45`). El conjunto
  evaluado no validó ese corte: los puntajes superiores registrados estuvieron
  entre `0.5746` y `0.8673`, ninguno quedó bajo `0.45` y las 41 consultas
  consumieron tokens. Por ello, el filtro de similitud no se activó en esa
  ejecución y sus resultados reflejan la abstención decidida por el modelo a
  partir del prompt, no una validación empírica del umbral.

## Resultados de la evaluación

Ejecución del 25 de septiembre de 2026 con `ministral-8b-latest`,
`mistral-embed`, 40 documentos (10 procedimientos y 30 FAQ), 41 consultas
(31 respondibles y 10 de abstención), `TOP_K=3` y chunking de 200/20 (78
chunks):

| Métrica | Resultado |
|---|---:|
| Exactitud | 90.2% |
| Precisión | 93.5% |
| Recall | 93.5% |
| Especificidad | 80.0% |
| Hit rate @3 | 100.0% |
| Latencia promedio / máxima | 1.91 s / 5.33 s |
| Tiempo total | 78.1 s |
| Tokens de entrada / salida | 17,281 / 1,779 |
| Errores del proveedor | 0 |
| Costo estimado del reporte (referencial) | CLP $4 |

La matriz de confusión fue:

| Resultado del asistente | Esperado: responde | Esperado: abstiene |
|---|---:|---:|
| Responde | TP = 29 | FP = 2 |
| Se abstiene | FN = 2 | TN = 8 |

Los dos falsos positivos fueron `¿Atienden celulares?` y `¿Con quién hablo para
reclamar a la central?`; los dos falsos negativos fueron `¿Mi garantía cubre
si se me cayó té al teclado?` y `¿Los datos que respaldan se borran después?`.
El hit rate @3 cuenta si aparece al menos un documento esperado entre los tres
primeros resultados para las consultas con documentos de referencia. El 100%
es un techo en esta prueba sobre un corpus de 40 documentos, no una prueba de
que toda la calidad de recuperación esté resuelta.

Evidencia de esta corrida: [JSON](evaluation/results/evaluacion_2026-09-25_20-41-41.json)
y [resumen TXT](evaluation/results/evaluacion_2026-09-25_20-41-41.txt).

## Costos y limitaciones

Los CLP $4 son la estimación que produjo el evaluador usando las tarifas de
referencia configuradas: `TARIFA_IN_CLP=145` y `TARIFA_OUT_CLP=580` por millón
de tokens, basadas en una referencia anterior de GPT-4o-mini. No son tarifas
Mistral verificadas ni el importe facturado por el proveedor. Antes de tratar el
costo como representativo de Mistral, configura `TARIFA_IN_CLP` y
`TARIFA_OUT_CLP` con sus tarifas vigentes expresadas en CLP por millón de
tokens.

Las llamadas a embeddings y chat dependen de disponibilidad, cuotas y tarifas
del proveedor. La evaluación mide este dataset y esta configuración; no
garantiza el mismo resultado con otros modelos, corpus o umbrales.

## Estructura del proyecto

```text
.
├── data/
│   ├── corpus/
│   │   ├── procedimientos/   # 10 procedimientos en Markdown
│   │   └── faq/              # 30 preguntas frecuentes en Markdown
│   ├── entidades.json        # entidades del negocio
│   └── output/               # índice FAISS, reportes y sesiones generados
├── evaluation/
│   ├── dataset_evaluacion.json  # consultas y documentos esperados
│   └── results/                  # resultados JSON/TXT por ejecución
├── scripts/
│   ├── generar_corpus.py     # corpus y dataset
│   ├── build_index.py        # chunking, embeddings e índice FAISS
│   └── evaluate.py           # métricas, trazabilidad y costos estimados
├── src/
│   ├── config.py             # configuración central
│   ├── rag.py                # motor RAG, abstención y trazabilidad
│   └── app.py                # interfaz de consola
├── informe/                  # memoria técnica del proyecto
└── requirements.txt
```

## Cómo funciona

1. **Corpus**: 10 procedimientos internos y 30 preguntas frecuentes escritas
   en Markdown, más un dataset de 41 consultas de evaluación con su etiqueta
   (respondible o abstinencia) y los documentos relevantes.
2. **Chunking**: `RecursiveCharacterTextSplitter` con configuración
   seleccionada tras comparar 3 opciones (`data/output/chunking_report.json`).
   Los documentos cortos (FAQ) se conservan enteros para no separar preguntas
   de su respuesta.
3. **Embeddings**: `mistral-embed` en la configuración documentada; índice
   FAISS con metadatos de origen (documento, versión, tema, fecha) para
   trazabilidad. El proveedor y el modelo se pueden configurar.
4. **Generación**: `RetrievalQA` (LangChain) con prompt fundamentado que
   exige responder solo a partir del contexto recuperado y abstenerse
   explícitamente si no hay evidencia suficiente. La abstinencia se activa
   por umbral de similitud y por instrucción al modelo; ambos casos devuelven
   la misma frase canónica.
5. **Trazabilidad**: cada respuesta reporta los documentos recuperados con su
   score, versión y fecha, además de latencia y tokens consumidos.

## Mejoras futuras

- **Interfaz web**: Streamlit permite el prototipo Python más rápido; FastAPI
  con un frontend separado ofrece más control y separación, a costa de más
  trabajo de implementación y despliegue; Gradio es una alternativa rápida
  para una demostración.
- Añadir una compuerta de intención y alcance para reducir los dos falsos
  positivos fuera de dominio.
- Ampliar la evaluación con ejemplos difíciles y preguntas fuera de alcance, y
  probar el umbral con consultas que crucen su corte antes de considerarlo
  calibrado.
- Incorporar trazas de LangSmith y métricas de generación alineadas con los
  objetivos del curso.
- Añadir memoria multi-turno con reglas explícitas de privacidad y retención.
