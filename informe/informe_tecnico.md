# Informe Técnico - Asistente Documental RAG para TecnoApoyo

**Curso:** ISY0101 - Ingeniería de Soluciones con IA
**Autor:** Estudiante TecnoApoyo
**Fecha:** 9 de noviembre de 2026
**Versión:** 1.2

---

## 1. Resumen ejecutivo

Este proyecto implementa un asistente de atención al cliente basado en
**recuperación aumentada por generación (RAG)** para la pyme ficticia
TecnoApoyo, dedicada a la reparación de computadores en Santiago. El sistema
responde consultas de clientes (políticas, precios, garantías, procedimientos)
utilizando únicamente la documentación oficial de la empresa. Cuando la
consulta está fuera del alcance documentado, el asistente **se abstiene** de
responder con una frase canónica, evitando alucinaciones y compromisos que la
empresa no puede respaldar.

La solución sigue el patrón de los laboratorios del curso (LangChain +
FAISS + RetrievalQA) y se ejecuta contra la **API de Mistral** mediante un
endpoint compatible con OpenAI: `ministral-8b-latest` para generación y
`mistral-embed` (1024 dimensiones) para los embeddings, sobre
`https://api.mistral.ai/v1`. En un dataset de **41 consultas etiquetadas** se
obtuvo una exactitud de **90,2%**, con precisión de **96,6%** y sensibilidad de
**90,3%**, especificidad de **90,0%** —1 falso positivo sobre 10 consultas de
abstención— y una tasa de acierto del **96,8% (30/31 consultas respondibles)**
en top-3. Son las métricas medidas el 9 de noviembre de 2026
(`evaluation/results/evaluacion_2026-11-09_19-36-18.json`); uno de los tres
falsos negativos se debió a un alias de más en la frontera de dominio, ya
corregido en el código pero **no vuelto a medir** (§5.2). El costo estimado de
las llamadas de generación es de **≈$3 CLP con tarifas referenciales** (§6.3);
no corresponde a una facturación de Mistral y excluye los embeddings.

---

## 2. Introducción y problema

### 2.1 Contexto del negocio

TecnoApoyo es una pyme con manuales de procedimientos y respuestas a
preguntas frecuentes que se actualizan regularmente. El personal de atención
al cliente debe responder consultas fragmentadas sobre precios, garantías,
mantenimiento, retiro a domicilio y plazos. Reproducir de memoria esta
información presenta tres problemas:

1. **Inconsistencia**: cada ejecutivo responde de forma distinta.
2. **Desactualización**: las versiones antiguas de documentos se mezclan con
   las vigentes.
3. **Riesgo legal/comercial**: afirmar datos inexistentes (estado de una
   orden, precios no publicados) compromete a la empresa.

### 2.2 Pregunta del problema (IE8 - implicaciones de diseño)

> ¿Cómo entregar respuestas consistentes y verificables a clientes de una
> pyme, restringiendo el sistema al conocimiento documentado y abstiniéndose
> en caso contrario?

Esta pregunta define tres requisitos funcionales:

| ID | Requisito | Criterio de éxito medible |
|----|-----------|---------------------------|
| RF1 | Responder consultas respaldadas por documentación | Revisión de la literatura que justifique RAG |
| RF2 | Abstenerse cuando no hay evidencia suficiente | Especificidad de 100% en consultas no cubiertas |
| RF3 | Evidenciar el origen de cada respuesta (trazabilidad) | Cada respuesta reporta documentos y scores |

---

## 3. Revisión de la literatura (IE5)

### 3.1 Recuperación aumentada por generación (RAG)

El enfoque RAG combina un motor de recuperación con un modelo generativo:
en lugar de preguntar al modelo de lenguaje por su memoria paramétrica, se
recuperan primero los pasajes relevantes de un corpus y se inyectan en el
prompt (Lewis et al., 2020). Esto permite (i) actualizar el conocimiento
editando los documentos, (ii) citar las fuentes de la respuesta y (iii)
reducir alucinaciones al restringir el espacio de respuesta. RAG es la
arquitectura estándar para asistentes documentales empresariales.

### 3.2 Limitación de la memoria paramétrica

Los LLM memorizan conocimiento del *preentrenamiento*, con corte temporal y
sin control sobre la vigencia. Para una pyme, cuyos precios y políticas
cambian cada trimestre, depender de esa memoria es inaceptable. La literatura
indica que la **recuperación** (no el modelo) es la fuente principal de
exactitud factual en RAG (Chen et al., 2023): si el pasaje correcto no se
recupera, el modelo no puede responder bien.

### 3.3 Estrategias de chunking

El resultado de la recuperación depende de cómo se divide el corpus. El
estándar es el *recursive character splitter* con tamaño de chunk y overlap,
pero un hallazgo relevante para este proyecto es que **los documentos cortos
deben conservarse enteros**: separar la pregunta de su respuesta degrada la
recuperación. Por ello se compararon tres configuraciones (ver §5.1).

### 3.4 Abstinencia como patrón de seguridad

En sistemas de soporte, la abstinencia ("conocimiento del límite") es una
propiedad deseable documentada en la literatura de IA confiable: el modelo
debe reconocer cuándo no tiene evidencia en lugar de inventar (Amodei et al.,
2016, sobre alineación; enfoques modernos de *refusal elicitation*). Este
proyecto la implementa a tres niveles: frontera de dominio antes de recuperar
(`PRODUCTOS_FUERA_DE_ALCANCE`), umbral de similitud e instrucción al modelo con
frase canónica. Se trata de una descripción del **diseño** presente en el
código; conviene advertir desde ya que la corrida evaluada en §5.2 ejercitó la
frontera de dominio y la capa de instrucción, pero no el umbral, que nunca se
activó (§6.2).

### 3.5 Brecha identificada

Los laboratorios del curso construyen RAG de demostración con API externa
pero **no incorporan abstinencia ni trazabilidad ni evaluación de costos**.
Este proyecto extiende el patrón base con esas tres capas: un umbral de
abstinencia sobre el score de recuperación, la exposición de documentos,
versiones y scores en cada respuesta, y la estimación referencial del costo de
generación por corrida (§6.3).

El proyecto demuestra además la **portabilidad del stack entre proveedores**:
la migración descrita en §7 se resolvió modificando únicamente el endpoint,
la clave y los identificadores de modelo, sin alterar el grafo de LangChain
ni la persistencia de FAISS, y manteniendo la compatibilidad completa con el
stack del curso.

---

## 4. Método

### 4.1 Arquitectura

```
                      +------------------------+    +------------------------+
   Consulta usuario    |  mistral-embed         |    |  ministral-8b-latest   |
   ------------------->|  (embeddings, API)     |    |  (generación, API)     |
                      +------------------------+    +------------------------+
                              |                             ^
                              v                             |
   corpus md --> chunking --> FAISS (k=3) --contexto--> RetrievalQA
                               (metadatos de               (prompt fundamentado)
                                origen, scores)            (abstinencia)

   Ambas llamadas (embeddings y generación) se resuelven contra
   https://api.mistral.ai/v1 mediante una interfaz compatible con OpenAI.
```

### 4.2 Componentes

1. **Corpus sintético** (`scripts/generar_corpus.py`): 10 procedimientos
   (PROC-001..010) y 30 preguntas frecuentes (FAQ-001..030) en Markdown,
   coherentes entre sí (mismos precios, plazos y garantías en ambas fuentes).
2. **Índice vectorial** (`scripts/build_index.py`): embeddings
   `mistral-embed` (1024 dimensiones), chunking recursivo comparado, índice
   FAISS con metadatos de origen (documento, versión, tema, fecha).
3. **Motor RAG** (`src/rag.py`): `RetrievalQA` con prompt fundamentado y
   control de abstinencia; expone scores, documentos, versiones, latencia y
   tokens de cada consulta.
4. **Interfaz** (`src/app.py`): consola con rich, historial y exportación de
   sesiones.
5. **Evaluación** (`scripts/evaluate.py`): dataset de 41 consultas con
   métricas de clasificación y de recuperación.

### 4.3 Dataset de evaluación

Se definieron **41 consultas** en dos grupos:

- **31 respondibles** (deben ser respondidas): preguntas directas sobre las
  políticas (precio del diagnóstico, garantías, horario) y reformulaciones
  semánticas (p. ej. "¿me pueden venir a buscar el computador a mi casa?").
  Cada consulta se etiquetó con **todos** los documentos que la responden
  (PROC y/o FAQ), para evaluar recuperación de forma justa.
- **10 de abstinencia** (deben ser rechazadas): estado de una orden de
  servicio, precios de servicios no ofrecidos (iPhone, celulares), datos no
  públicos (cámara del local) y temas fuera de dominio (abogados, dólar).

Para cada consulta se midió TP/TN/FP/FN (responde o abstiene vs. etiqueta) y
la recuperación (documento relevante presente en top-3).

---

## 5. Experimentos y resultados

### 5.1 Selección de la configuración de chunking

| Configuración | Chunks | Longitud media | Observación |
|---------------|--------|----------------|-------------|
| 200 chars, overlap 20 | 78 | 176,6 | FAQ enteras |
| 350 chars, overlap 50 | 60 | 230,9 | Divisibles al 1,5x |
| 500 chars, overlap 75 | 42 | 328,6 | Poca granularidad |

Se seleccionó **chunk 200 / overlap 20 con chunking adaptativo**: los
documentos cortos (FAQ) se conservan enteros. *Evidencia*:
`data/output/chunking_report.json`.

La selección se basó en qué longitud media queda más cerca del tamaño objetivo
de 200 caracteres: `chunk 200 / overlap 20` es la configuración más cercana
(176,6 caracteres). El número de chunks se reporta como dato descriptivo, pero
no fue un criterio de selección; el chunking adaptativo conserva, además, los
documentos cortos enteros. **No se comparó la calidad de recuperación entre
configuraciones**; la evidencia de recuperación de §5.2 corresponde a esta
configuración y extender el análisis a las otras dos queda como trabajo
futuro (§7).

### 5.2 Resultado principal de la evaluación

Base: ministral-8b-latest, mistral-embed, k=3, umbral de abstinencia 0,45.
Corrida del 9 de noviembre de 2026 (`evaluacion_2026-11-09_19-36-18`).

> **Nota sobre el umbral de abstinencia**: el corte de 0,45 no se activó en
> ninguna de las 38 consultas que llegaron a la recuperación, cuyos puntajes se
> sitúan entre 0,5746 y 0,8671. Las otras 3 consultas no llegaron al índice: la
> frontera de dominio (`PRODUCTOS_FUERA_DE_ALCANCE`) las cortó antes de buscar,
> de modo que las métricas de esta tabla reflejan el comportamiento de la capa
> de generación y el de esa frontera, no el del umbral (§6.2).

| Métrica | Fórmula | Valor |
|---------|---------|-------|
| Exactitud (accuracy) | (TP+TN)/(TP+TN+FP+FN) | **90,2%** |
| Precisión | TP/(TP+FP) | **96,6%** |
| Sensibilidad (recall) | TP/(TP+FN) | **90,3%** |
| Especificidad | TN/(TN+FP) | **90,0%** |
| Tasa de acierto @3 | doc. esperado presente en top-3 | **96,8%** |
| Latencia promedio | tiempo por consulta | **2,50 s** |
| Latencia máxima | peor caso observado | **6,07 s** |
| Tiempo total de la prueba | 41 consultas | **1.302,3 s** |
| Consultas con error del proveedor | — | **0** |

> **Nota sobre la tasa de acierto @3**: es una *tasa de acierto* (hit rate),
> es decir, indica si al menos un documento esperado aparece dentro del top-3;
> no es una métrica consciente del orden (*rank-aware*). El 96,8% corresponde
> a 30 de 31 consultas respondibles; la única respondible sin documento
> esperado en el top-3 es "¿En qué horario atienden por teléfono?", que la
> frontera de dominio cortó antes de recuperar (ver el párrafo siguiente). Las
> 10 consultas de
> abstención no tienen documentos esperados asociados. Dado el tamaño acotado
> del corpus y de la muestra, este resultado describe únicamente el conjunto
> evaluado y no basta para demostrar una calidad de recuperación generalizable.

**Matriz de confusión**

| | Responde | Abstiene |
|---|---|---|
| **Debía responder** | 28 (TP) | 3 (FN) |
| **Debía abstenerse** | 1 (FP) | 9 (TN) |

*Evidencia completa*: `evaluation/results/evaluacion_2026-11-09_19-36-18.json`
y `evaluation/results/evaluacion_2026-11-09_19-36-18.txt`.

**Por qué la matriz no coincide con la corrida de septiembre.** El 7 de
noviembre se agregó la frontera de dominio `PRODUCTOS_FUERA_DE_ALCANCE` (commit
`ccc4d3b`), que corta la consulta antes de recuperar cuando nombra un producto
que el corpus no documenta. No cambió el modelo: las dos corridas usan
`ministral-8b-latest` y `mistral-embed` sobre el mismo índice. Ese guard movió
dos de los 41 veredictos. "¿Atienden celulares?" pasó de responderse a
abstenerse —la corrección esperada, que explica la subida de especificidad— y
"¿En qué horario atienden por teléfono?" pasó de responderse a abstenerse
porque `telefono` y `telefonos` figuraban entre los alias de `celular`: de ahí
el tercer falso negativo y la baja del recall.

**Corrección pendiente de medición.** Los alias `telefono` y `telefonos` ya no
están en la lista. La corrección **no se ha vuelto a medir**: restituye
exactamente una consulta a "responde" —la de horario por teléfono, que en la
corrida previa al guard recuperó FAQ-029, FAQ-001 y FAQ-006 y se respondió—, de
modo que la matriz proyectada es TP=29, TN=9, FP=1, FN=2 y la exactitud
proyectada 92,7% (38 de 41). Es una **proyección, no un resultado**: no se
proyecta ninguna otra métrica y las cifras medidas siguen siendo las de la
tabla de arriba. La re-medición queda como trabajo futuro (§7).

---

## 6. Análisis y discusión

### 6.1 Interpretación de los resultados

- **Especificidad 90,0% con 1 falso positivo**: el asistente no siempre
  se abstiene cuando corresponde. Para una pyme este es el punto más
  crítico, porque un compromiso falso tiene impacto legal/comercial directo.
  El FP es "¿Con quién hablo para reclamar a la central?" (0,6072): el
  documento recuperado era temáticamente cercano —el procedimiento de reclamos
  por garantía—, pero la intención del usuario queda fuera del alcance
  documentado. Es la limitación honesta de un umbral de similitud puro: la
  similitud no distingue "parecido a un documento" de "parecido a un
  documento pero fuera de alcance para responder". En esta corrida el umbral
  tampoco habría servido de red: no se activó en ninguna consulta, de modo que
  la capa de similitud no tuvo ocasión de advertirlo (§6.2). El otro falso
  positivo de septiembre ("¿Atienden celulares?") ya no existe: la frontera de
  dominio lo corta antes de recuperar, porque ahí el producto sí es un equipo.
- **Sensibilidad 90,3%**: 28 de 31 consultas respondibles se respondieron
  correctamente. Dos de los 3 FN son del modelo —"¿Mi garantía cubre si se me
  cayó té al teclado?" (0,6980) y "¿Los datos que respaldan se borran
  después?" (0,6896)—: ambos puntajes están claramente por sobre el corte —de
  hecho, por encima de varias abstinencias correctas del mismo conjunto— y la
  recuperación entregó el documento esperado dentro del top-3; el modelo se
  abstuvo igualmente, devolviendo la frase canónica pese a tener contexto
  pertinente delante. Se trata, por tanto, de un problema de **calibración de la
  instrucción** y no de recuperación ni de umbral (§6.2). El tercero es de otra
  naturaleza: "¿En qué horario atienden por teléfono?" recuperó **cero**
  documentos porque la frontera de dominio la cortó antes de buscar, y es el
  defecto de alias descrito en §5.2; no es un caso de calibración del modelo.
- **Tasa de acierto @3 96,8%**: el documento esperado apareció en el top-3 en
  30 de las 31 consultas respondibles; la restante es la consulta de horario por
  teléfono, cortada antes de recuperar (§5.2). El resultado corresponde a este
  conjunto y no demuestra por sí solo que la recuperación mantenga esa calidad
  con otros corpus o consultas; tampoco permite comparar configuraciones de
  chunking (§5.2). No debe interpretarse como una calidad de recuperación
  excepcional.

### 6.2 Diagnóstico de la sobre-abstinencia

Durante el desarrollo se observó que el modelo tendía a abstenerse incluso
con contexto válido: la primera corrida registrada (`evaluacion_2026-09-23_22-42`,
hoy superada) mostraba un recall de 58,1% con especificidad de 100%. Después se
ajustó la instrucción del prompt hacia "ante la duda, responde con la
información disponible" y se configuró un umbral de 0,50 a 0,45. La corrida
actual obtuvo un recall de 90,3%, pero esta comparación no permite atribuir la
mejora únicamente a esos ajustes: la corrida anterior usó `llama3.2:3b` y
`nomic-embed-text`, mientras que la actual usa modelos de Mistral. Por tanto,
se trata de una diferencia observada entre configuraciones, no de una medición
aislada del efecto del prompt o del umbral. El informe declara un umbral de
0,45 para la corrida actual, pero el JSON no registra el valor efectivo y
`UMBRAL_ABSTENCION` puede sobrescribirse mediante la configuración del entorno.

La evidencia indica que **el umbral de similitud no se activó en ninguna de
las consultas que llegaron a la recuperación**. Los valores `top_score` del
archivo de evidencia van de **0,5746 a 0,8671**; con el umbral de 0,45
informado, ninguna habría quedado por debajo del corte. Además, 38 de las 41
consultas registran tokens mayores que cero. Como el motor asigna cero tokens a
una consulta detenida por el umbral —y también a una cortada por la frontera de
dominio, que ni consulta el índice—, esas 38 llegaron al modelo de lenguaje y
las 3 restantes no. El valor efectivo del umbral, sin embargo, no quedó guardado
en el JSON.

Con los puntajes observados, cualquier umbral de hasta 0,57 habría dejado pasar
las 38 consultas que recuperaron. Por eso, el valor de 0,45 no tuvo un efecto
observable en esta corrida y no puede considerarse un óptimo empírico. **La
abstención observada provino de la capa de generación** —la instrucción al
modelo y la frase canónica— y, en 2 de las 10 consultas de abstención, de la
frontera de dominio, que las cortó antes de recuperar; no de un corte numérico
sobre el score de recuperación.

Los errores de la corrida confirman esa lectura, con un matiz. Los 2 FN del
modelo ("¿Mi garantía cubre si se me cayó té al teclado?", 0,6980; "¿Los datos
que respaldan se borran después?", 0,6896) no son casos de puntaje limítrofe:
ambos scores están claramente por sobre el corte y por encima de varias
abstenciones correctas del mismo conjunto. El documento esperado se recuperó
dentro del top-3 y, aun así, el modelo devolvió la frase canónica de
abstención. Se trata de **calibración de la instrucción**, no de un problema de
recuperación ni de umbral. El FP ("¿Con quién hablo para reclamar a la
central?", 0,6072) es el caso especular: el modelo respondió cuando debía
abstenerse, y la capa de similitud no tenía manera de advertirlo, porque ninguna
consulta del conjunto obtuvo un puntaje lo suficientemente bajo como para
activar el corte. El tercer FN no pertenece a esta lectura: "¿En qué horario
atienden por teléfono?" tiene `top_score` 0 porque no se recuperó nada, y su
causa es el alias de más en la frontera de dominio, ya corregido y pendiente de
re-medición (§5.2).

Los rangos de puntajes se solapan: las consultas respondibles empiezan en
0,6896 y las de abstención llegan hasta 0,7063, el puntaje de "¿Cuál es el
estado de mi orden de servicio 4590?", que se abstuvo **correctamente**. Dos de
los tres errores que decidió el modelo caen en el intervalo compartido
(0,6896–0,7063); el falso positivo "¿Con quién hablo para reclamar a la
central?" (0,6072) queda por debajo. En esta muestra, un umbral superior a
0,6072 y de hasta 0,6896 podría detener ese falso positivo sin bloquear
consultas respondibles según sus puntajes. Es una hipótesis basada en los
resultados almacenados, no una corrida con ese umbral: debe verificarse mediante
una nueva evaluación y no demuestra que el corte generalice.

La implicación de diseño es directa. El diseño de **tres capas** —frontera de
dominio antes de recuperar, corte por similitud en la recuperación e instrucción
en la generación— existe en el código; **esta corrida ejercitó la frontera de
dominio y la capa de instrucción, no el umbral**. El primer umbral con una
decimal que excluiría una consulta es 0,60 (con 0,57 no se excluye ninguna); la
consulta "¿Me pueden recomendar un abogado?" tiene un score de 0,5746 y ya se
abstuvo **correctamente**. Con un corte de 0,60 se mantendría esa abstención y
se evitaría invocar al modelo para esa consulta, lo que podría reducir tokens y
latencia, aunque no cambiaría su clasificación. Para confirmar el efecto de
otros cortes sobre el conjunto completo se requiere una nueva evaluación con
cada umbral. La calibración general también exige un dataset representativo de
consultas fuera de alcance, incluidas las difíciles; la corrida reportada no
ejercitó esta capa (§6.4).

### 6.3 Estimación del costo de inferencia (IE8)

La corrida de evaluación (41 consultas) registró **15.961 tokens de entrada**
y **1.722 de salida** para las llamadas de generación. Al aplicar las tarifas
de referencia de GPT-4o-mini —**145 CLP/M** (entrada) y **580 CLP/M** (salida)—,
el costo estimado es el siguiente. No es un registro de facturación de Mistral
y no incluye el costo de las llamadas de embeddings.

| Concepto | Cálculo | Valor |
|----------|---------|-------|
| Tokens de entrada (41 consultas) | — | 15.961 |
| Tokens de salida | — | 1.722 |
| Costo estimado de entrada | 15.961/1e6 × 145 | `$2,31 CLP` |
| Costo estimado de salida | 1.722/1e6 × 580 | `$1,00 CLP` |
| **Costo estimado de generación** | 2,31 + 1,00 | **`$3,31 CLP` (≈ `$3 CLP` redondeado al peso)** |
| Latencia media por consulta | — | 2,50 s |

El consumo medio fue de **≈389 tokens de entrada por consulta** (15.961 / 41).
Esta estimación permite dimensionar el uso de tokens de generación, pero no
basta para comparar el costo total de la API de Mistral con la atención humana:
para ello se requieren las tarifas vigentes del proveedor, el costo de los
embeddings y una estimación del costo de atención.

### 6.4 Limitaciones

1. **Capa de similitud sin verificar**: el umbral informado **no se activó**
   en ninguna de las 38 consultas que llegaron a la recuperación (puntajes
   entre 0,5746 y 0,8671); las otras 3 las cortó la frontera de dominio antes
   de buscar, así que esta corrida no demuestra la contribución del umbral. Los
   puntajes sugieren un umbral candidato que podría detener el falso positivo de
   0,6072 sin bloquear las consultas respondibles de este conjunto, pero debe
   validarse con una nueva evaluación y datos representativos antes de
   considerarlo calibrado (§6.2).
2. **Calibración de la instrucción**: en los 2 FN del modelo, la recuperación
   entregó el documento esperado en el top-3 y los puntajes (0,6980 y 0,6896)
   superaron el corte informado. Aun así, el modelo optó por la frase canónica,
   por lo que la instrucción del prompt es un aspecto a revisar.
3. **Falsos positivos por topicalidad**: 1 de 10 consultas fuera de alcance
   fue respondida (especificidad 90,0%) porque el documento recuperado era
   temáticamente cercano aunque la intención no estuviera documentada. La
   frontera de dominio ya corta los productos listados; lo que queda fuera de su
   alcance requiere una capa de **clasificación de intención** previa a la
   recuperación (§7).
4. **Corpus sintético**: la documentación fue creada para el proyecto; validar
   con documentación real de la pyme es trabajo futuro.
5. **Sin conversación multi-turno**: el historial se mantiene en sesión pero
   el prompt no encadena contexto previo; una consulta de seguimiento se
   trata de forma independiente.
6. **Cobertura y tamaño del corpus**: la tasa de acierto @3 fue de 96,8% (30/31
   consultas respondibles; la restante es la consulta de horario por teléfono,
   cortada por la frontera de dominio) en un corpus de 40 documentos con k=3.
   Este resultado describe el conjunto evaluado, pero no demuestra que la
   recuperación sea trivial ni que mantenga la misma calidad con corpus reales,
   más grandes y heterogéneos. El comportamiento en esos escenarios no fue
   medido.

---

## 7. Conclusiones y trabajo futuro

**Respuesta al problema**: se construyó un asistente RAG que responde con
evidencia documental y se abstiene fuera de alcance. En 41 consultas de
evaluación obtuvo **90,2%** de exactitud, **96,6%** de precisión, **90,3%** de
sensibilidad, **90,0%** de especificidad y una tasa de acierto del **96,8%
(30/31 consultas respondibles)** en top-3, con **1 falso positivo** sobre 10
consultas de abstención (corrida del 9 de noviembre de 2026,
`evaluacion_2026-11-09_19-36-18`). Con el alias de teléfono corregido —aún sin
re-medir— la matriz proyectada es TP=29, TN=9, FP=1, FN=2 y la exactitud
proyectada 92,7% (§5.2).

**Cumplimiento de requisitos**:

| Requisito | Estado |
|-----------|--------|
| RF1 responder con documentación | Cumplido (tasa de acierto @3 = 96,8%; 30/31 consultas respondibles) |
| RF2 abstención fuera de alcance | **Cumplimiento parcial** (especificidad 90,0%; 1 de 10 consultas no cubiertas fue respondida) |
| RF3 trazabilidad | Cumplido (documentos, versiones, scores) |

El criterio de éxito de RF2 exigía **una especificidad de 100% en las consultas
no cubiertas**, y el resultado medido queda por debajo de ese umbral. La brecha
no proviene de la recuperación —que localizó el documento esperado en 30 de 31
consultas respondibles; la restante la cortó la frontera de dominio— sino de la
ausencia de una capa que discrimine la intención del usuario antes de aceptar
una coincidencia temática. La frontera de dominio es un primer paso en esa
dirección, pero es una lista enumerativa: acierta cuando el producto está
listado y falla cuando la palabra no es un equipo, como mostró el alias de
teléfono (§5.2).

**Trabajo futuro**:
1. Incorporar una **capa de clasificación de intención** (*query routing*) que
   rechace consultas fuera de alcance antes de la recuperación: es la
   intervención de mayor impacto sobre el falso positivo restante, hoy cubierto
   solo por una lista enumerativa de productos.
2. Añadir *query re-writing* o expansión de la consulta para reforzar las
   reformulaciones en las que el modelo se abstiene pese a haber recuperado el
   documento esperado (los 2 FN del modelo, §6.2).
3. Habilitar contexto multi-turno en el prompt (memoria de conversación).
4. Comparar la calidad de recuperación entre las tres configuraciones de
   chunking (§5.1).
5. Migrar el corpus a documentación real y agregar roles de usuario
   (ejecutivo, técnico, cliente) con permisos distintos.
6. Construir un dataset con consultas fuera de alcance y difíciles que
   produzcan puntajes bajos, para recién entonces poder **calibrar el umbral
   de similitud**, hoy sin verificar (§6.4).
7. Re-ejecutar `scripts/evaluate.py` sobre el índice actual para reemplazar la
   proyección de §5.2 por una medición del efecto del alias corregido.

**Reflexión sobre el proceso de diseño (IE8)**: la decisión más importante no
fue arquitectónica sino de límites: definir explícitamente *qué no debe
responder el sistema*. Eso guió el diseño de tres capas de abstinencia —frontera
de dominio antes de recuperar, corte por similitud en la recuperación e
instrucción en la generación— y la elección de métricas (especificidad y falsos
positivos); con el conjunto disponible se logró ejercitar la frontera de dominio
y la capa de instrucción, no el umbral (§6.2), lo que deja esa capa como trabajo
futuro declarado y no como una capacidad demostrada. La frontera de dominio, a
su vez, mostró su propio límite: una lista enumerativa de productos puede cortar
de más si incluye una palabra que no es un equipo, como ocurrió con el teléfono
(§5.2). La migración hacia la **API de Mistral** fue una decisión deliberada de
portabilidad, no un cambio
de arquitectura: el mismo stack (LangChain + FAISS + RetrievalQA) se trasplantó
modificando solo la configuración —endpoint, clave e identificadores de
modelo—. La corrida permitió medir tokens y latencia de generación; el costo se
presenta como una estimación referencial y no como facturación real (§6.3).
También dejó la lección central del proyecto:
con la recuperación resuelta, el error residual no es de conocimiento sino de
**intención**.

## Referencias

[1] P. Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive
NLP Tasks", en *Advances in Neural Information Processing Systems* (NeurIPS),
2020.

[2] J. Chen, H. Lin, X. Han et al., "Benchmarking Large Language Models in
Retrieval-Augmented Generation", *arXiv preprint arXiv:2309.01431*, 2023.

[3] D. Amodei, C. Olah, J. Steinhardt et al., "Concrete Problems in AI
Safety", *arXiv preprint arXiv:1606.06565*, 2016.

[4] Mistral AI, "Models overview: modelos de generación y de embeddings
disponibles en la plataforma", https://docs.mistral.ai/getting-started/models/models_overview/
(accedido en septiembre de 2026).

[5] Ollama, "Ollama: Run Llama, Mistral, and other models locally",
https://ollama.com (accedido en septiembre de 2026). Referencia histórica:
documenta la configuración local previa a la migración descrita en §7; no
corresponde al runtime actual.

[6] LangChain, "LangChain Documentation", https://python.langchain.com
(accedido en septiembre de 2026).

[7] LangChain Community, "FAISS vector store",
https://python.langchain.com/docs/integrations/vectorstores/faiss (accedido
en septiembre de 2026).

---

## Anexo A - Evidencia generada

| Artefacto | Ruta |
|-----------|------|
| Corpus (procedimientos) | `data/corpus/procedimientos/` |
| Corpus (FAQ) | `data/corpus/faq/` |
| Manifiesto del índice | `data/output/documentos.json` |
| Dataset de evaluación | `evaluation/dataset_evaluacion.json` |
| Reporte de chunking | `data/output/chunking_report.json` |
| Resultado de evaluación (resumen) | `evaluation/results/evaluacion_2026-11-09_19-36-18.txt` |
| Detalle por consulta | `evaluation/results/evaluacion_2026-11-09_19-36-18.json` |
| Sesiones de demo | `data/output/sesiones/` |

> Las corridas anteriores (`evaluacion_final.json`/`.txt`, los archivos
> `evaluacion_2026-09-23_*.json`/`.txt` y `evaluacion_2026-09-25_20-41-41.*`)
> permanecen en esa carpeta pero están **superadas** por la corrida
> `2026-11-09_19-36-18`, que es la única citada en §5.2.
