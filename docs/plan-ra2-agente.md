# Plan de implementación RA2 — Capa de agentes sobre TecnoApoyo RAG

**Estado:** propuesta, pendiente de aprobación.
**Base analizada:** `main_files/Evaluaciones.md` (Parcial 2, 35%), `main_files/Experiencias_aprendizaje.md` (IL2.1–IL2.3), material `RA2/`.
**Proyecto:** motor RAG existente (`src/rag.py`, FAISS + Mistral + abstención + evaluación).

## 1. Diagnóstico

| # | Exige el Parcial 2 | Estado actual | Brecha |
|---|---|---|---|
| 1 | Análisis del flujo organizacional y tareas cognitivas complejas | Disperso en el informe | Documento explícito |
| 2 | Agente con herramientas de consulta, escritura y razonamiento (IL2.1) | RAG fijo `retrieve → generate`; sin `bind_tools` ni executor | Envolver el RAG como herramienta + herramientas de casos |
| 3 | Memoria short-term y long-term con continuidad (IL2.2) | `historial` en RAM por proceso; `/guardar` exporta texto y no recarga | Ventana/resumen + almacén persistente |
| 4 | Planificación y decisiones multi-etapa con adaptación (IL2.3) | Flujo de un solo paso | Planificador + ejecución con re-plan |
| 5 | README técnico de diseño y orquestación (IL2.4) | README de instalación y uso | README de arquitectura |

Además: el repositorio no tiene commits; todo está untracked.

## 2. Arquitectura propuesta

```
CLI rich (src/app.py)
   │
AgenteTecnoApoyo (src/agente.py)          ← allowlist, máx. iteraciones, confirmación
   ├── Planificador (src/planificador.py)  → plan JSON + re-plan
   ├── Memoria (src/memoria.py)
   │     ├── corto plazo: buffer / ventana / resumen incremental
   │     └── largo plazo: SQLite (hechos, preferencias, casos)
   └── Herramientas (src/herramientas.py)  [function calling nativo]
         ├── buscar_documentacion   (envuelve RAGEngine.recuperar)   consulta
         ├── consultar_caso / listar_casos                          consulta
         ├── crear_caso / agregar_nota / actualizar_estado          escritura
         ├── escalar_a_humano                                       escritura
         └── calcular_plazo                                         razonamiento
RAGEngine (src/rag.py) — INTACTO: FAISS, prompt, abstención, tokens
Trazas JSONL (data/output/trazas) + LangSmith opcional
```

Regla de diseño: `rag.py` no se reescribe, se expone. La abstención medida y la matriz TP/TN/FP/FN no se degradan.

## 3. Fases

### F0 — Base (½ día)
- Primer commit del baseline en rama `feature/ra2-agente`.
- `pytest` + `tests/` con test de humo del RAG actual.
- Corregir el banner que dice "Ollama" en `src/app.py`.
- Nuevas variables en `.env.example`; sin dependencias nuevas salvo `pytest`.

### F1 — IL2.1: agente y herramientas (1–1.5 días)
- `src/casos.py`: SQLite con 10 casos sintéticos coherentes con `PROC-*` y validación de estados.
- `src/herramientas.py`: contratos Pydantic + `@tool`.
- `src/agente.py`: `bind_tools` + executor con `max_iterations` y fallback al pipeline actual si el modelo no soporta tool calling.
- Toda escritura se propone y se confirma en consola.
- Verificación: tests con LLM falso + corrida real de un caso de soporte.

### F2 — IL2.2: memoria (¾ día)
- `src/memoria.py` con las tres estrategias del material (buffer, ventana, resumen) y almacén SQLite durable.
- Herramienta `recordar`; comandos `/memoria` y `/sesion cargar`.
- Verificación: test de reinicio de proceso y test de ventana.

### F3 — IL2.3: planificación (1 día)
- `src/planificador.py`: plan JSON validado contra allowlist, ejecución con dependencias y re-plan con tope.
- Dos modos comparables (reactivo vs. planificado).
- Verificación: test de re-plan y corrida E2E multi-etapa.
- CrewAI como demo aislada (`scripts/demo_crewai.py`, `requirements-agentes.txt`), fuera del runtime.

### F4 — IL2.4: documentación y observabilidad (¾ día)
- Traza por turno (plan, herramientas, argumentos, ok/error, tokens, latencia) a JSONL.
- README técnico: diagrama mermaid, contratos de herramientas, modelo de memoria, ciclo de planificación, límites y mapa IL2.1–IL2.4.
- Actualizar `whitebox.md`; informe v1.2 aparte, sin sobrescribir v1.0/v1.1.

### F5 — Evaluación y cierre (½ día)
- `scripts/evaluate_agente.py` con 10–15 tareas: selección de herramienta, argumentos, escritura confirmada, memoria y plan válido.
- Regresión del RAG sin cambios.

**Total estimado:** 18–24 h efectivas (4–5 días).

## 4. Fuera de alcance
MCP server, microservicios, event-driven y web/WhatsApp: son conceptos de investigación de RA2, no exigencia de la rúbrica.

## 5. Riesgos
1. Tool calling con `ministral-8b`: sonda temprana; si es inestable, modelo de selección configurable.
2. Repositorio sin commits: el baseline de F0 lo resuelve.
3. Dominio de escritura: casos sintéticos en SQLite, coherentes con el corpus existente.
