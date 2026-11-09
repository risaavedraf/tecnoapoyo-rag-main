# -*- coding: utf-8 -*-
"""
Evaluación del asistente RAG (Fase 5).

Ejecuta el dataset de evaluación (evaluation/dataset_evaluacion.json, 41
consultas: 31 respondibles y 10 de abstinencia) contra el motor, y registra:

  - métricas de recuperación: tasa de acierto @k (hit rate), es decir, en
    cuántas consultas aparece AL MENOS UNO de los documentos esperados entre
    los top-k recuperados;
  - cumplimiento de abstinencia (cuántas veces abstiene cuando debe y
    responde cuando puede);
  - matriz de confusión 2x2 (responde/abstiene vs debe/dice);
  - costos estimados por tokens (IN/OUT con tarifa de referencia),
  - latencias, y los casos fallidos con su trazabilidad.

Guarda el resultado en evaluation/results/evaluacion_<fecha>.json y
evaluation/results/evaluacion_<fecha>.txt (legible para el informe).

Uso:
    python scripts/evaluate.py
"""
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.rag import RAGEngine

# Tarifas de referencia para el análisis de costos (IE8), citando el modelo
# oficial del curso. Son tarifas de un proveedor externo; sirven para estimar
# el costo si el asistente se migrara a una API en producción.
# UNIDAD: CLP (pesos chilenos) POR MILLÓN de tokens, no CLP por token.
# Referencia: precios públicos GPT-4o-mini (entrada US$0,15 / salida US$0,60 por
# millón de tokens) convertidos a CLP a ~965 CLP/USD -> 145 y 580 CLP/millón.
# OJO: las tarifas son específicas de cada proveedor; deben corresponder al
# proveedor realmente configurado en LLM_BASE_URL (Mistral cobra otras). Se
# pueden sobreescribir con TARIFA_IN_CLP y TARIFA_OUT_CLP.
TARIFA_IN_CLP = float(os.getenv("TARIFA_IN_CLP", "145"))    # CLP por millón de tokens de entrada
TARIFA_OUT_CLP = float(os.getenv("TARIFA_OUT_CLP", "580"))  # CLP por millón de tokens de salida


def cargar_dataset():
    datos = json.loads(config.RUTA_EVAL.read_text(encoding="utf-8"))
    return datos.get("consultas", datos) if isinstance(datos, dict) else datos


def evaluar():
    motor = RAGEngine()
    consultas = cargar_dataset()

    dir_salida = config.DIR_EVAL_RESULTADOS
    dir_salida.mkdir(parents=True, exist_ok=True)

    detalle = []
    tp = tn = fp = fn = 0
    errores_proveedor = 0
    aciertos_at_k = 0
    n_evaluadas_at_k = 0
    tokens_in = tokens_out = 0
    latencias = []

    inicio_total = time.time()
    for i, c in enumerate(consultas, 1):
        pregunta = c["query"]
        tipo_esperado = "respondible" if c.get("responde", True) else "abstencion"
        esperado = tipo_esperado == "respondible"

        r = motor.consultar(pregunta)

        # tasa de acierto @k: ¿aparece AL MENOS UNO de los documentos esperados
        # entre los top-k? No es precision@k (esa ponderaría cada documento
        # recuperado), por eso se reporta con su propio nombre. Se calcula antes
        # del chequeo de error porque la recuperación ocurre antes de la llamada
        # al LLM: si el proveedor se cae, este dato sigue siendo válido.
        esperados_docs = c.get("documentos", [])
        if esperados_docs:
            recuperados = [d["document_id"] for d in r.documentos_recuperados[:config.TOP_K]]
            aciertos_at_k += any(e in recuperados for e in esperados_docs)
            n_evaluadas_at_k += 1

        # Integridad de la evidencia: RAGEngine.consultar() ya no lanza la
        # excepción del proveedor, devuelve una respuesta con error=... y
        # abstencion=False. Si esa consulta se puntuara como una cualquiera,
        # una caída del LLM contaría cada respondible como acierto (obtuvo=True)
        # y cada abstención como acierto también (obtuvo=False==esperado), y la
        # corrida cerraría con 100% de exactitud: un artefacto falso
        # indistinguible de una corrida realmente perfecta, y el informe cita
        # estos números como evidencia. Por eso el error se detecta ANTES de la
        # matriz de confusión y la consulta queda fuera de sus denominadores.
        if r.error:
            errores_proveedor += 1
            detalle.append({
                "consulta": pregunta,
                "esperado": tipo_esperado,
                "obtenido": "ERROR_PROVEEDOR",
                "correcto": False,
                "error": r.error,
                "top_score": r.documentos_recuperados[0]["score"] if r.documentos_recuperados else 0,
                "documentos_recuperados": [d["document_id"] for d in r.documentos_recuperados],
                "latencia": r.latencia_seg,
                "tokens_total": r.tokens_total,
            })
            print(f"[{i:02d}/{len(consultas)}] {'ERROR':9s} | esperado={tipo_esperado:12s} "
                  f"| proveedor: {r.error[:60]}")
            continue

        obtuvo = not r.abstencion

        if obtuvo and esperado:
            tp += 1
        elif not obtuvo and not esperado:
            tn += 1
        elif obtuvo and not esperado:
            fp += 1
        else:
            fn += 1

        tokens_in += r.tokens_entrada
        tokens_out += r.tokens_salida
        latencias.append(r.latencia_seg)

        cero_ok = "ok" if (obtuvo == esperado) else "INCORRECTO"
        detalle.append({
            "consulta": pregunta,
            "esperado": tipo_esperado,
            "obtenido": "responde" if obtuvo else "abstiene",
            "correcto": obtuvo == esperado,
            "top_score": r.documentos_recuperados[0]["score"] if r.documentos_recuperados else 0,
            "documentos_recuperados": [d["document_id"] for d in r.documentos_recuperados],
            "latencia": r.latencia_seg,
            "tokens_total": r.tokens_total,
        })
        print(f"[{i:02d}/{len(consultas)}] {cero_ok:9s} | esperado={tipo_esperado:12s} "
              f"| obtenido={'responde' if obtuvo else 'abstiene':8s} | {pregunta[:70]}")

    # Si TODAS las consultas fallaron con el proveedor no hay nada que medir:
    # seguir adelante escribiría artefactos con cero consultas evaluadas y
    # métricas en cero, indistinguibles de un modelo que nunca recuperó nada.
    # Una corrida sin respuesta del proveedor no es evidencia válida, así que
    # se aborta sin escribir nada.
    if errores_proveedor and errores_proveedor == len(consultas):
        print("\n" + "=" * 72)
        print("EVALUACIÓN ABORTADA: TODAS LAS CONSULTAS FALLARON CON EL PROVEEDOR DE LLM")
        print(f"  Consultas con error : {errores_proveedor}/{len(consultas)}")
        print(f"  Primer error        : {detalle[0]['error']}")
        print("  No se escribieron archivos de resultados: una corrida en la que")
        print("  el proveedor nunca respondió no produce evidencia válida.")
        print("  Revisa LLM_BASE_URL, LLM_API_KEY y la cuota del proveedor.")
        print("=" * 72)
        return

    segundos = round(time.time() - inicio_total, 1)
    total = tp + tn + fp + fn
    exactitud = (tp + tn) / total if total else 0
    precision_abst = tp / (tp + fp) if (tp + fp) else 0
    sensibilidad = tp / (tp + fn) if (tp + fn) else 0
    especificidad = tn / (tn + fp) if (tn + fp) else 0
    tasa_acierto_at_k = aciertos_at_k / n_evaluadas_at_k if n_evaluadas_at_k else 0

    costo_in = (tokens_in / 1_000_000) * TARIFA_IN_CLP
    costo_out = (tokens_out / 1_000_000) * TARIFA_OUT_CLP
    costo_total = costo_in + costo_out

    latencia_prom = sum(latencias) / len(latencias) if latencias else 0
    latencia_max = max(latencias) if latencias else 0

    resumen = {
        "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "modelo": config.LLM_MODELO,
        "embeddings": config.EMBEDDING_MODELO,
        "total_consultas": total,
        "consultas_con_error_proveedor": errores_proveedor,
        "respondibles_esperados": tp + fn,
        "abstenciones_esperadas": tn + fp,
        "matriz_confusion": {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
        "exactitud": round(exactitud, 4),
        "precision": round(precision_abst, 4),
        "sensibilidad_recall": round(sensibilidad, 4),
        "especificidad": round(especificidad, 4),
        "tasa_acierto_at_k": round(tasa_acierto_at_k, 4),
        "tokens_entrada": tokens_in,
        "tokens_salida": tokens_out,
        "costo_estimado_clp": round(costo_total),
        "costo_in_clp": round(costo_in),
        "costo_out_clp": round(costo_out),
        "tarifa_in_clp_millon": TARIFA_IN_CLP,
        "tarifa_out_clp_millon": TARIFA_OUT_CLP,
        "latencia_promedio_seg": round(latencia_prom, 2),
        "latencia_max_seg": round(latencia_max, 2),
        "tiempo_total_seg": segundos,
        "detalle": detalle,
    }

    base = f"evaluacion_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    ruta_json = dir_salida / f"{base}.json"
    ruta_json.write_text(json.dumps(resumen, ensure_ascii=False, indent=2),
                         encoding="utf-8")

    lineas = [
        "=" * 72,
        "EVALUACIÓN DEL ASISTENTE RAG - TECNOAPOYO",
        f"Fecha: {resumen['fecha']} | Modelo: {resumen['modelo']} | "
        f"Embeddings: {resumen['embeddings']}",
        "=" * 72,
        f"Total de consultas        : {total}",
        f"  respondibles esperadas  : {resumen['respondibles_esperados']}",
        f"  abstenciones esperadas  : {resumen['abstenciones_esperadas']}",
        f"  con error del proveedor : {errores_proveedor} (excluidas de las métricas)",
        "",
        f"Matriz de confusión (2x2) : TP={tp}  TN={tn}  FP={fp}  FN={fn}",
        f"Exactitud (accuracy)      : {exactitud:.1%}",
        f"Precisión (precision)     : {precision_abst:.1%}",
        f"Sensibilidad (recall)     : {sensibilidad:.1%}",
        f"Especificidad             : {especificidad:.1%}",
        f"Tasa de acierto @k (top-{config.TOP_K})  : {tasa_acierto_at_k:.1%}",
        "",
        f"Tokens de entrada         : {tokens_in}",
        f"Tokens de salida          : {tokens_out}",
        f"Costo estimado (CLP)      : ${costo_total:,.0f}",
        f"  entrada (${TARIFA_IN_CLP:,.0f}/M) : ${costo_in:,.2f}",
        f"  salida  (${TARIFA_OUT_CLP:,.0f}/M) : ${costo_out:,.2f}",
        "",
        f"Latencia promedio         : {latencia_prom:.2f}s",
        f"Latencia máxima           : {latencia_max:.2f}s",
        f"Tiempo total de la prueba  : {segundos}s",
    ]
    ruta_txt = dir_salida / f"{base}.txt"
    ruta_txt.write_text("\n".join(lineas), encoding="utf-8")

    print("\n" + "\n".join(lineas))
    print(f"\n[OK] Resultados -> {ruta_json}")
    print(f"[OK] Resumen    -> {ruta_txt}")


if __name__ == "__main__":
    evaluar()