"""Evaluación del motor RAG (Fase 5): pausa entre consultas del CLI.

Ningún test toca la red ni al proveedor: el motor y el arranque se reemplazan
por dobles, y el directorio de resultados es un `tmp_path`. La pausa existe
porque el proveedor limita por minuto: sin ella, 41 consultas seguidas se
comen el límite y la corrida termina en 429, que no es evidencia válida.
"""
import json
import time as _time
from types import SimpleNamespace

from scripts import evaluate
from src import config


class RespuestaFalsa:
    """Doble de la respuesta del motor: solo los campos que el resumen lee."""

    def __init__(self, *, abstencion=False, error=None):
        self.abstencion = abstencion
        self.error = error
        self.documentos_recuperados = [{"document_id": "d1", "score": 0.9}]
        self.latencia_seg = 0.1
        self.tokens_total = 10
        self.tokens_entrada = 5
        self.tokens_salida = 5


def preparar(monkeypatch, tmp_path, respuestas):
    """Aísla la corrida: sin red, sin clave, sin escribir en el repo."""
    consultas = [
        {"query": f"pregunta {i}", "responde": True, "documentos": ["d1"]}
        for i in range(1, len(respuestas) + 1)
    ]
    esperas = []

    monkeypatch.setattr(evaluate, "cargar_dataset", lambda: consultas)
    monkeypatch.setattr(
        evaluate,
        "arranque",
        SimpleNamespace(reconectar_flujos=lambda: None, exigir_clave=lambda: None),
    )
    monkeypatch.setattr(
        evaluate, "RAGEngine", lambda: SimpleNamespace(consultar=lambda _: respuestas.pop(0))
    )
    monkeypatch.setattr(
        evaluate, "time", SimpleNamespace(sleep=esperas.append, time=_time.time,
                                          perf_counter=_time.perf_counter)
    )
    # El total del reporte se mide con `duracion_desde`, que vive en
    # `src.observabilidad`: el mismo reloj falso en los dos módulos, o el total
    # saldría de mezclar dos bases de tiempo.
    monkeypatch.setattr("src.observabilidad.time", evaluate.time)
    monkeypatch.setattr(config, "DIR_EVAL_RESULTADOS", tmp_path)
    return esperas


def test_parser_acepta_pausa_en_segundos():
    args = evaluate._parser().parse_args(["--pausa", "30"])
    assert args.pausa == 30.0


def test_parser_sin_pausa_duerme_cero():
    assert evaluate._parser().parse_args([]).pausa == 0.0


def test_pausa_duerme_entre_consultas_y_no_en_los_extremos(monkeypatch, tmp_path):
    esperas = preparar(
        monkeypatch, tmp_path, [RespuestaFalsa() for _ in range(3)]
    )

    evaluate.evaluar(pausa_seg=30.0)

    # Tres consultas tienen dos intervalos: dormir antes de la primera o
    # después de la última solo alargaría la corrida sin espaciar nada.
    assert esperas == [30.0, 30.0]


def test_pausa_cero_no_duerme(monkeypatch, tmp_path):
    esperas = preparar(
        monkeypatch, tmp_path, [RespuestaFalsa() for _ in range(3)]
    )

    evaluate.evaluar()

    assert esperas == []


def test_pausa_negativa_es_error_de_uso(monkeypatch, tmp_path):
    preparar(monkeypatch, tmp_path, [RespuestaFalsa()])

    codigo = evaluate.main(["--pausa", "-1"])

    assert codigo == 1
    assert list(tmp_path.iterdir()) == []


def test_main_con_pausa_corre_y_deja_la_pausa_en_el_resumen(monkeypatch, tmp_path):
    esperas = preparar(
        monkeypatch, tmp_path, [RespuestaFalsa() for _ in range(2)]
    )

    codigo = evaluate.main(["--pausa", "30"])

    assert codigo == 0
    assert esperas == [30.0]
    (ruta_json,) = list(tmp_path.glob("*.json"))
    resumen = json.loads(ruta_json.read_text(encoding="utf-8"))
    assert resumen["pausa_seg"] == 30.0
