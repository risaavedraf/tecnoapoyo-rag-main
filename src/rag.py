# -*- coding: utf-8 -*-
"""
Núcleo RAG del asistente de TecnoApoyo (Fase 3).

Sigue el patrón del ramo (RA1/IL1.3 e IL1.4):
  recuperación FAISS -> prompt fundamentado -> generación con abstinencia.

Componentes:
  - RAGEngine: carga el índice FAISS, recupera, construye el prompt con
    instrucciones estrictas de abstinencia y devuelve respuesta + metadatos
    de trazabilidad (documentos recuperados, latencia, tokens) más el campo
    error, que describe el fallo del proveedor si la llamada al LLM falla.
  - Criterio de abstinencia: si la recuperación baja de un umbral de
    similitud, o el modelo marca NO_CONTEXT, se responde explicitando que
    no hay evidencia suficiente.
"""
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain.chains import RetrievalQA  # noqa: E402
from langchain_core.callbacks import BaseCallbackHandler  # noqa: E402
from langchain_community.vectorstores import FAISS  # noqa: E402
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # noqa: E402

from src import config  # noqa: E402


PROMPT_FUNDAMENTADO = """Actúa como el asistente documental de TecnoApoyo, una pyme de reparación de computadores en Santiago.

Solo puedes responder usando la información del contexto recuperado a continuación. El contexto proviene de la documentación oficial de TecnoApoyo.

<contexto>
{context}
</contexto>

Reglas estrictas:
1. Si el contexto contiene la respuesta o información suficiente para responder la intención de la pregunta, responde de forma breve, clara y en español, citando los datos del contexto.
2. Solo debes abstenerte si el contexto NO contiene ninguna información relacionada con la pregunta (por ejemplo, estado de una orden de servicio, precios no documentados, opiniones o datos completamente fuera de alcance). Ante la duda, responde con la información disponible del contexto en lugar de abstenerte.
3. Si te abstienes, responde EXACTAMENTE con la frase: "No tengo información suficiente en la documentación de TecnoApoyo para responder eso."
4. No inventes datos, precios ni políticas. No uses conocimiento general fuera del contexto.
5. Responde solo la respuesta final, sin explicar tu proceso.

Pregunta: {question}
"""


@dataclass
class ResultadoConsulta:
    """Respuesta del asistente con trazabilidad completa."""
    pregunta: str
    respuesta: str
    abstencion: bool
    documentos_recuperados: List[dict] = field(default_factory=list)
    latencia_seg: float = 0.0
    tokens_entrada: int = 0
    tokens_salida: int = 0
    tokens_total: int = 0
    # Descripción del fallo del proveedor (red/API) si la generación falló;
    # None cuando la consulta se resolvió normalmente.
    error: Optional[str] = None


class ColectorTokens(BaseCallbackHandler):
    """Captura el uso real de tokens desde los callbacks de LangChain."""

    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0

    def on_llm_end(self, response, **kwargs):
        uso = (getattr(response, "llm_output", None) or {}).get("token_usage", {})
        self.prompt_tokens += uso.get("prompt_tokens", 0)
        self.completion_tokens += uso.get("completion_tokens", 0)
        self.total_tokens += uso.get("total_tokens", 0)


class RAGEngine:
    def __init__(self, carga_indice=True):
        self.embeddings = OpenAIEmbeddings(
            model=config.EMBEDDING_MODELO,
            base_url=config.LLM_BASE_URL,
            api_key=config.LLM_API_KEY,
            check_embedding_ctx_length=False,
        )
        self.llm = ChatOpenAI(
            model=config.LLM_MODELO,
            base_url=config.LLM_BASE_URL,
            api_key=config.LLM_API_KEY,
            temperature=config.TEMPERATURA,
        )
        self.vector_store = None
        self.qa = None
        self.historial: List[dict] = []
        self.colector_tokens = ColectorTokens()
        if carga_indice:
            self.cargar_indice()

    # -- Índice -----------------------------------------------------------
    def cargar_indice(self):
        if not config.RUTA_INDICE.exists():
            raise FileNotFoundError(
                f"No existe el índice FAISS en {config.RUTA_INDICE}. "
                "Ejecuta primero: python scripts/build_index.py"
            )
        self.vector_store = FAISS.load_local(
            str(config.RUTA_INDICE),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        self._construir_cadena()

    def _construir_cadena(self):
        """Reproduce el patrón RetrievalQA del ramo con nuestro prompt."""
        from langchain.prompts import PromptTemplate  # noqa: E402
        prompt = PromptTemplate(
            template=PROMPT_FUNDAMENTADO,
            input_variables=["context", "question"],
        )
        retriever = self.vector_store.as_retriever(
            search_kwargs={"k": config.TOP_K}
        )
        self.qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt},
        )

    # -- Recuperación ------------------------------------------------------
    def recuperar(self, pregunta: str, k: Optional[int] = None):
        """Devuelve documentos con score de relevancia normalizado 0-1.

        similarity_search_with_relevance_scores() entrega MAYOR = más
        relevante, que es la convención usada para el umbral de abstinencia.
        """
        k = k or config.TOP_K
        resultados = self.vector_store.similarity_search_with_relevance_scores(
            pregunta, k=k
        )
        return [
            {
                "texto": doc.page_content,
                "score": round(float(score), 4),
                "document_id": doc.metadata.get("document_id", "?"),
                "titulo": doc.metadata.get("titulo", "?"),
                "tema": doc.metadata.get("tema", "?"),
                "version": doc.metadata.get("version", "?"),
                "fecha": doc.metadata.get("fecha", "?"),
            }
            for doc, score in resultados
        ]

    # -- Consulta principal ------------------------------------------------
    def consultar(self, pregunta: str, k: Optional[int] = None,
                  umbral_abstencion: Optional[float] = None) -> ResultadoConsulta:
        """Ejecuta el pipeline RAG completo con métricas internas.

        umbral_abstencion es configurable (config.UMBRAL_ABSTENCION, variable
        de entorno UMBRAL_ABSTENCION) y depende del modelo de embeddings: cada
        modelo distribuye los scores de relevancia de otra forma, así que un
        umbral calibrado con uno no sirve para otro. El argumento explícito
        manda sobre la configuración.
        """
        if umbral_abstencion is None:
            umbral_abstencion = config.UMBRAL_ABSTENCION
        inicio = time.time()
        error = None

        recuperados = self.recuperar(pregunta, k=k)
        mejor_score = recuperados[0]["score"] if recuperados else 0.0

        if not recuperados or mejor_score < umbral_abstencion:
            respuesta = "No tengo información suficiente en la documentación de TecnoApoyo para responder eso."
            abstencion = True
            tokens_entrada = tokens_salida = tokens_total = 0
        else:
            self.colector_tokens.reset()
            # Un proveedor remoto puede responder 401/429/5xx o cortar la
            # conexión por timeout, y eso abortaría toda la sesión de consola
            # sin dejar trazabilidad. Un Ollama local que se cae es otro
            # escenario (se relanza el proceso), así que aquí solo se aísla el
            # error de la llamada HTTP y se informa al usuario.
            try:
                resultado = self.qa.invoke(
                    {"query": pregunta},
                    config={"callbacks": [self.colector_tokens]},
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                respuesta = (
                    "No pude obtener respuesta del proveedor de lenguaje. "
                    "Revisa LLM_BASE_URL, LLM_API_KEY y la cuota disponible."
                )
                # abstencion=False: el motor no se abstuvo por falta de
                # evidencia, la llamada falló; marcarla como abstención
                # falsearía la etiqueta de abstinencia y las métricas.
                abstencion = False
                tokens_entrada = tokens_salida = tokens_total = 0
            else:
                respuesta = resultado["result"].strip()
                abstencion = respuesta.startswith(
                    "No tengo información suficiente en la documentación"
                )
                tokens_entrada = self.colector_tokens.prompt_tokens
                tokens_salida = self.colector_tokens.completion_tokens
                tokens_total = self.colector_tokens.total_tokens

        latencia = round(time.time() - inicio, 2)
        # El error también queda en el historial: la sesión exportada por
        # /guardar debe contar la misma causa que la consola, no solo el mensaje
        # genérico. Los consumidores existentes (app.py) siguen leyendo
        # pregunta/respuesta/abstencion, así que la clave nueva no los rompe.
        self.historial.append({"pregunta": pregunta, "respuesta": respuesta,
                               "abstencion": abstencion, "error": error})

        return ResultadoConsulta(
            pregunta=pregunta,
            respuesta=respuesta,
            abstencion=abstencion,
            documentos_recuperados=recuperados,
            latencia_seg=latencia,
            tokens_entrada=tokens_entrada,
            tokens_salida=tokens_salida,
            tokens_total=tokens_total,
            error=error,
        )


def mostrar_resultado(consulta: ResultadoConsulta):
    """Impresión legible del resultado con trazabilidad visible."""
    print("\n" + "=" * 70)
    print(f"Pregunta: {consulta.pregunta}")
    print("-" * 70)
    if consulta.error:
        print(f"ERROR DEL PROVEEDOR DEL LLM: {consulta.error}")
    if consulta.abstencion:
        print("ABSTENCIÓN (sin evidencia suficiente):")
    print(consulta.respuesta)
    print("-" * 70)
    print("Documentos recuperados (trazabilidad):")
    for doc in consulta.documentos_recuperados:
        print(f"   [{doc['document_id']}] score={doc['score']} "
              f"(v{doc['version']}, {doc['fecha']})")
    print(f"Latencia: {consulta.latencia_seg}s | tokens: "
          f"{consulta.tokens_total} ({consulta.tokens_entrada} in / "
          f"{consulta.tokens_salida} out)")
    print("=" * 70)


if __name__ == "__main__":
    engine = RAGEngine()
    while True:
        try:
            pregunta = input("Tú> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not pregunta:
            continue
        if pregunta.lower() in ("salir", "exit", "quit"):
            break
        resultado = engine.consultar(pregunta)
        mostrar_resultado(resultado)