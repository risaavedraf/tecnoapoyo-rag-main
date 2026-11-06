# -*- coding: utf-8 -*-
"""
Construcción del índice vectorial (Fase 2 - RAG).

Flujo (idéntico al patrón RA1/IL1.3 del ramo):
  corpus -> chunking con RecursiveCharacterTextSplitter -> embeddings
  (config.EMBEDDING_MODELO, por defecto mistral-embed) -> índice FAISS ->
  documentos enriquecidos con
  metadatos de origen (document_id, titulo, tema, version, fecha).

Además compara las configuraciones de chunking definidas en config.py y
persiste en data/output/chunking_report.json los resultados (número de
chunks y longitud media) para respaldar la decisión de diseño (IE5).

Uso:
    python scripts/build_index.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_community.vectorstores import FAISS  # noqa: E402
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # noqa: E402
from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: E402

from src import config  # noqa: E402


# ---------------------------------------------------------------------------
# Carga del corpus
# ---------------------------------------------------------------------------
def cargar_corpus():
    """Lee todos los .md de procedimientos y FAQ y devuelve (documento, metadatos)."""
    documentos = []
    archivos = sorted(
        list(config.DIR_PROCEDIMIENTOS.glob("*.md"))
        + list(config.DIR_FAQ.glob("*.md"))
    )
    for archivo in archivos:
        texto = archivo.read_text(encoding="utf-8")
        if "PROC-" in archivo.name:
            doc_id = archivo.stem
            titulo = texto.split("\n")[0].lstrip("# ").strip()
            version, tema, fecha = _metadatos_proc(texto)
        else:  # FAQ-xxx.md
            doc_id = archivo.stem
            titulo = "Pregunta frecuente"
            version, tema, fecha = "1.0", "faq", "2026-04-01"
        documentos.append({
            "texto": texto,
            "source": archivo.name,
            "document_id": doc_id,
            "titulo": titulo,
            "version": version,
            "tema": tema,
            "fecha": fecha,
        })
    return documentos


def _metadatos_proc(texto):
    import re
    version = "1.0"
    tema = "procedimiento"
    fecha = "2026-01-01"
    patron = {"Versión": "version", "Tema": "tema", "Fecha": "fecha"}
    for linea in texto.splitlines():
        for etiqueta, campo in patron.items():
            m = re.search(r"^\-\s+\*\*" + etiqueta + r"\*\*\s*:\s*(.+)$", linea)
            if m:
                if campo == "version":
                    version = m.group(1).strip()
                elif campo == "tema":
                    tema = m.group(1).strip()
                else:
                    fecha = m.group(1).strip()
    return version, tema, fecha


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def dividir(texto, chunk_size, chunk_overlap):
    # Chunking adaptativo: los documentos cortos (p. ej. FAQ) se conservan
    # enteros para no separar la pregunta de su respuesta; solo se divide
    # si el texto supera 1.5x el tamaño del chunk.
    if len(texto) <= int(chunk_size * 1.5):
        return [texto]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(texto)


def probar_chunking(documentos):
    """Compara las configuraciones de chunking y guarda un reporte."""
    reporte = []
    for cfg in config.CONFIGURACIONES_CHUNKING:
        todo_chunks = []
        longitudes = []
        for doc in documentos:
            chunks = dividir(doc["texto"], cfg["chunk_size"], cfg["chunk_overlap"])
            todo_chunks.extend(chunks)
            longitudes.extend(len(c) for c in chunks)
        reporte.append({
            "configuracion": cfg["nombre"],
            "chunk_size": cfg["chunk_size"],
            "chunk_overlap": cfg["chunk_overlap"],
            "num_chunks": len(todo_chunks),
            "longitud_media": round(sum(longitudes) / len(longitudes), 1) if longitudes else 0,
            "longitud_max": max(longitudes) if longitudes else 0,
        })
    # Elegimos la configuración con longitud media más cercana a chunk_size (menor pérdida)
    elegida = min(reporte, key=lambda r: abs(r["longitud_media"] - r["chunk_size"]))
    ruta = config.DIR_OUTPUT / "chunking_report.json"
    ruta.write_text(json.dumps({"configuraciones": reporte, "elegida": elegida},
                               ensure_ascii=False, indent=2), encoding="utf-8")
    print("[OK] Reporte de chunking ->", ruta)
    for r in reporte:
        print("   ", r["configuracion"], "| chunks:", r["num_chunks"],
              "| long media:", r["longitud_media"])
    print("-> Elegida:", elegida["configuracion"])
    return elegida


# ---------------------------------------------------------------------------
# Construcción del índice
# ---------------------------------------------------------------------------
def construir_indice(documentos, chunk_size, chunk_overlap):
    embeddings = OpenAIEmbeddings(
        model=config.EMBEDDING_MODELO,
        base_url=config.LLM_BASE_URL,
        api_key=config.LLM_API_KEY,
        check_embedding_ctx_length=False,
    )

    textos = []
    metadatos = []
    for doc in documentos:
        chunks = dividir(doc["texto"], chunk_size, chunk_overlap)
        for chunk in chunks:
            textos.append(chunk)
            metadatos.append({
                "source": doc["source"],
                "document_id": doc["document_id"],
                "titulo": doc["titulo"],
                "tema": doc["tema"],
                "version": doc["version"],
                "fecha": doc["fecha"],
            })

    vector_db = FAISS.from_texts(texts=textos, embedding=embeddings, metadatas=metadatos)

    config.DIR_OUTPUT.mkdir(parents=True, exist_ok=True)
    vector_db.save_local(str(config.RUTA_INDICE))
    info = {
        "modelo_embeddings": config.EMBEDDING_MODELO,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "num_chunks": len(textos),
        "num_documentos": len(metadatos),
    }
    config.RUTA_DOCUMENTOS.write_text(json.dumps(info, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    print(f"[OK] Índice FAISS creado con {len(textos)} chunks -> {config.RUTA_INDICE}")
    return vector_db


# ---------------------------------------------------------------------------
# Verificación
# ---------------------------------------------------------------------------
def verificar_indice(vector_db):
    """Consulta de humo: reproduce el patrón RetrievalQA del ramo."""
    llm = ChatOpenAI(
        model=config.LLM_MODELO,
        base_url=config.LLM_BASE_URL,
        api_key=config.LLM_API_KEY,
        temperature=config.TEMPERATURA,
    )
    from langchain.chains import RetrievalQA  # noqa: E402
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(search_kwargs={"k": config.TOP_K}),
    )
    query = "¿Cuánto cuesta el diagnóstico básico?"
    print("\nConsulta de prueba:", query)
    print("Respuesta:", qa.invoke({"query": query})["result"])


def main():
    config.DIR_OUTPUT.mkdir(parents=True, exist_ok=True)
    documentos = cargar_corpus()
    print(f"[OK] Corpus cargado: {len(documentos)} documentos")

    elegida = probar_chunking(documentos)
    vector_db = construir_indice(documentos, elegida["chunk_size"], elegida["chunk_overlap"])
    verificar_indice(vector_db)


if __name__ == "__main__":
    main()