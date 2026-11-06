# -*- coding: utf-8 -*-
"""
TecnoApoyo - Corpus sintético (Fase 1 del encargo).

Genera la base de conocimiento documental de TecnoApoyo, PYME ficticia de
reparación de computadores en Santiago, con procedimientos/políticas y FAQ
100% sintéticos con metadatos (document_id, titulo, tema, version, fecha).

Salidas:
  - data/corpus/procedimientos/PROC-xxx.md
  - data/corpus/faq/FAQ-xxx.md
  - data/entidades.json
  - evaluation/dataset_evaluacion.json   (dataset de evaluación, 30-50 consultas)

Patrón seguido (RA1/IL1.3): corpus -> chunking (RecursiveCharacterTextSplitter)
con metadatos -> embeddings -> indice vectorial FAISS.
"""
import json
import random
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
DIR_PROC = RAIZ / "data" / "corpus" / "procedimientos"
DIR_FAQ = RAIZ / "data" / "corpus" / "faq"
DIR_EVAL = RAIZ / "evaluation"

PROCEDIMIENTOS = [
    {
        "document_id": "PROC-001",
        "titulo": "Procedimiento de recepción de equipos",
        "tema": "procedimiento",
        "version": "1.2",
        "fecha": "2026-01-15",
        "contenido": (
            "El procedimiento de recepción de equipos inicia cuando un cliente "
            "ingresa al taller con un equipo para diagnóstico o reparación. El "
            "personal de atención registra los datos del cliente, describe el "
            "síntoma reportado y asigna un número de orden de servicio. El "
            "técnico realiza un diagnóstico inicial básico de hasta 15 minutos. "
            "El costo del diagnóstico es de 15000 pesos chilenos y se descuenta "
            "del valor total si el cliente decide reparar el equipo en "
            "TecnoApoyo. El equipo queda internado en el taller y se entrega un "
            "comprobante de recepción al cliente. Se informa al cliente el "
            "tiempo estimado de diagnóstico de 48 horas hábiles."
        ),
    },
    {
        "document_id": "PROC-002",
        "titulo": "Política de garantía de fábrica",
        "tema": "politica",
        "version": "2.0",
        "fecha": "2026-02-10",
        "contenido": (
            "La garantía de fábrica cubre los defectos originales de fabricación "
            "durante 24 meses desde la fecha de compra. No cubre mal uso, golpes "
            "ni daños por líquidos o sobrecargas eléctricas. Para hacer efectiva "
            "la garantía se debe presentar la boleta de compra y el equipo no "
            "debe presentar señales de manipulación no autorizada. Las piezas "
            "instaladas que fallan dentro del período de garantía se reemplazan "
            "sin costo. La garantía extendida agrega 12 meses adicionales al "
            "período de fábrica y cubre fallas de hardware, excluyendo siempre "
            "daños por líquido o golpes."
        ),
    },
    {
        "document_id": "PROC-003",
        "titulo": "Procedimiento de mantenimiento preventivo",
        "tema": "procedimiento",
        "version": "1.0",
        "fecha": "2026-01-20",
        "contenido": (
            "El mantenimiento preventivo se recomienda cada 6 meses o cuando el "
            "equipo presente sobrecalentamiento. Incluye limpieza interna del "
            "ventilador y disipador, reemplazo de pasta térmica, revisión de la "
            "temperatura del procesador y actualización de controladores. El "
            "servicio tiene un costo de 25000 pesos chilenos para notebooks y de "
            "30000 pesos para equipos de escritorio. Se recomienda respaldar la "
            "información del cliente antes de iniciar cualquier intervención. El "
            "procedimiento completo toma entre 60 y 90 minutos."
        ),
    },
    {
        "document_id": "PROC-004",
        "titulo": "Política de presupuestos y diagnósticos",
        "tema": "politica",
        "version": "1.3",
        "fecha": "2026-03-01",
        "contenido": (
            "El diagnóstico básico tiene un costo de 15000 pesos chilenos y se "
            "descuenta si se repara en TecnoApoyo. Para reparaciones superiores "
            "a 25000 pesos chilenos, se entrega un presupuesto detallado que "
            "debe ser aprobado por el cliente antes de continuar. La reparación "
            "solo inicia después de la aprobación escrita del presupuesto. Los "
            "presupuestos tienen una validez de 10 días hábiles. En caso de "
            "requerir piezas adicionales no consideradas, se genera un nuevo "
            "presupuesto para su aprobación. El cliente puede retirar su equipo "
            "sin reparar pagando únicamente el diagnóstico."
        ),
    },
    {
        "document_id": "PROC-005",
        "titulo": "Procedimiento de garantías y reportes",
        "tema": "procedimiento",
        "version": "1.1",
        "fecha": "2026-03-15",
        "contenido": (
            "Las garantías de fábrica se gestionan según la política vigente. "
            "Para equipos con garantía extendida, el cliente debe presentar la "
            "boleta y el certificado de garantía. Las piezas alternativas "
            "compatibles tienen garantía de 3 meses. Las piezas originales "
            "tienen garantía de 6 meses. Los reportes de garantía se procesan "
            "dentro de un plazo máximo de 5 días hábiles desde la recepción del "
            "equipo. Para hacer efectiva una garantía se debe presentar la "
            "boleta de compra original y el equipo no debe presentar señales de "
            "manipulación."
        ),
    },
    {
        "document_id": "PROC-006",
        "titulo": "Política de precios y costos",
        "tema": "politica",
        "version": "1.2",
        "fecha": "2026-02-20",
        "contenido": (
            "El diagnóstico básico cuesta 15000 pesos chilenos. El mantenimiento "
            "preventivo de notebook cuesta 25000 pesos. El mantenimiento de "
            "escritorio cuesta 30000 pesos. Las reparaciones se presupuestan "
            "según el costo de piezas y el tiempo estimado. Los precios incluyen "
            "IVA. El pago mínimo para iniciar una reparación con presupuesto de "
            "50000 pesos o más es de un 50% de abono anticipado. No se realizan "
            "pagos en cuotas. Las tarifas pueden variar sin previo aviso, pero "
            "el presupuesto aprobado mantiene su valor por 10 días hábiles."
        ),
    },
    {
        "document_id": "PROC-007",
        "titulo": "Política de retiro a domicilio",
        "tema": "politica",
        "version": "1.1",
        "fecha": "2026-03-10",
        "contenido": (
            "El servicio de retiro a domicilio está disponible para clientes "
            "dentro del radio de Santiago y comunas aledañas. El costo del "
            "retiro es de 5000 pesos chilenos por traslado. El retiro debe "
            "coordinarse con al menos un día de anticipación. Al confirmar el "
            "retiro se solicita la dirección, un número de contacto y una ventana "
            "horaria de preferencia. El técnico que realiza el retiro verifica "
            "visualmente el equipo y entrega un comprobante. No se realizan "
            "retiros fuera del área definida por política."
        ),
    },
    {
        "document_id": "PROC-008",
        "titulo": "Política de respaldo de datos",
        "tema": "politica",
        "version": "1.0",
        "fecha": "2026-02-05",
        "contenido": (
            "Antes de cualquier intervención técnica se recomienda respaldar la "
            "información del cliente. El respaldo se realiza en un disco externo "
            "con encriptación y se conserva en el taller por hasta 15 días. El "
            "cliente puede solicitar el respaldo de sus datos por un costo "
            "adicional de 10000 pesos chilenos. Los datos respaldados se "
            "eliminan de forma segura una vez entregado el equipo. El taller no "
            "se hace responsable de la pérdida de datos si el cliente no autoriza "
            "el respaldo previo."
        ),
    },
    {
        "document_id": "PROC-009",
        "titulo": "Procedimiento de actualización de software",
        "tema": "procedimiento",
        "version": "1.0",
        "fecha": "2026-01-25",
        "contenido": (
            "La actualización de software incluye la instalación de las versiones "
            "más recientes del sistema operativo y de los programas instalados. "
            "Antes de actualizar, se verifica que el equipo cumpla los requisitos "
            "mínimos y se respalda la configuración. Las actualizaciones se "
            "realizan durante el horario de atención y pueden requerir la entrega "
            "del equipo por varias horas. Se recomienda actualizar cada mes. El "
            "servicio de actualización tiene un costo de 10000 pesos chilenos y "
            "se agrega a la orden de servicio."
        ),
    },
    {
        "document_id": "PROC-010",
        "titulo": "Política de datos y privacidad",
        "tema": "politica",
        "version": "1.1",
        "fecha": "2026-03-05",
        "contenido": (
            "Los datos personales de los clientes se tratan conforme a la Ley 19"
            "628 sobre protección de la vida privada. Los datos se utilizan "
            "únicamente para la gestión de la orden de servicio y no se "
            "comparten con terceros sin consentimiento. La información del "
            "cliente se almacena de forma segura y se elimina según la política "
            "de retención. El cliente tiene derecho a solicitar acceso, "
            "rectificación o eliminación de sus datos. La confidencialidad de "
            "la información técnica del equipo también está protegida."
        ),
    },
]

FAQ = [
    ("¿Cuál es el horario de atención?", "El horario de atención es de lunes a viernes de 09:00 a 18:00 horas y los sábados de 10:00 a 14:00 horas."),
    ("¿Dónde están ubicados?", "Estamos ubicados en la comuna de Santiago, Región Metropolitana. Consulta la dirección exacta en nuestra página de contacto."),
    ("¿Cuánto demora un diagnóstico?", "El diagnóstico tiene una duración estimada de 48 horas hábiles. El informe se entrega al cliente por correo o al retirar el equipo."),
    ("¿Qué cubre la garantía?", "La garantía de fábrica cubre defectos de fabricación durante 24 meses. Las piezas instaladas tienen garantía de 6 meses (originales) o 3 meses (compatibles)."),
    ("¿Cómo extiendo la garantía?", "La garantía extendida agrega 12 meses adicionales al período de fábrica y se debe contratar dentro del primer año de compra."),
    ("¿Atienden sábados?", "Sí, los sábados atendemos de 10:00 a 14:00 horas en el taller."),
    ("¿Necesito boleta para la garantía?", "Sí, se debe presentar la boleta de compra original para hacer efectiva la garantía."),
    ("¿Qué pasa si el equipo vuelve a fallar?", "Si la falla recurrente ocurre dentro del período de garantía de la reparación, la pieza se reemplaza sin costo."),
    ("¿Se puede pagar en cuotas?", "No, no se realizan pagos en cuotas. El pago se realiza al momento de retirar el equipo."),
    ("¿Ofrecen diagnóstico gratis?", "No, el diagnóstico tiene un costo de 15000 pesos chilenos, que se descuenta si reparas en TecnoApoyo."),
    ("¿Trabajan con piezas originales?", "Ofrecemos piezas originales y alternativas compatibles. Las originales tienen garantía de 6 meses y las compatibles de 3 meses."),
    ("¿Cuánto tarda una reparación?", "Los plazos dependen de la disponibilidad de piezas. El técnico entrega una estimación al momento del diagnóstico."),
    ("¿Retiran el equipo a domicilio?", "Sí, el retiro a domicilio está disponible dentro del radio de Santiago por 5000 pesos chilenos por traslado."),
    ("¿Qué pasa si no retiro mi equipo?", "Los equipos no reclamados dentro de 60 días quedan sujetos a la política de equipos abandonados y pueden ser disponibles para terceros."),
    ("¿Ofrecen respaldo de datos?", "Sí, el respaldo de datos tiene un costo de 10000 pesos chilenos y se realiza antes de cada intervención que lo requiera."),
    ("¿Cómo reclamo una garantía?", "Debes presentar la boleta de compra en el taller y el equipo no debe presentar daños por mal uso. El reporte se procesa en hasta 5 días hábiles."),
    ("¿Qué incluye el mantenimiento preventivo?", "Incluye limpieza interna, reemplazo de pasta térmica, revisión de temperatura y actualización de controladores. Notebook 25000, escritorio 30000 pesos."),
    ("¿Garantizan la reparación?", "Sí, las reparaciones tienen garantía de 6 meses sobre las piezas originales instaladas."),
    ("¿Qué medios de pago aceptan?", "Aceptamos efectivo, tarjeta de débito y tarjeta de crédito. No se realizan pagos en cuotas."),
    ("¿Hacen mantenimiento a notebooks gaming?", "Sí, se realiza mantenimiento preventivo a todos los notebooks, incluidos los de gama gaming, con foco en el sistema de refrigeración."),
    ("¿Atienden equipos de otras marcas?", "Sí, TecnoApoyo repara equipos de distintas marcas, incluidas las principales del mercado."),
    ("¿Cuánto demora el abono del diagnóstico?", "El abono del diagnóstico se descuenta automáticamente del presupuesto si reparas el equipo en TecnoApoyo."),
    ("¿Puedo supervisar el diagnóstico?", "El equipo queda internado durante el diagnóstico. Puedes consultar el avance de la orden llamando al taller."),
    ("¿Qué cubren los 60 días de retiros?", "Los equipos deben retirarse dentro de 60 días desde la notificación de la reparación terminada. Después de ese plazo se aplica la política de abandono."),
    ("¿Se puede cancelar una reparación?", "Sí, puedes cancelar la reparación antes de que se inicie. Si ya inició, se cobran el diagnóstico y las piezas ya instaladas."),
    ("¿Dan garantía por mano de obra?", "La mano de obra tiene garantía de 6 meses, siempre que el equipo no presente daños por mal uso."),
    ("¿Cuál es el costo del diagnóstico?", "El diagnóstico básico cuesta 15000 pesos chilenos y se descuenta del total si reparas en TecnoApoyo."),
    ("¿Qué pasa con los datos del cliente?", "Los datos se protegen conforme a la Ley 19.628. El respaldo se recomienda antes de toda intervención."),
    ("¿Tienen atención telefónica?", "Sí, la atención telefónica se realiza de lunes a viernes de 09:00 a 18:00 horas."),
    ("¿Los precios incluyen IVA?", "Sí, todos los precios publicados incluyen IVA."),
]

# Consultas de evaluación: 41 total.
#   - respondibles  -> deben producir respuesta (con documentos recuperados)
#   - no respondibles -> deben abstenerse (sin respuesta)
DATASET_EVALUACION = [
    # NOTA: "documentos" lista TODOS los documentos relevantes (PROC y FAQ
    # equivalentes) para que la tasa de acierto @k evalúe si se recuperó al
    # menos una fuente legítima de la respuesta.
    # --- Respondibles: políticas y procedimientos ---
    {"query": "¿Qué cubre la garantía de fábrica?", "responde": True, "documentos": ["PROC-002", "FAQ-004", "FAQ-005"]},
    {"query": "¿Cuánto cuesta el diagnóstico básico?", "responde": True, "documentos": ["PROC-004", "PROC-006", "FAQ-010", "FAQ-027"]},
    {"query": "¿Cuál es el horario de atención?", "responde": True, "documentos": ["FAQ-001", "FAQ-029"]},
    {"query": "¿Atienden los sábados?", "responde": True, "documentos": ["FAQ-001", "FAQ-006"]},
    {"query": "¿Qué incluye el mantenimiento preventivo?", "responde": True, "documentos": ["PROC-003", "FAQ-017"]},
    {"query": "¿Cuánto demora un diagnóstico?", "responde": True, "documentos": ["PROC-001", "PROC-004", "FAQ-003"]},
    {"query": "¿Cómo extiendo la garantía?", "responde": True, "documentos": ["PROC-002", "FAQ-005"]},
    {"query": "¿Retiran equipos a domicilio?", "responde": True, "documentos": ["PROC-007", "FAQ-013"]},
    {"query": "¿Cuánto demora una reparación?", "responde": True, "documentos": ["PROC-004", "FAQ-012"]},
    {"query": "¿Qué cubre la garantía de una pieza?", "responde": True, "documentos": ["PROC-005", "FAQ-004"]},
    {"query": "¿Ofrecen respaldo de datos?", "responde": True, "documentos": ["PROC-008", "FAQ-015"]},
    {"query": "¿Cuánto cuesta el mantenimiento preventivo?", "responde": True, "documentos": ["PROC-003", "PROC-006", "FAQ-017"]},
    {"query": "¿Se puede pagar en cuotas?", "responde": True, "documentos": ["PROC-006", "FAQ-009", "FAQ-019"]},
    {"query": "¿Trabajan con piezas originales?", "responde": True, "documentos": ["PROC-005", "FAQ-011"]},
    {"query": "¿Qué medios de pago aceptan?", "responde": True, "documentos": ["PROC-006", "FAQ-019"]},
    {"query": "¿Cómo reclamo una garantía?", "responde": True, "documentos": ["PROC-005", "FAQ-016"]},
    {"query": "¿Qué pasa si el equipo vuelve a fallar?", "responde": True, "documentos": ["PROC-005", "FAQ-008", "FAQ-018"]},
    {"query": "¿Cuál es el costo del diagnóstico?", "responde": True, "documentos": ["PROC-004", "PROC-006", "FAQ-010", "FAQ-027"]},
    {"query": "¿Qué pasa con los datos del cliente?", "responde": True, "documentos": ["PROC-010", "FAQ-028"]},
    {"query": "¿Necesito boleta para la garantía?", "responde": True, "documentos": ["PROC-002", "FAQ-007"]},
    {"query": "¿Qué piezas me instalan y me avisan del costo antes?", "responde": True, "documentos": ["PROC-004", "PROC-005", "FAQ-011"]},
    # --- Respondibles: reformulaciones (prueba robustez semántica) ---
    {"query": "¿Me pueden venir a buscar el computador a mi casa?", "responde": True, "documentos": ["PROC-007", "FAQ-013"]},
    {"query": "Dime cuánto se paga por el diagnóstico si no reparo nada", "responde": True, "documentos": ["PROC-004", "PROC-006", "FAQ-010", "FAQ-027"]},
    {"query": "¿Cada cuánto recomiendan llevar el equipo a mantención?", "responde": True, "documentos": ["PROC-003", "FAQ-017"]},
    {"query": "¿Mi garantía cubre si se me cayó té al teclado?", "responde": True, "documentos": ["PROC-002", "FAQ-004"]},
    {"query": "¿Los datos que respaldan se borran después?", "responde": True, "documentos": ["PROC-008", "FAQ-028"]},
    {"query": "¿Dan garantía por la mano de obra de la reparación?", "responde": True, "documentos": ["PROC-005", "FAQ-018", "FAQ-026"]},
    {"query": "¿En qué horario atienden por teléfono?", "responde": True, "documentos": ["FAQ-001", "FAQ-029"]},
    {"query": "¿Aceptan tarjeta de crédito para pagar?", "responde": True, "documentos": ["PROC-006", "FAQ-019"]},
    {"query": "¿Qué pasa si me demoro más de 60 días en retirar el equipo?", "responde": True, "documentos": ["FAQ-014", "FAQ-024"]},
    {"query": "¿El diagnóstico se descuenta si reparo ahí?", "responde": True, "documentos": ["PROC-004", "PROC-006", "FAQ-010", "FAQ-022", "FAQ-027"]},
    # --- No respondibles: fuera de alcance / sin evidencia ---
    {"query": "¿Cuál es el estado de mi orden de servicio 4590?", "responde": False, "documentos": []},
    {"query": "¿Está listo mi notebook para retirar?", "responde": False, "documentos": []},
    {"query": "¿Cuánto vale reparar una pantalla de iPhone 15?", "responde": False, "documentos": []},
    {"query": "¿Atienden celulares?", "responde": False, "documentos": []},
    {"query": "¿Cuál es la cámara de seguridad del local?", "responde": False, "documentos": []},
    {"query": "¿Me pueden recomendar un abogado?", "responde": False, "documentos": []},
    {"query": "¿Cuál es el precio del dólar de hoy?", "responde": False, "documentos": []},
    {"query": "¿Con quién hablo para reclamar a la central?", "responde": False, "documentos": []},
    {"query": "¿Qué tan bueno es su servicio según sus clientes?", "responde": False, "documentos": []},
    {"query": "¿Hacen descuentos por volumen a empresas?", "responde": False, "documentos": []},
]


def main():
    DIR_PROC.mkdir(parents=True, exist_ok=True)
    DIR_FAQ.mkdir(parents=True, exist_ok=True)
    DIR_EVAL.mkdir(parents=True, exist_ok=True)

    for p in PROCEDIMIENTOS:
        md = f"# {p['titulo']}\n\n" \
             f"- **Documento**: `{p['document_id']}`\n" \
             f"- **Tema**: {p['tema']}\n" \
             f"- **Versión**: {p['version']}\n" \
             f"- **Fecha**: {p['fecha']}\n\n" \
             f"{p['contenido']}\n"
        (DIR_PROC / f"{p['document_id']}.md").write_text(md, encoding="utf-8")

    for i, (q, a) in enumerate(FAQ, start=1):
        md = f"# Pregunta frecuente\n\n" \
             f"- **Documento**: `FAQ-{i:03d}`\n" \
             f"- **Tema**: faq\n\n" \
             f"**Pregunta:** {q}\n\n**Respuesta:** {a}\n"
        (DIR_FAQ / f"FAQ-{i:03d}.md").write_text(md, encoding="utf-8")

    (RAIZ / "data" / "entidades.json").parent.mkdir(parents=True, exist_ok=True)
    (RAIZ / "data" / "entidades.json").write_text(
        json.dumps(PROCEDIMIENTOS, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Garantiza el balance entre consultas respondibles y no respondibles
    respondibles = [c for c in DATASET_EVALUACION if c["responde"]]
    no_respondibles = [c for c in DATASET_EVALUACION if not c["responde"]]
    # La descripción se arma con los conteos reales: escrita a mano se quedó
    # desactualizada (decía 40/30/10 con 41/31/10 en el dataset).
    n_total = len(DATASET_EVALUACION)
    n_respond = len(respondibles)
    n_no_respond = len(no_respondibles)
    pct_abstencion = round(100 * n_no_respond / n_total) if n_total else 0
    dataset = {
        "nombre": "Dataset evaluacion TecnoApoyo MVP",
        "descripcion": (f"{n_total} consultas: {n_respond} respondibles + "
                        f"{n_no_respond} no respondibles "
                        f"({pct_abstencion}% abstencion)."),
        "n": n_total,
        "n_respondibles": n_respond,
        "n_no_respondibles": n_no_respond,
        "consultas": DATASET_EVALUACION,
    }
    (DIR_EVAL / "dataset_evaluacion.json").write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"[OK] Procedimientos: {len(PROCEDIMIENTOS)} -> {DIR_PROC}")
    print(f"[OK] FAQ: {len(FAQ)} -> {DIR_FAQ}")
    print(f"[OK] Dataset evaluacion: {len(DATASET_EVALUACION)} consultas "
          f"({len(respondibles)} respondibles / {len(no_respondibles)} abstencion)")
    print(f"[OK] data/entidades.json")


if __name__ == "__main__":
    main()
