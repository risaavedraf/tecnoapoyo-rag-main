"""Memoria del agente (RA2 / IL2.2): corto y largo plazo, sin red.

La memoria de corto plazo se prueba con un resumidor falso inyectado: la
estrategia `resumen` no debe depender del proveedor para poder verificarse.

La memoria de largo plazo se prueba sobre archivos SQLite temporales. El
"reinicio de proceso" es una instancia nueva sobre la misma ruta, igual que en
la suite de casos.
"""
import pytest

from src.memoria import (
    ESTRATEGIAS_CORTO_PLAZO,
    Memoria,
    MemoriaCortoPlazo,
    MemoriaLargoPlazo,
)


def resumidor_falso(resumen_previo, turnos):
    """Resumidor inyectado: conserva los hechos, no comprime a la nada."""
    lineas = [t["contenido"] for t in turnos]
    base = f"{resumen_previo} " if resumen_previo else ""
    return (base + " | ".join(lineas)).strip()


# --- corto plazo ---------------------------------------------------------
def test_estrategias_disponibles():
    assert ESTRATEGIAS_CORTO_PLAZO == ("buffer", "ventana", "resumen")


def test_buffer_conserva_todos_los_turnos():
    memoria = MemoriaCortoPlazo(estrategia="buffer", ventana=2)
    for i in range(5):
        memoria.agregar("usuario", f"pregunta {i}")
    assert [t["contenido"] for t in memoria.turnos()] == [
        "pregunta 0", "pregunta 1", "pregunta 2", "pregunta 3", "pregunta 4",
    ]
    assert "pregunta 0" in memoria.texto()


def test_ventana_olvida_el_turno_viejo():
    memoria = MemoriaCortoPlazo(estrategia="ventana", ventana=2)
    for i in range(6):
        memoria.agregar("usuario", f"pregunta {i}")
    contenidos = [t["contenido"] for t in memoria.turnos()]
    assert contenidos == ["pregunta 4", "pregunta 5"]
    # El turno 0 (N-5) ya no aparece en el prompt.
    assert "pregunta 0" not in memoria.texto()


def test_resumen_conserva_los_hechos_del_turno_olvidado():
    memoria = MemoriaCortoPlazo(estrategia="resumen", ventana=1,
                                resumidor=resumidor_falso)
    memoria.agregar("usuario", "mi correo es ana@example.com")
    memoria.agregar("usuario", "vivo en Ñuñoa")
    memoria.agregar("usuario", "tengo un notebook Lenovo")
    # El turno viejo salió de la ventana pero sigue en el resumen.
    assert [t["contenido"] for t in memoria.turnos()] == ["tengo un notebook Lenovo"]
    assert "ana@example.com" in memoria.resumen
    assert "ana@example.com" in memoria.texto()


def test_resumen_sin_resumidor_no_toca_la_red_y_conserva_texto():
    memoria = MemoriaCortoPlazo(estrategia="resumen", ventana=1)
    memoria.agregar("usuario", "dato importante")
    memoria.agregar("usuario", "otro dato")
    assert "dato importante" in memoria.resumen or "dato importante" in memoria.texto()


def test_limpiar_vacia_el_corto_plazo():
    memoria = MemoriaCortoPlazo(estrategia="buffer")
    memoria.agregar("usuario", "algo")
    memoria.limpiar()
    assert memoria.turnos() == []
    assert memoria.texto() == ""


# --- largo plazo: hechos ------------------------------------------------
@pytest.fixture
def largo(tmp_path):
    return MemoriaLargoPlazo(ruta=tmp_path / "memoria.db", usuario="ana")


def test_un_hecho_sobrevive_al_reinicio(tmp_path):
    ruta = tmp_path / "memoria.db"
    MemoriaLargoPlazo(ruta=ruta, usuario="ana").guardar_hecho(
        "correo", "ana@example.com")
    reabierta = MemoriaLargoPlazo(ruta=ruta, usuario="ana")
    hechos = {h["clave"]: h["valor"] for h in reabierta.listar_hechos()}
    assert hechos == {"correo": "ana@example.com"}


def test_upsert_por_usuario_y_clave_no_duplica(largo):
    largo.guardar_hecho("correo", "viejo@example.com")
    largo.guardar_hecho("correo", "nuevo@example.com")
    hechos = largo.listar_hechos()
    assert len(hechos) == 1
    assert hechos[0]["valor"] == "nuevo@example.com"


def test_aislamiento_entre_usuarios(tmp_path):
    ruta = tmp_path / "memoria.db"
    ana = MemoriaLargoPlazo(ruta=ruta, usuario="ana")
    beto = MemoriaLargoPlazo(ruta=ruta, usuario="beto")
    ana.guardar_hecho("correo", "ana@example.com")
    assert [h["clave"] for h in beto.listar_hechos()] == []
    beto.guardar_hecho("correo", "beto@example.com")
    assert beto.listar_hechos()[0]["valor"] == "beto@example.com"
    assert ana.listar_hechos()[0]["valor"] == "ana@example.com"


def test_borrar_hecho(largo):
    largo.guardar_hecho("correo", "ana@example.com")
    assert largo.borrar_hecho("correo") is True
    assert largo.listar_hechos() == []
    # Borrar algo inexistente no miente sobre el efecto.
    assert largo.borrar_hecho("correo") is False


def test_hechos_relevantes_prioriza_coincidencias_y_respeta_el_tope(largo):
    largo.guardar_hecho("ciudad", "vive en Ñuñoa")
    largo.guardar_hecho("equipo", "notebook Lenovo IdeaPad")
    relevantes = largo.hechos_relevantes("¿cuál es mi notebook Lenovo?")
    assert relevantes
    assert relevantes[0]["clave"] == "equipo"
    acotados = largo.hechos_relevantes("notebook", limite_caracteres=12)
    assert sum(len(h["valor"]) for h in acotados) <= 12 or len(acotados) == 1


def test_hechos_relevantes_sin_coincidencias_cae_a_recencia(largo):
    largo.guardar_hecho("a", "primero")
    largo.guardar_hecho("b", "segundo")
    assert [h["clave"] for h in largo.hechos_relevantes("consulta sin relación")]


# --- largo plazo: sesiones ---------------------------------------------
def test_sesion_guarda_y_recarga_turnos(tmp_path):
    ruta = tmp_path / "memoria.db"
    largo = MemoriaLargoPlazo(ruta=ruta, usuario="ana")
    id_sesion = largo.crear_sesion()
    largo.guardar_turno(id_sesion, "usuario", "¿cuánto demora el diagnóstico?")
    largo.guardar_turno(id_sesion, "asistente", "Hasta 48 horas hábiles.")

    reabierta = MemoriaLargoPlazo(ruta=ruta, usuario="ana")
    turnos = reabierta.cargar_turnos(id_sesion)
    assert [t["rol"] for t in turnos] == ["usuario", "asistente"]
    assert turnos[0]["contenido"] == "¿cuánto demora el diagnóstico?"


def test_listar_sesiones_y_borrarlas(largo):
    primera = largo.crear_sesion()
    segunda = largo.crear_sesion()
    ids = [s["id"] for s in largo.listar_sesiones()]
    assert ids == [segunda, primera] or ids == [primera, segunda]
    assert largo.borrar_sesion(primera) is True
    assert primera not in [s["id"] for s in largo.listar_sesiones()]
    assert largo.cargar_turnos(primera) == []


def test_sesion_inexistente_no_inventa_turnos(largo):
    assert largo.cargar_turnos("SES-9999") == []


# --- fachada ------------------------------------------------------------
def test_memoria_sin_hechos_no_inyecta_bloque(tmp_path):
    memoria = Memoria(
        corto=MemoriaCortoPlazo(estrategia="buffer"),
        largo=MemoriaLargoPlazo(ruta=tmp_path / "memoria.db", usuario="ana"),
    )
    assert memoria.bloque_para("¿cuánto demora?") == ""


def test_memoria_inyecta_solo_hechos_relevantes(tmp_path):
    memoria = Memoria(
        corto=MemoriaCortoPlazo(estrategia="buffer"),
        largo=MemoriaLargoPlazo(ruta=tmp_path / "memoria.db", usuario="ana"),
    )
    memoria.recordar("equipo", "notebook Lenovo IdeaPad")
    memoria.recordar("mascota", "un gato llamado Tilo")
    bloque = memoria.bloque_para("¿qué notebook tengo?")
    assert "Memoria relevante del usuario" in bloque
    assert "Lenovo" in bloque
    assert "Tilo" not in bloque


def test_memoria_registra_turnos_en_la_sesion_abierta(tmp_path):
    memoria = Memoria(
        corto=MemoriaCortoPlazo(estrategia="buffer"),
        largo=MemoriaLargoPlazo(ruta=tmp_path / "memoria.db", usuario="ana"),
    )
    id_sesion = memoria.abrir_sesion()
    memoria.registrar("usuario", "hola")
    memoria.registrar("asistente", "buenas")
    assert [t["contenido"] for t in memoria.corto.turnos()] == ["hola", "buenas"]
    assert len(memoria.largo.cargar_turnos(id_sesion)) == 2


def test_memoria_recarga_una_sesion_en_el_corto_plazo(tmp_path):
    ruta = tmp_path / "memoria.db"
    memoria = Memoria(
        corto=MemoriaCortoPlazo(estrategia="buffer"),
        largo=MemoriaLargoPlazo(ruta=ruta, usuario="ana"),
    )
    id_sesion = memoria.abrir_sesion()
    memoria.registrar("usuario", "pregunta previa")

    otra = Memoria(
        corto=MemoriaCortoPlazo(estrategia="buffer"),
        largo=MemoriaLargoPlazo(ruta=ruta, usuario="ana"),
    )
    cantidad = otra.cargar_sesion(id_sesion)
    assert cantidad == 1
    assert [t["contenido"] for t in otra.corto.turnos()] == ["pregunta previa"]
    assert otra.sesion_id == id_sesion


def test_memoria_borra_hechos_desde_la_fachada(tmp_path):
    memoria = Memoria(
        corto=MemoriaCortoPlazo(estrategia="buffer"),
        largo=MemoriaLargoPlazo(ruta=tmp_path / "memoria.db", usuario="ana"),
    )
    memoria.recordar("correo", "ana@example.com")
    assert memoria.borrar_hecho("correo") is True
    assert memoria.listar_hechos() == []
