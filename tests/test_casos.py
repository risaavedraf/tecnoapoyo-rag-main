"""Contratos del repositorio de casos: validación, persistencia e ids.

Sin red y sin base compartida: cada test usa su propio archivo SQLite en
tmp_path, y "reiniciar el proceso" equivale a construir una instancia nueva
sobre la misma ruta.
"""
import json

import pytest

from src import config
from src.casos import (
    ESTADOS_VALIDOS,
    PRIORIDADES_VALIDAS,
    ErrorCaso,
    RepositorioCasos,
    clave_de_cliente,
)


@pytest.fixture
def repo(tmp_path):
    return RepositorioCasos(ruta=tmp_path / "casos.db")


def test_siembra_los_diez_casos_versionados(repo):
    assert repo.contar() == 10
    primero = repo.obtener_caso("CASO-0001")
    assert primero["cliente"] == "María Fernanda Rojas"
    assert primero["estado"] == "cerrado"
    assert primero["procedimiento"] == "PROC-001"
    assert primero["notas"] == [
        "Recepción registrada con firma del cliente.",
        "Equipo entregado el 2026-03-06.",
    ]
    # La semilla versionada es la fuente: cada caso referencia un PROC real.
    semilla = json.loads(config.RUTA_CASOS_SEMILLA.read_text(encoding="utf-8"))
    assert len(semilla["casos"]) == 10
    for caso in semilla["casos"]:
        assert caso["estado"] in ESTADOS_VALIDOS
        assert caso["prioridad"] in PRIORIDADES_VALIDAS
        assert (config.DIR_PROCEDIMIENTOS / f"{caso['procedimiento']}.md").exists()


def test_sembrar_dos_veces_no_duplica(tmp_path):
    ruta = tmp_path / "casos.db"
    RepositorioCasos(ruta=ruta)
    segundo = RepositorioCasos(ruta=ruta)
    assert segundo.contar() == 10
    assert segundo.sembrados == 0


def test_crear_caso_genera_id_secuencial_y_sobrevive_al_reinicio(tmp_path):
    ruta = tmp_path / "casos.db"
    repo = RepositorioCasos(ruta=ruta)
    creado = repo.crear_caso(
        cliente="Cliente Nuevo",
        categoria="recepcion",
        descripcion="Equipo con pantalla quebrada.",
        prioridad="alta",
        fecha_creacion="2026-04-01",
    )
    assert creado["id"] == "CASO-0011"
    assert creado["estado"] == "abierto"
    assert creado["notas"] == []

    # Reinicio de proceso: instancia nueva sobre el mismo archivo.
    reabierto = RepositorioCasos(ruta=ruta)
    recuperado = reabierto.obtener_caso("CASO-0011")
    assert recuperado["cliente"] == "Cliente Nuevo"
    assert recuperado["prioridad"] == "alta"
    assert reabierto.contar() == 11


def test_estado_invalido_se_rechaza_y_no_modifica_el_caso(repo):
    antes = repo.obtener_caso("CASO-0004")
    with pytest.raises(ErrorCaso) as excinfo:
        repo.actualizar_estado("CASO-0004", "en_limbo")
    assert excinfo.value.codigo == "estado_invalido"
    assert "abierto, en_progreso, escalado, cerrado" in str(excinfo.value)
    # El efecto observable que importa: el estado persistido no cambió.
    despues = repo.obtener_caso("CASO-0004")
    assert despues == antes
    assert despues["estado"] == "abierto"


def test_actualizar_estado_valido_persiste(repo):
    actualizado = repo.actualizar_estado("CASO-0004", "en_progreso")
    assert actualizado["estado"] == "en_progreso"
    assert repo.obtener_caso("CASO-0004")["estado"] == "en_progreso"


def test_escalar_marca_estado_y_deja_el_motivo_como_nota(repo):
    escalado = repo.escalar("CASO-0004", "Requiere autorización de descuento")
    assert escalado["estado"] == "escalado"
    assert "Escalado: Requiere autorización de descuento" in escalado["notas"]


def test_agregar_nota_persiste_tras_reinicio(tmp_path):
    ruta = tmp_path / "casos.db"
    repo = RepositorioCasos(ruta=ruta)
    repo.agregar_nota("CASO-0009", "Se respondió por correo.")
    assert repo.obtener_caso("CASO-0009")["notas"] == ["Se respondió por correo."]
    assert RepositorioCasos(ruta=ruta).obtener_caso("CASO-0009")["notas"] == [
        "Se respondió por correo."
    ]


def test_caso_inexistente_devuelve_codigo_no_encontrado(repo):
    with pytest.raises(ErrorCaso) as excinfo:
        repo.obtener_caso("CASO-9999")
    assert excinfo.value.codigo == "no_encontrado"
    with pytest.raises(ErrorCaso) as excinfo:
        repo.agregar_nota("CASO-9999", "nota huérfana")
    assert excinfo.value.codigo == "no_encontrado"


def test_listar_filtra_por_estado_y_rechaza_estado_invalido(repo):
    abiertos = repo.listar_casos(estado="abierto")
    # Del más reciente al más antiguo, igual que el listado sin filtro.
    assert [caso["id"] for caso in abiertos] == ["CASO-0009", "CASO-0007", "CASO-0004"]
    assert all(caso["estado"] == "abierto" for caso in abiertos)
    assert len(repo.listar_casos()) == 10
    assert len(repo.listar_casos(limite=2)) == 2
    with pytest.raises(ErrorCaso) as excinfo:
        repo.listar_casos(estado="pendiente")
    assert excinfo.value.codigo == "estado_invalido"


# --- orden, total y filtro por cliente -----------------------------------
def test_listar_casos_devuelve_el_mas_reciente_primero(repo):
    ids = [caso["id"] for caso in repo.listar_casos()]
    assert ids[0] == "CASO-0010"
    assert ids == sorted(ids, reverse=True)


def test_un_caso_recien_creado_aparece_en_la_primera_pagina(repo):
    creado = repo.crear_caso(
        cliente="Cliente Once",
        categoria="recepcion",
        descripcion="Equipo con pantalla quebrada.",
    )
    assert creado["id"] == "CASO-0011"
    # El default de la herramienta es 10: con la semilla completa, el caso más
    # nuevo no puede quedar fuera de la página.
    ids = [caso["id"] for caso in repo.listar_casos(limite=10)]
    assert creado["id"] in ids
    assert ids[0] == creado["id"]


def test_contar_respeta_los_mismos_filtros_que_el_listado(repo):
    assert repo.contar() == 10
    assert repo.contar() == len(repo.listar_casos(limite=100))
    assert repo.contar(estado="abierto") == 3
    assert repo.contar(estado="abierto") == len(
        repo.listar_casos(estado="abierto", limite=100)
    )
    assert repo.contar(cliente="Valentina Soto") == 1
    repo.crear_caso(
        cliente="Valentina Soto",
        categoria="reparacion",
        descripcion="Teclado con teclas pegadas.",
    )
    assert repo.contar(cliente="Valentina Soto") == 2
    # Los dos filtros se combinan: el conteo no es el del listado sin filtros.
    assert repo.contar(estado="abierto", cliente="Valentina Soto") == 2
    assert repo.contar(estado="cerrado", cliente="Valentina Soto") == 0
    assert repo.contar(estado="abierto") == 4


def test_el_filtro_de_cliente_ignora_mayusculas(repo):
    casos = repo.listar_casos(cliente="VALENTINA SOTO")
    assert [caso["id"] for caso in casos] == ["CASO-0007"]
    assert casos[0]["cliente"] == "Valentina Soto"
    assert repo.contar(cliente="  valentina soto  ") == 1


def test_la_clave_de_cliente_quita_tildes_mayusculas_y_espacios_de_mas():
    """S1: la igualdad de nombres se decide con una sola clave.

    `MARIA FERNANDA ROJAS` y `  María   Fernanda Rojas ` son la misma clave.
    No se toca lo almacenado: la clave solo sirve para comparar.
    """
    assert clave_de_cliente("  MARÍA   Fernanda Rojas ") == "maria fernanda rojas"
    assert clave_de_cliente("Jorge Peña") == clave_de_cliente("jorge pena")
    assert clave_de_cliente("O'Higgins & Cía.") == clave_de_cliente(
        "o'higgins & cia."
    )
    # Un nombre distinto no comparte clave con otro: sin tilde no hay colisión.
    assert clave_de_cliente("María Fernanda Rojas") != clave_de_cliente(
        "María Fernanda Rojas Silva"
    )


def test_el_filtro_de_cliente_ignora_las_tildes(repo):
    """S1: quien escribe `MARIA FERNANDA ROJAS` encuentra `María Fernanda Rojas`."""
    casos = repo.listar_casos(cliente="MARIA FERNANDA ROJAS")
    assert [caso["id"] for caso in casos] == ["CASO-0001"]
    # El nombre devuelto es el canónico de la base, no el texto de la consulta.
    assert casos[0]["cliente"] == "María Fernanda Rojas"
    assert repo.contar(cliente="MARIA FERNANDA ROJAS") == 1
    # Todas las grafías del mismo nombre dan el mismo resultado.
    for consulta in (
        "maria fernanda rojas",
        "mArIa fErNaNdA rOjAs",
        "  María   Fernanda   Rojas  ",
        "MARÍA FERNANDA ROJAS",
    ):
        assert repo.contar(cliente=consulta) == 1, consulta
        assert len(repo.listar_casos(cliente=consulta, limite=100)) == 1, consulta


def test_contar_y_listar_coinciden_con_tildes(repo):
    """S1: `total` (de `contar`) y la página (de `listar_casos`) no divergen."""
    repo.crear_caso(
        cliente="María Fernanda Rojas",
        categoria="recepcion",
        descripcion="Segundo equipo de la misma clienta.",
    )
    for consulta in ("MARIA FERNANDA ROJAS", "maría fernanda rojas"):
        listados = repo.listar_casos(cliente=consulta, limite=100)
        assert repo.contar(cliente=consulta) == len(listados) == 2
        assert {caso["cliente"] for caso in listados} == {"María Fernanda Rojas"}


def test_dos_clientes_que_difieren_en_algo_mas_que_tildes_siguen_separados(repo):
    """S4: normalizar tildes no puede fusionar a dos clientes distintos."""
    repo.crear_caso(
        cliente="María Fernanda Rojas Silva",
        categoria="recepcion",
        descripcion="Equipo de otra persona con nombre parecido.",
    )
    propios = repo.listar_casos(cliente="MARIA FERNANDA ROJAS", limite=100)
    assert [caso["cliente"] for caso in propios] == ["María Fernanda Rojas"]
    assert repo.contar(cliente="MARIA FERNANDA ROJAS") == 1
    assert repo.contar(cliente="MARIA FERNANDA ROJAS SILVA") == 1
    # Un nombre que no existe no se inventa por parecido: la comparación es de
    # igualdad, no de prefijo ni de subcadena.
    assert repo.contar(cliente="MARIA FERNANDA ROJAS PEREZ") == 0
    assert repo.contar(cliente="MARIA FERNANDA") == 0
    assert repo.listar_casos(cliente="MARIA FERNANDA ROJAS PEREZ", limite=100) == []


def test_el_filtro_de_cliente_no_interpola_texto_en_el_sql(repo):
    cliente_comilla = "O'Higgins & Cía."
    repo.crear_caso(
        cliente=cliente_comilla,
        categoria="recepcion",
        descripcion="Impresora sin conexión.",
    )
    propios = repo.listar_casos(cliente=cliente_comilla)
    assert [caso["cliente"] for caso in propios] == [cliente_comilla]
    assert repo.contar(cliente=cliente_comilla) == 1
    # La inyección clásica se trata como nombre literal: no devuelve todo ni
    # rompe la sentencia.
    assert repo.contar(cliente="' OR 1=1 --") == 0
    assert repo.listar_casos(cliente="' OR 1=1 --") == []
    assert repo.contar() == 11


@pytest.mark.parametrize("campo", ["cliente", "categoria", "descripcion"])
def test_campos_obligatorios_vacios_se_rechazan(repo, campo):
    argumentos = {
        "cliente": "Cliente Válido",
        "categoria": "recepcion",
        "descripcion": "Descripción válida.",
    }
    argumentos[campo] = "   "
    antes = repo.contar()
    with pytest.raises(ErrorCaso) as excinfo:
        repo.crear_caso(**argumentos)
    assert excinfo.value.codigo == "validacion"
    assert repo.contar() == antes


def test_prioridad_invalida_se_rechaza(repo):
    antes = repo.contar()
    with pytest.raises(ErrorCaso) as excinfo:
        repo.crear_caso(
            cliente="Cliente", categoria="recepcion",
            descripcion="Algo.", prioridad="urgentísima",
        )
    assert excinfo.value.codigo == "prioridad_invalida"
    assert repo.contar() == antes


def test_semilla_con_estado_invalido_aborta_sin_sembrar(tmp_path):
    semilla = tmp_path / "semilla_mala.json"
    semilla.write_text(json.dumps({
        "casos": [{
            "id": "CASO-0001", "cliente": "X", "categoria": "recepcion",
            "descripcion": "Y", "estado": "en_limbo", "prioridad": "media",
            "fecha_creacion": "2026-04-01",
        }]
    }), encoding="utf-8")
    ruta = tmp_path / "casos.db"
    # La semilla es código versionado: un estado fuera de la allowlist es un
    # defecto que debe detenerse, no un caso a medio cargar.
    with pytest.raises(ErrorCaso) as excinfo:
        RepositorioCasos(ruta=ruta, semilla=semilla)
    assert excinfo.value.codigo == "estado_invalido"
    # Nada quedó persistido: la carga abortó antes del primer INSERT.
    assert RepositorioCasos(ruta=ruta, semilla=tmp_path / "ausente.json").contar() == 0
