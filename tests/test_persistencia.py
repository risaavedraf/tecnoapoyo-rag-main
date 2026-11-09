"""Infraestructura SQLite compartida (S15): generador de ids y contrato de error."""
import pytest

from src import persistencia
from src.casos import ErrorCaso
from src.memoria import ErrorMemoria


@pytest.fixture
def abierta(tmp_path):
    with persistencia.conexion(tmp_path / "filas.db") as conexion:
        conexion.execute("CREATE TABLE filas (id TEXT PRIMARY KEY)")
        yield conexion


def test_siguiente_id_es_secuencial_y_con_formato_de_cuatro_digitos(abierta):
    assert persistencia.siguiente_id(abierta, tabla="filas", prefijo="F") == "F-0001"
    abierta.execute("INSERT INTO filas (id) VALUES ('F-0001')")
    assert persistencia.siguiente_id(abierta, tabla="filas", prefijo="F") == "F-0002"


@pytest.mark.parametrize("tabla,columna", [
    ("filas; DROP TABLE filas--", "id"),
    ("filas", "id) --"),
    ("filas ", "id"),
    ("", "id"),
    ("filas' OR '1'='1", "id"),
    ("filas", "id; DROP TABLE filas"),
])
def test_siguiente_id_rechaza_identificadores_no_simples(abierta, tabla, columna):
    with pytest.raises(ValueError):
        persistencia.siguiente_id(abierta, tabla=tabla, prefijo="F", columna=columna)
    assert abierta.execute("SELECT COUNT(*) FROM filas").fetchone()[0] == 0


def test_siguiente_id_continua_desde_el_conteo_si_el_ultimo_id_es_ajeno(abierta):
    abierta.execute("INSERT INTO filas (id) VALUES ('F-0001')")
    abierta.execute("INSERT INTO filas (id) VALUES ('ZZZ-AJENO')")
    assert persistencia.siguiente_id(abierta, tabla="filas", prefijo="F") == "F-0003"


def test_el_prefijo_no_entra_en_la_sentencia(abierta):
    assert persistencia.siguiente_id(
        abierta, tabla="filas", prefijo="X' OR 1=1 --") == "X' OR 1=1 ---0001"
    assert abierta.execute("SELECT COUNT(*) FROM filas").fetchone()[0] == 0


def test_texto_obligatorio_lanza_el_error_del_dominio():
    with pytest.raises(ErrorCaso) as caso:
        persistencia.texto_obligatorio("   ", "cliente", ErrorCaso)
    assert str(caso.value) == "El campo 'cliente' no puede venir vacío."
    assert caso.value.codigo == "validacion"
    with pytest.raises(ErrorMemoria) as memoria:
        persistencia.texto_obligatorio("   ", "clave", ErrorMemoria)
    assert str(memoria.value) == "El campo 'clave' no puede venir vacío."
    assert memoria.value.codigo == "validacion"
    assert issubclass(ErrorCaso, persistencia.ErrorPersistencia)
    assert issubclass(ErrorMemoria, persistencia.ErrorPersistencia)
    assert ErrorCaso is not ErrorMemoria
