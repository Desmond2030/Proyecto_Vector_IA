# ESTADO_ARCHIVO: EN_MODIFICACION
"""Lectura mediante las clases del proyecto; toda escritura queda dentro de api/."""
import base64
import csv
import json
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import pandas as pd
from src.Tabular_Data_Processor import TabularDataProcessor

ROOT = Path(__file__).resolve().parents[1]
API = ROOT / 'api'
UPLOADS = API / 'archivos'
TMP = API / '.tmp'
EXTENSIONES = {'modelo': {'.keras', '.h5'}, 'scaler': {'.pkl', '.joblib'}, 'datos': {'.csv', '.xlsx', '.parquet'}}


def serializar(value):
    """Convierte resultados tabulares y NumPy a JSON, conservando nulos."""
    if isinstance(value, pd.DataFrame):
        return json.loads(value.to_json(orient='records', date_format='iso'))
    if isinstance(value, dict):
        return {str(k): serializar(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializar(v) for v in value]
    if hasattr(value, 'tolist'):
        return serializar(value.tolist())
    if isinstance(value, float) and not __import__('math').isfinite(value):
        return None
    return value


def tabla(frame, **extra):
    return {'columnas': list(frame.columns), 'filas': serializar(frame), **extra}


def ruta_permitida(nombre, tipo='modelo'):
    roots = [ROOT / 'data'] if tipo == 'datos' else [ROOT / 'models', UPLOADS]
    path = (ROOT / nombre).resolve()
    if not any(path.is_relative_to(r.resolve()) for r in roots):
        raise ValueError('La ruta debe pertenecer al inventario del proyecto.')
    if not path.is_file() or path.suffix.lower() not in EXTENSIONES[tipo]:
        raise ValueError(f'No existe un archivo compatible: {nombre}')
    return path


def inventario():
    result = {k: [] for k in EXTENSIONES}
    for kind, suffixes in EXTENSIONES.items():
        roots = [ROOT / 'data'] if kind == 'datos' else [ROOT / 'models', UPLOADS]
        for root in roots:
            if root.exists():
                result[kind].extend(str(p.relative_to(ROOT)) for p in root.rglob('*')
                                    if p.is_file() and p.suffix.lower() in suffixes and p.resolve().is_relative_to(root.resolve()))
        result[kind].sort()
    return result


def cargar_archivo(path, tipo):
    path = Path(path)
    processor = TabularDataProcessor()
    if tipo == 'scaler':
        if path.suffix.lower() == '.pkl':
            processor.save_load_scaler(str(path), load=True)
        else:
            processor.scaler = joblib.load(path)
        return processor.scaler
    if tipo != 'datos':
        raise ValueError('El cargador tabular recibe datos o scaler.')
    if path.suffix.lower() == '.csv':
        with path.open(encoding='utf-8-sig', newline='') as stream:
            columns = next(csv.reader(stream), [])
        if not columns or len(set(columns)) != len(columns):
            raise ValueError('El CSV necesita encabezados únicos.')
    processor.load_data(str(path))
    if processor.data.empty or len(processor.data) > 10000 or len(processor.data.columns) > 200:
        raise ValueError('Admite de 1 a 10.000 filas y hasta 200 columnas.')
    return processor


@contextmanager
def imagenes_temporales(imagenes):
    TMP.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=TMP) as folder:
        paths = {}
        total = 0
        for image in imagenes:
            name = image.nombre
            if Path(name).name != name or '\\' in name or Path(name).suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
                raise ValueError('Cada imagen debe tener un nombre simple JPG o PNG.')
            if name in paths:
                raise ValueError(f'Nombre de imagen ambiguo: {name}')
            content = base64.b64decode(image.contenido, validate=True)
            total += len(content)
            if len(content) > 20 * 1024**2 or total > 100 * 1024**2:
                raise ValueError('Máximo 20 MB por imagen y 100 MB por lote.')
            path = Path(folder) / name
            path.write_bytes(content)
            paths[name] = str(path)
        yield paths


def guardar_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(serializar(data), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)
