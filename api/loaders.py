# ESTADO_ARCHIVO: EN_MODIFICACION
"""Lectura de originales; escrituras solo dentro de api/."""
import base64
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
PROCESADOS = ROOT / 'data/proceced/archivos_parquet'
IMAGENES = ROOT / 'data/raw/Imagenes'
EXTENSIONES = {'modelo': {'.keras', '.h5'}, 'scaler': {'', '.pkl', '.joblib'}, 'datos': {'.parquet'}}
EXT_IMAGEN = {'.jpg', '.jpeg', '.png'}


def serializar(value):
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
    # Los Parquet del proyecto conservan la fecha como índice.
    if frame.index.name and frame.index.name not in frame.columns:
        frame = frame.reset_index()
    return {'columnas': list(frame.columns), 'filas': serializar(frame), **extra}


def ruta_permitida(nombre, tipo='modelo'):
    roots = [PROCESADOS] if tipo == 'datos' else [ROOT / 'models', UPLOADS]
    if tipo == 'imagen':
        roots = [IMAGENES]
    path = (ROOT / nombre).resolve()
    suffixes = EXT_IMAGEN if tipo == 'imagen' else EXTENSIONES[tipo]
    if not any(path.is_relative_to(r.resolve()) for r in roots):
        raise ValueError('La ruta debe pertenecer al inventario del proyecto.')
    if not path.is_file() or path.suffix.lower() not in suffixes:
        raise ValueError(f'No existe un archivo compatible: {nombre}')
    if tipo == 'scaler' and path == (ROOT / 'models/scaler.pkl').resolve():
        raise ValueError('Ese escalador pertenece al proyecto anterior.')
    return path


def inventario():
    result = {k: [] for k in EXTENSIONES}
    for kind, suffixes in EXTENSIONES.items():
        roots = [PROCESADOS] if kind == 'datos' else [ROOT / 'models', UPLOADS]
        for root in roots:
            if root.exists():
                for p in root.rglob('*'):
                    if p.is_file() and p.suffix.lower() in suffixes:
                        try:
                            ruta_permitida(str(p.relative_to(ROOT)), kind)
                            if kind == 'scaler' and p.name.startswith('.'):
                                continue
                            result[kind].append(str(p.relative_to(ROOT)))
                        except ValueError:
                            continue
        result[kind].sort()
    return result


def cargar_archivo(path, tipo):
    path = Path(path)
    processor = TabularDataProcessor()
    if tipo == 'scaler':
        if path.suffix.lower() == '.joblib':
            processor.scaler = joblib.load(path)
        else:
            processor.save_load_scaler(str(path), load=True)
        return processor.scaler
    processor.load_data(str(path))
    if processor.data.empty:
        raise ValueError('El dataset está vacío.')
    return processor


def datos_procesados(nombre):
    frame = cargar_archivo(ruta_permitida(nombre, 'datos'), 'datos').data
    return frame.reset_index() if frame.index.name and frame.index.name not in frame.columns else frame.copy()


def muestras(conjunto='val', especie=None, limite=12, pagina=0):
    if conjunto not in ('train', 'val', 'test'):
        raise ValueError('Conjunto de imágenes desconocido.')
    folder = IMAGENES / conjunto
    clases = sorted(p.name for p in folder.iterdir() if p.is_dir())
    if especie and especie not in clases:
        raise ValueError('Especie desconocida.')
    paths = []
    counts = {}
    for label in clases:
        found = sorted(p for p in (folder / label).iterdir() if p.is_file() and p.suffix.lower() in EXT_IMAGEN)
        counts[label] = len(found)
        if not especie or especie == label:
            paths.extend(found)
    selected = paths[pagina * limite:(pagina + 1) * limite]
    return {'clases': clases, 'conteos': counts, 'total': len(paths),
            'imagenes': [{'ruta': str(p.relative_to(ROOT)), 'nombre': p.name, 'especie': p.parent.name}
                         for p in selected]}


@contextmanager
def imagen_temporal(imagen):
    if imagen.ruta:
        yield ruta_permitida(imagen.ruta, 'imagen')
        return
    name = imagen.nombre
    if Path(name).name != name or chr(92) in name or Path(name).suffix.lower() not in EXT_IMAGEN:
        raise ValueError('Selecciona una imagen JPG o PNG con nombre simple.')
    content = base64.b64decode(imagen.contenido, validate=True)
    if not content or len(content) > 20 * 1024**2:
        raise ValueError('La imagen debe contener entre 1 byte y 20 MB.')
    TMP.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=TMP) as folder:
        path = Path(folder) / name
        path.write_bytes(content)
        yield path


def guardar_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(serializar(data), ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)
