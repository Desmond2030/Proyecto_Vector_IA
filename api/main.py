# ESTADO_ARCHIVO: EN_MODIFICACION
"""API local de VectorIA. Un worker: uvicorn api.main:app."""
import sys
sys.dont_write_bytecode = True
import base64
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import RLock
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import JSONResponse
from api import evaluation
from api.datasets import generar_correlaciones, preparar
from api.loaders import (API, TMP, UPLOADS, EXTENSIONES, datos_procesados, guardar_json,
                         inventario, ruta_permitida, serializar, tabla, muestras)
from api.predict import predecir
from api.schemas import Configuracion, Preparacion, Analisis, Inferencia, Evaluacion

CONFIG = API / 'configuracion.json'
lock = RLock()
models, errores_carga = {}, {}
configuracion = Configuracion()
revision = uuid4().hex


def cargar_configuracion(config):
    from api.model_adapter import ModeloInferencia
    candidates, errors = {}, {}
    for slot in ('cnn', 'rnn', 'ann'):
        cfg = getattr(config, slot)
        if not cfg.archivo:
            errors[slot] = 'Selecciona un modelo en Predicción.'
            continue
        try:
            candidates[slot] = ModeloInferencia(slot, cfg)
        except Exception as exc:
            errors[slot] = str(exc)
    return candidates, errors


@asynccontextmanager
async def lifespan(app):
    global configuracion
    if CONFIG.exists():
        try:
            configuracion = Configuracion.model_validate_json(CONFIG.read_text(encoding='utf-8'))
            candidates, errors = cargar_configuracion(configuracion)
            models.update(candidates)
            errores_carga.update(errors)
        except Exception as exc:
            errores_carga['configuracion'] = str(exc)
    yield


app = FastAPI(title='VectorIA', version='1.1.0', lifespan=lifespan)


@app.exception_handler(ValueError)
@app.exception_handler(KeyError)
@app.exception_handler(TypeError)
@app.exception_handler(OSError)
async def entrada_invalida(request, exc):
    return JSONResponse(status_code=422, content={'detail': str(exc)})


def validar_revision(value):
    if value != revision:
        raise HTTPException(409, 'Cambió la configuración. Actualiza la página y vuelve a enviar los datos.')


@app.get('/health')
def health():
    return {'servicio': 'vectoria', 'version': app.version, 'estado': 'ok',
            'modelos': {slot: slot in models for slot in ('cnn', 'rnn', 'ann')}, 'errores': errores_carga}


@app.get('/artefactos')
def artefactos():
    return inventario()


@app.get('/configuracion')
def estado():
    with lock:
        return {'configuracion': configuracion.model_dump(), 'revision': revision,
                'identidades': {slot: model.identidad for slot, model in models.items()}, 'errores': errores_carga.copy()}


@app.put('/configuracion')
def configurar(request: Configuracion):
    global configuracion, revision
    with lock:
        candidates, errors = cargar_configuracion(request)
        guardar_json(CONFIG, request.model_dump())
        configuracion = request
        models.clear()
        models.update(candidates)
        errores_carga.clear()
        errores_carga.update(errors)
        revision = uuid4().hex
        return estado()


@app.get('/datos')
def datos(nombre: str | None = None):
    if nombre is None:
        return inventario()['datos']
    return tabla(datos_procesados(nombre), fuente=nombre)


@app.get('/imagenes')
def imagenes(conjunto: str = 'val', especie: str | None = None,
             limite: int = Query(12, ge=1, le=100), pagina: int = Query(0, ge=0)):
    return muestras(conjunto, especie, limite, pagina)


@app.get('/imagen')
def imagen(nombre: str):
    path = ruta_permitida(nombre, 'imagen')
    if path.stat().st_size > 20 * 1024**2:
        raise ValueError('La imagen supera 20 MB.')
    return {'contenido': base64.b64encode(path.read_bytes()).decode(), 'nombre': path.name}


@app.post('/files/{tipo}', status_code=201)
async def subir(tipo: str, nombre: str, request: Request):
    if tipo not in ('modelo', 'scaler') or Path(nombre).name != nombre or chr(92) in nombre or ':' in nombre or Path(nombre).suffix.lower() not in EXTENSIONES[tipo]:
        raise HTTPException(422, 'Tipo, nombre o extensión no admitidos.')
    TMP.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=TMP) as directory:
        path = Path(directory) / nombre
        size = 0
        with path.open('wb') as stream:
            async for chunk in request.stream():
                size += len(chunk)
                if size > 500 * 1024**2:
                    raise HTTPException(413, 'El archivo supera 500 MB.')
                stream.write(chunk)
        if not size:
            raise HTTPException(422, 'Archivo vacío.')
        UPLOADS.mkdir(exist_ok=True)
        target = UPLOADS / nombre
        try:
            with target.open('xb') as dest, path.open('rb') as source:
                import shutil
                shutil.copyfileobj(source, dest)
        except FileExistsError as exc:
            raise HTTPException(409, 'Ya existe ese nombre. Selecciónalo o renombra el archivo.') from exc
    return {'archivo': str(target.relative_to(API.parent)), 'mensaje': 'Archivo disponible para seleccionar.'}


@app.post('/datasets/prepare')
def preparar_tabla(request: Preparacion):
    return preparar(request)


@app.post('/datasets/correlaciones')
def correlaciones(request: Analisis):
    return generar_correlaciones(request)


@app.post('/predict/flujo')
def inferir(request: Inferencia):
    with lock:
        validar_revision(request.revision)
        return serializar(predecir(request, models, errores_carga))


@app.get('/evaluaciones')
def evaluaciones():
    with lock:
        return {'resultados': evaluation.leer()['resultados'], 'vigentes': evaluation.vigentes(models)}


@app.post('/evaluate')
def evaluar(request: Evaluacion):
    with lock:
        validar_revision(request.revision)
        try:
            return evaluation.evaluar(request, models, errores_carga)
        except Exception as exc:
            raise HTTPException(422, f'No se pudo evaluar: {exc}') from exc
