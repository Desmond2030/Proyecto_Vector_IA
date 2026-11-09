# ESTADO_ARCHIVO: EN_MODIFICACION
"""API local de VectorIA. Ejecutar con un worker: uvicorn api.main:app."""
import sys
sys.dont_write_bytecode = True
import json
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import RLock
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from api import evaluation
from api.datasets import generar_correlaciones, preparar
from api.loaders import API, TMP, UPLOADS, EXTENSIONES, cargar_archivo, guardar_json, inventario, ruta_permitida, serializar, tabla
from api.predict import predecir
from api.schemas import Configuracion, Preparacion, Analisis, Inferencia, Evaluacion, Importacion, SeleccionEvaluacion

CONFIG = API / 'configuracion.json'
lock = RLock()
models, errores_carga = {}, {}
configuracion = Configuracion()
revision = uuid4().hex


@asynccontextmanager
async def lifespan(app):
    global configuracion
    if CONFIG.exists():
        try:
            configuracion = Configuracion.model_validate_json(CONFIG.read_text(encoding='utf-8'))
            from api.model_adapter import ModeloInferencia
            for slot in ('cnn', 'rnn', 'ann'):
                cfg = getattr(configuracion, slot)
                if cfg.archivo:
                    try:
                        models[slot] = ModeloInferencia(slot, cfg)
                    except Exception as exc:
                        errores_carga[slot] = str(exc)
        except Exception as exc:
            errores_carga['configuracion'] = str(exc)
    yield


app = FastAPI(title='VectorIA', version='1.0.0', lifespan=lifespan)


@app.exception_handler(ValueError)
@app.exception_handler(KeyError)
@app.exception_handler(TypeError)
@app.exception_handler(OSError)
async def entrada_invalida(request, exc):
    return JSONResponse(status_code=422, content={'detail': str(exc)})


def validar_revision(value):
    if value is not None and value != revision:
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
    return {'configuracion': configuracion.model_dump(), 'revision': revision,
            'identidades': {slot: model.identidad for slot, model in models.items()}}


@app.put('/configuracion')
def configurar(request: Configuracion):
    global configuracion, revision
    from api.model_adapter import ModeloInferencia
    with lock:
        candidates = {}
        for slot in ('cnn', 'rnn', 'ann'):
            cfg = getattr(request, slot)
            if cfg.archivo:
                try:
                    candidates[slot] = ModeloInferencia(slot, cfg)
                except Exception as exc:
                    raise HTTPException(422, f'{slot.upper()}: {exc}') from exc
        guardar_json(CONFIG, request.model_dump())
        configuracion = request
        models.clear()
        models.update(candidates)
        errores_carga.clear()
        revision = uuid4().hex
        return estado()


@app.get('/datos')
def datos(nombre: str | None = None):
    if nombre is None:
        return inventario()['datos']
    return tabla(cargar_archivo(ruta_permitida(nombre, 'datos'), 'datos').data, fuente=nombre)


@app.post('/files/{tipo}', status_code=201)
async def subir(tipo: str, nombre: str, request: Request):
    if tipo not in EXTENSIONES or Path(nombre).name != nombre or '\\' in nombre or ':' in nombre or Path(nombre).suffix.lower() not in EXTENSIONES[tipo]:
        raise HTTPException(422, 'Tipo, nombre o extensión no admitidos.')
    TMP.mkdir(exist_ok=True)
    limit = (20 if tipo == 'datos' else 500) * 1024**2
    with TemporaryDirectory(dir=TMP) as directory:
        path = Path(directory) / nombre
        size = 0
        with path.open('wb') as stream:
            async for chunk in request.stream():
                size += len(chunk)
                if size > limit:
                    raise HTTPException(413, 'El archivo supera el límite de carga.')
                stream.write(chunk)
        if not size:
            raise HTTPException(422, 'Archivo vacío.')
        if tipo == 'datos':
            return tabla(cargar_archivo(path, tipo).data, fuente=nombre)
        UPLOADS.mkdir(exist_ok=True)
        target = UPLOADS / nombre
        try:
            with target.open('xb') as dest, path.open('rb') as source:
                import shutil
                shutil.copyfileobj(source, dest)
        except FileExistsError as exc:
            raise HTTPException(409, 'Ya existe ese nombre. Selecciónalo o renombra el archivo.') from exc
        return {'archivo': str(target.relative_to(API.parent)), 'mensaje': 'Subido. Se validará al aplicar la configuración.'}


@app.post('/datasets/prepare')
def preparar_tabla(request: Preparacion):
    return preparar(request)


@app.post('/datasets/correlaciones')
def correlaciones(request: Analisis):
    return generar_correlaciones(request)


@app.post('/predict/{modo}')
def inferir(modo: str, request: Inferencia):
    if modo not in ('mosquito', 'casos', 'accion', 'flujo'):
        raise HTTPException(404, 'Modo desconocido.')
    with lock:
        validar_revision(request.revision)
        return serializar(predecir(request, models, modo))


@app.get('/evaluaciones')
def evaluaciones():
    with lock:
        data = evaluation.leer()
        data['vigentes'] = {slot: next((r for r in data['resultados']
                              if r['id'] == data['seleccion'].get(slot) and r['identidad'] == model.identidad), None)
                            for slot, model in models.items()}
        return data


@app.post('/evaluaciones/seleccionar')
def seleccionar(request: SeleccionEvaluacion):
    with lock:
        data = evaluation.leer()
        model = models.get(request.modelo)
        if model is None or not any(r['id'] == request.id and r['modelo'] == request.modelo and r['identidad'] == model.identidad for r in data['resultados']):
            raise ValueError('La evaluación debe pertenecer al artefacto activo.')
        data['seleccion'][request.modelo] = request.id
        guardar_json(evaluation.ARCHIVO, data)
        return {'seleccion': data['seleccion']}


@app.post('/evaluaciones/importar')
def importar_metricas(request: Importacion):
    with lock:
        validar_revision(request.revision)
        return evaluation.importar(request, models)


@app.post('/evaluate')
def evaluar(request: Evaluacion):
    with lock:
        validar_revision(request.revision)
        try:
            return evaluation.evaluar(request, models)
        except Exception as exc:
            raise HTTPException(422, f'No se pudo evaluar: {exc}') from exc
