# ESTADO_ARCHIVO: EN_MODIFICACION
"""Inferencia común para un registro o un lote; no escribe datasets originales."""
import numpy as np
import pandas as pd
from api.datasets import matriz, ventana
from api.loaders import imagenes_temporales


def activo(models, slot):
    if slot not in models:
        raise ValueError(f'Modelo {slot.upper()} pendiente: selecciónalo en Modelos.')
    return models[slot]


def pronosticar(model, historial, registro):
    values, reference = ventana(historial, registro, model.config, model.scaler)
    prediction = model.ejecutar(values).reshape(1, model.config.horizonte)
    if model.scaler_salida is not None:
        scaler = model.scaler_salida
        if scaler.n_features_in_ == 1:
            prediction = scaler.inverse_transform(prediction.reshape(-1, 1)).reshape(1, -1)
        elif scaler.n_features_in_ == model.config.horizonte:
            prediction = scaler.inverse_transform(prediction)
        else:
            raise ValueError('El scaler de salida debe tener 1 variable o una por horizonte.')
    if not np.isfinite(prediction).all():
        raise ValueError('El pronóstico desescalado no es finito.')
    return {'semanas': [{'fecha': (reference + pd.Timedelta(weeks=i + 1)).date().isoformat(), 'casos': float(p)}
                        for i, p in enumerate(prediction[0])], 'total': float(prediction.sum())}


def recomendar(model, registro):
    values = matriz(pd.DataFrame([registro]), model.config, model.scaler)
    scores = model.ejecutar(values)[0]
    return {'accion': model.config.clases[int(np.argmax(scores))]}


def predecir(request, models, modo):
    if modo != 'flujo':
        activo(models, {'mosquito': 'cnn', 'casos': 'rnn', 'accion': 'ann'}[modo])
    records = request.registros
    if modo == 'mosquito' and not records:
        records = [{request.columna_imagen: image.nombre} for image in request.imagenes]
    if not records:
        raise ValueError('Agrega al menos una observación.')
    results = []
    with imagenes_temporales(request.imagenes) as images:
        detections = {}
        for position, original in enumerate(records, 1):
            row = dict(original)
            result = {'fila': position, 'id': row.get('id', position), 'entrada': original,
                      'errores': {}, 'artefactos': {}, 'cnn': None, 'rnn': None, 'ann': None}
            if modo in ('mosquito', 'flujo'):
                try:
                    model = activo(models, 'cnn')
                    name = row.get(request.columna_imagen)
                    if name not in images:
                        raise ValueError(f'No se cargó la imagen: {name or "sin nombre"}')
                    if name not in detections:
                        detections[name] = model.clasificar(images[name])
                    result['cnn'] = detections[name]
                    row['aegypti_detectado'] = result['cnn']['aegypti_detectado']
                    result['artefactos']['cnn'] = model.config.archivo
                except Exception as exc:
                    result['errores']['cnn'] = str(exc)
            if modo in ('casos', 'flujo'):
                try:
                    model = activo(models, 'rnn')
                    result['rnn'] = pronosticar(model, request.historial, row)
                    result['artefactos']['rnn'] = model.config.archivo
                    if modo == 'flujo' and model.config.horizonte != 4:
                        raise ValueError('El flujo ANN necesita un pronóstico de cuatro semanas.')
                    row['casos_pred_4sem'] = result['rnn']['total']
                except Exception as exc:
                    result['errores']['rnn'] = str(exc)
            if modo in ('accion', 'flujo'):
                try:
                    if modo == 'flujo' and ('cnn' in result['errores'] or 'rnn' in result['errores']):
                        raise ValueError('Falta completar detección y pronóstico para el flujo integrado.')
                    model = activo(models, 'ann')
                    result['ann'] = recomendar(model, row)
                    result['artefactos']['ann'] = model.config.archivo
                except Exception as exc:
                    result['errores']['ann'] = str(exc)
            result['entrada_modelos'] = row
            results.append(result)
    return {'resultados': results}
