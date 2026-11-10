# ESTADO_ARCHIVO: EN_MODIFICACION
"""Una observación: CNN, cuatro semanas RNN y recomendación ANN."""
import numpy as np
import pandas as pd
from api.datasets import ventana
from api.loaders import datos_procesados, imagen_temporal


def activo(models, slot, errores=None):
    if slot not in models:
        reason = (errores or {}).get(slot, 'Selecciona el modelo y sus escaladores en Predicción.')
        raise ValueError(f'{slot.upper()} pendiente: {reason}')
    return models[slot]


def pronosticar(model, fecha, ubicacion):
    cfg = model.config
    history = datos_procesados(cfg.serie)
    values, reference = ventana(history.to_dict('records'),
                                {cfg.fecha: fecha, cfg.grupo: ubicacion}, cfg, model.scaler)
    output = model.ejecutar(values).reshape(-1, 1)
    prediction = model.scaler_salida.inverse_transform(output).reshape(-1)
    if not np.isfinite(prediction).all():
        raise ValueError('El pronóstico desescalado no es finito.')
    return {'semanas': [{'fecha': (reference + pd.Timedelta(weeks=i + 1)).date().isoformat(), 'casos': float(p)}
                        for i, p in enumerate(prediction)], 'total': float(prediction.sum())}


def predecir(request, models, errores=None):
    row = request.contexto.model_dump()
    result = {'cnn': None, 'rnn': None, 'ann': None, 'errores': {}, 'artefactos': {},
              'fecha': request.fecha, 'ubicacion': request.ubicacion}
    try:
        model = activo(models, 'cnn', errores)
        with imagen_temporal(request.imagen) as path:
            result['cnn'] = model.clasificar(str(path))
        row['aegypti_detectado'] = result['cnn']['aegypti_detectado']
        result['artefactos']['cnn'] = model.config.archivo
    except Exception as exc:
        result['errores']['cnn'] = str(exc)
    try:
        model = activo(models, 'rnn', errores)
        result['rnn'] = pronosticar(model, request.fecha, request.ubicacion)
        row['casos_pred_4sem'] = result['rnn']['total']
        result['artefactos']['rnn'] = model.config.archivo
    except Exception as exc:
        result['errores']['rnn'] = str(exc)
    try:
        model = activo(models, 'ann', errores)
        if result['cnn'] is None or result['rnn'] is None:
            raise ValueError('La recomendación requiere completar CNN y RNN.')
        result['ann'] = model.recomendar(pd.DataFrame([row]))
        result['artefactos']['ann'] = model.config.archivo
    except Exception as exc:
        result['errores']['ann'] = str(exc)
    result['entrada_modelos'] = row
    return result
