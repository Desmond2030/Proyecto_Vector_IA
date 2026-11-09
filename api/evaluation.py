# ESTADO_ARCHIVO: EN_MODIFICACION
"""Evaluación real o importada, siempre vinculada a un artefacto y conjunto."""
from datetime import datetime, timezone
from uuid import uuid4
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_absolute_error, mean_squared_error
from api.datasets import matriz
from api.loaders import API, guardar_json, imagenes_temporales, serializar
from api.predict import activo, pronosticar

ARCHIVO = API / 'evaluaciones.json'


def leer():
    import json
    return json.loads(ARCHIVO.read_text(encoding='utf-8')) if ARCHIVO.exists() else {'resultados': [], 'seleccion': {}}


def registrar(model, conjunto, metricas, origen, **details):
    result = {'id': uuid4().hex, 'modelo': model.slot, 'archivo_modelo': model.config.archivo,
              'identidad': model.identidad, 'conjunto': conjunto, 'origen': origen,
              'fecha': datetime.now(timezone.utc).isoformat(), 'metricas': metricas, **details}
    return serializar(result)


def guardar(resultados):
    data = leer()
    for result in resultados:
        data['resultados'].append(result)
        data['seleccion'][result['modelo']] = result['id']
    guardar_json(ARCHIVO, data)
    return resultados


def importar(request, models):
    frame = pd.DataFrame(request.filas)
    needed = {'modelo', 'conjunto', 'metrica', 'valor'}
    if not needed.issubset(frame):
        raise ValueError('El CSV requiere modelo, conjunto, metrica y valor.')
    if frame[list(needed)].isna().any().any():
        raise ValueError('Las métricas no pueden tener campos vacíos.')
    frame['valor'] = pd.to_numeric(frame['valor'], errors='raise')
    if not np.isfinite(frame['valor']).all():
        raise ValueError('Las métricas deben ser finitas.')
    if frame.duplicated(['modelo', 'conjunto', 'metrica']).any():
        raise ValueError('Hay métricas duplicadas para el mismo modelo y conjunto.')
    results = []
    for (slot, conjunto), group in frame.groupby(['modelo', 'conjunto']):
        model = activo(models, slot)
        if not str(conjunto).strip():
            raise ValueError('Identifica el conjunto de evaluación.')
        results.append(registrar(model, str(conjunto), dict(zip(group.metrica, group.valor)), 'CSV importado'))
    return guardar(results)


def evaluar(request, models):
    model = activo(models, request.modelo)
    config = model.config
    if not request.registros:
        raise ValueError('Se requieren registros etiquetados.')
    frame = pd.DataFrame(request.registros)
    if request.modelo == 'cnn':
        import tensorflow as tf
        if request.objetivo not in frame:
            raise ValueError('Falta la columna con la especie real.')
        labels = frame[request.objetivo].astype(str).tolist()
        if not set(labels).issubset(config.clases):
            raise ValueError('Las etiquetas deben coincidir con las clases configuradas.')
        with imagenes_temporales(request.imagenes) as images:
            paths = [images[str(name)] for name in frame[request.columna_imagen]]
            # Preparación y evaluación de la clase CNN, sin gráficos ni particiones nuevas.
            ds = tf.data.Dataset.from_tensor_slices((paths, [config.clases.index(label) for label in labels]))
            ds = ds.map(lambda path, label: (model.cnn._cargar_imagen(path), label)).batch(16)
            model.cnn.val_data = ds
            try:
                report = model.cnn.evaluar(conjunto='val', mostrar=False)
            finally:
                model.cnn.val_data = None
        details = {'matriz': report['matriz_confusion'].reset_index(),
                   'por_clase': report['reporte_por_clase'].reset_index(), 'predicciones': report['predicciones']}
        metrics = report['metricas']
    elif request.modelo == 'ann':
        if request.objetivo not in frame:
            raise ValueError('Falta la acción real para evaluar.')
        truth = frame[request.objetivo].astype(str)
        if not set(truth).issubset(config.clases):
            raise ValueError('Las acciones reales no coinciden con las clases configuradas.')
        scores = model.ejecutar(matriz(frame, config, model.scaler))
        predicted = np.array(config.clases)[scores.argmax(axis=1)]
        report = classification_report(truth, predicted, labels=config.clases, output_dict=True, zero_division=0)
        metrics = {'accuracy': accuracy_score(truth, predicted), 'f1_macro': report['macro avg']['f1-score'],
                   'precision_macro': report['macro avg']['precision'], 'recall_macro': report['macro avg']['recall']}
        details = {'por_clase': [{'Clase': name, **report[name]} for name in config.clases],
                   'matriz': pd.DataFrame(confusion_matrix(truth, predicted, labels=config.clases),
                                          index=pd.Index(config.clases, name='Real'), columns=config.clases).reset_index(),
                   'predicciones': [{'real': a, 'predicha': b} for a, b in zip(truth, predicted)]}
    else:
        history = pd.DataFrame(request.historial)
        if config.objetivo not in history or config.fecha not in history:
            raise ValueError('El historial de evaluación requiere fecha y casos reales futuros.')
        history[config.fecha] = pd.to_datetime(history[config.fecha])
        points = []
        for row in request.registros:
            forecast = pronosticar(model, request.historial, row)
            truth = history
            if config.grupo:
                truth = truth.loc[truth[config.grupo].astype(str) == str(row[config.grupo])]
            for h, point in enumerate(forecast['semanas'], 1):
                match = truth.loc[truth[config.fecha] == pd.Timestamp(point['fecha']), config.objetivo]
                if len(match) != 1 or not np.isfinite(float(match.iloc[0])):
                    raise ValueError(f'Falta un valor real único para {point["fecha"]}.')
                points.append({'referencia': row[config.fecha], 'fecha': point['fecha'], 'horizonte': h,
                               'real': float(match.iloc[0]), 'prediccion': point['casos']})
        points = pd.DataFrame(points)
        metrics = {'mae': mean_absolute_error(points.real, points.prediccion),
                   'rmse': float(np.sqrt(mean_squared_error(points.real, points.prediccion)))}
        for h, group in points.groupby('horizonte'):
            metrics[f'mae_h{h}'] = mean_absolute_error(group.real, group.prediccion)
            metrics[f'rmse_h{h}'] = float(np.sqrt(mean_squared_error(group.real, group.prediccion)))
        details = {'predicciones': points}
    result = registrar(model, request.conjunto, metrics, 'Calculada', **details)
    guardar([result])
    return result
