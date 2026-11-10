# ESTADO_ARCHIVO: EN_MODIFICACION
"""Evaluaciones calculadas con las clases de src y persistidas en la API."""
from datetime import datetime, timezone
from uuid import uuid4
import json
import numpy as np
import pandas as pd
from api.loaders import API, ROOT, datos_procesados, guardar_json, muestras, serializar
from api.predict import activo

ARCHIVO = API / 'evaluaciones.json'
ACCIONES = 'data/proceced/archivos_parquet/vectoria_acciones.parquet'


def leer():
    return json.loads(ARCHIVO.read_text(encoding='utf-8')) if ARCHIVO.exists() else {'resultados': []}


def vigentes(models):
    records = leer()['resultados']
    return {slot: next((r for r in reversed(records)
                       if r['identidad'] == model.identidad and r['origen'] == 'Calculada'), None)
            for slot, model in models.items()}


def validacion_temporal(frame, fecha):
    frame = frame.sort_values(fecha, kind='stable').copy()
    dates = frame[fecha].drop_duplicates().sort_values()
    if len(dates) < 2:
        raise ValueError('No hay suficientes fechas para validación temporal.')
    return frame.loc[frame[fecha] >= dates.iloc[int(len(dates) * .8)]].copy()


def evaluar(request, models, errores=None):
    model = activo(models, request.modelo, errores)
    cfg = model.config
    if request.modelo == 'cnn':
        import tensorflow as tf
        images = []
        for label in cfg.clases:
            images.extend(muestras(request.conjunto, label, request.limite)['imagenes'])
        if not images:
            raise ValueError('No hay imágenes etiquetadas para evaluar.')
        paths = [str(ROOT / p['ruta']) for p in images]
        labels = [cfg.clases.index(p['especie']) for p in images]
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        ds = ds.map(lambda path, label: (model.instancia._cargar_imagen(path), label)).batch(16)
        model.instancia.val_data = ds
        try:
            report = model.instancia.evaluar(conjunto='val', mostrar=False)
        finally:
            model.instancia.val_data = None
        metrics = report['metricas']
        details = {'matriz': report['matriz_confusion'].reset_index(),
                   'por_clase': report['reporte_por_clase'].reset_index(), 'predicciones': report['predicciones']}
        source = f'Imágenes / {request.conjunto} / hasta {request.limite} por especie'
        count = len(images)
    elif request.modelo == 'ann':
        frame = validacion_temporal(datos_procesados(ACCIONES), cfg.fecha)
        model.instancia.make_prediction(data=frame[cfg.columnas], y=frame['recomendacion'],
            target='recomendacion', categorias=cfg.clases, eval=True, graficar=False)
        report = model.instancia.resultado
        macro = report['metricas'].set_index('promedio').loc['macro']
        metrics = {'accuracy': macro['accuracy'], 'f1_macro': macro['f1'],
                   'precision_macro': macro['precision'], 'recall_macro': macro['recall']}
        details = {'matriz': report['matriz_confusion'].reset_index(),
                   'por_clase': report['reporte_clasificacion'],
                   'predicciones': pd.DataFrame({'real': report['reales'].iloc[:, 0],
                                                  'predicha': report['predicciones'].iloc[:, 0]})}
        source, count = 'Acciones / validación temporal (último 20% de fechas)', len(frame)
    else:
        frame = datos_procesados(cfg.serie)
        frame[cfg.fecha] = pd.to_datetime(frame[cfg.fecha], errors='raise')
        if cfg.grupo:
            if request.ubicacion is None:
                raise ValueError('Selecciona la ubicación que deseas evaluar.')
            frame = frame.loc[frame[cfg.grupo].astype(str) == request.ubicacion]
        frame = validacion_temporal(frame, cfg.fecha).reset_index(drop=True)
        if frame[cfg.fecha].duplicated().any() or not frame[cfg.fecha].diff().dropna().eq(pd.Timedelta(days=7)).all():
            raise ValueError('La validación requiere semanas consecutivas de una sola ubicación.')
        x = model.scaler.transform(frame[cfg.columnas])
        y = model.scaler_salida.transform(frame[[cfg.objetivo]])
        # Reutilizar las ventanas de la clase; jamás ajustar escaladores al evaluar.
        x, y = model.instancia.generar_ventanas_dengue(x, y, tamaño_ventana=26, horizonte=4)
        if not len(x):
            raise ValueError('Se necesitan al menos 30 semanas en la partición de validación.')
        x, y = x[-request.limite:], y[-request.limite:]
        model.instancia.evaluar(x=x, y=y, y_scaler=model.scaler_salida, graficar=False)
        report = model.instancia.resultado
        summary = report['metricas'].set_index('horizonte')
        metrics = {'mae': summary.loc['global', 'MAE'], 'rmse': summary.loc['global', 'RMSE']}
        for h in range(1, 5):
            metrics[f'mae_h{h}'] = summary.loc[f'semana_{h}', 'MAE']
            metrics[f'rmse_h{h}'] = summary.loc[f'semana_{h}', 'RMSE']
        dates = frame[cfg.fecha].iloc[26:].tolist()
        dates = [dates[i:i + 4] for i in range(len(dates) - 3)][-len(x):]
        points = [{'referencia': (dates[i][0] - pd.Timedelta(weeks=1)).date().isoformat(),
                   'fecha': dates[i][h].date().isoformat(), 'horizonte': h + 1,
                   'real': float(report['reales'].iloc[i, h]), 'prediccion': float(report['predicciones'].iloc[i, h])}
                  for i in range(len(x)) for h in range(4)]
        details = {'por_horizonte': report['metricas'], 'predicciones': points}
        source = f'{cfg.serie.split("/")[-1]} / {request.ubicacion or "serie"} / últimas {len(x)} ventanas de validación temporal'
        count = len(x)
    result = serializar({'id': uuid4().hex, 'modelo': model.slot, 'archivo_modelo': cfg.archivo,
                         'identidad': model.identidad, 'conjunto': source, 'origen': 'Calculada',
                         'fecha': datetime.now(timezone.utc).isoformat(), 'muestras': count,
                         'metricas': metrics, **details})
    saved = leer()
    saved.pop('seleccion', None)
    saved['resultados'].append(result)
    guardar_json(ARCHIVO, saved)
    return result
