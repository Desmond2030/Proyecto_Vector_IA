# ESTADO_ARCHIVO: EN_MODIFICACION
"""Preparación con TabularDataProcessor y asociaciones con Dython."""
import numpy as np
import pandas as pd
from src.Tabular_Data_Processor import TabularDataProcessor
from api.loaders import tabla, serializar


def procesador(filas):
    processor = TabularDataProcessor()
    processor.data = pd.DataFrame(filas).copy()
    if processor.data.columns.duplicated().any():
        raise ValueError('No se permiten columnas duplicadas.')
    return processor


def convertir_dummies(processor, columnas):
    correspondence = []
    for column in columnas:
        if processor.data[column].isna().any():
            raise ValueError(f'{column}: elimina o completa los nulos antes de crear dummies.')
        # El método de src omite la última categoría observada. No lo reimplementamos.
        categories = list(processor.data[column].unique())
        before = set(processor.data.columns)
        processor.get_dummies(column, prefijo=column)
        created = [c for c in processor.data.columns if c not in before]
        correspondence.append({'variable': column, 'columnas': created, 'referencia': str(categories[-1])})
    return correspondence


def preparar(request):
    processor = procesador(request.filas)
    before = len(processor.data)
    if request.renombrar:
        names = [request.renombrar.get(c, c) for c in processor.data.columns]
        if len(names) != len(set(names)) or any(not c.strip() for c in names):
            raise ValueError('El mapeo produce nombres vacíos o repetidos.')
        processor.col_names = request.renombrar
    if request.eliminar_nulos:
        processor.data = processor.data.replace(r'^\s*$', np.nan, regex=True).dropna().copy()
    if processor.data.empty:
        raise ValueError('La preparación dejó el dataset sin filas.')
    if request.tipos:
        processor.set_types(request.tipos)
    for column, mapping in request.mapas.items():
        processor.set_types({column: 'string'})
        if processor.data[column].isna().any() or not set(processor.data[column]).issubset(mapping):
            raise ValueError(f'El mapa de {column} no cubre todos sus valores.')
        processor.replace_to_numeric_bool([column], map=mapping)
    dummies = convertir_dummies(processor, request.dummies)
    processor.set_info()
    return tabla(processor.data, dummies=dummies, excluidas=before - len(processor.data))


def generar_correlaciones(request):
    frame = pd.DataFrame(request.filas)
    columns = list(dict.fromkeys(request.columnas))
    if len(columns) < 2 or not set(columns).issubset(frame.columns):
        raise ValueError('Selecciona al menos dos columnas existentes.')
    categorical = [c for c in request.categoricas if c in columns]
    if request.modalidad == 'numericas':
        columns = [c for c in columns if c not in categorical]
        categorical = []
        if len(columns) < 2:
            raise ValueError('El análisis numérico necesita al menos dos variables numéricas.')
    original = frame[columns].replace(r'^\s*$', np.nan, regex=True)
    frame = original.dropna().copy()
    if len(frame) < 2:
        raise ValueError('Se necesitan al menos dos filas completas.')
    numerical = [c for c in columns if c not in categorical]
    frame[numerical] = frame[numerical].apply(pd.to_numeric, errors='raise')
    if not np.isfinite(frame[numerical].to_numpy(dtype=float)).all():
        raise ValueError('Hay valores numéricos infinitos.')
    dummy_info = []
    if request.modalidad == 'dummies':
        processor = procesador(serializar(frame))
        dummy_info = convertir_dummies(processor, categorical)
        frame = processor.data
    elif request.modalidad == 'numericas':
        frame = frame[numerical]
    constants = [c for c in frame if frame[c].nunique() < 2]
    usable = [c for c in frame if c not in constants]
    matrix = pd.DataFrame(np.nan, index=frame.columns, columns=frame.columns)
    if len(usable) >= 1:
        if request.modalidad == 'mixtas':
            from dython.nominal import associations
            computed = associations(frame[usable], nominal_columns=[c for c in categorical if c in usable],
                                    num_num_assoc=request.metodo, nom_nom_assoc='cramer',
                                    cramers_v_bias_correction=True, compute_only=True,
                                    nan_strategy='drop_samples')['corr']
        else:
            computed = frame[usable].corr(method=request.metodo)
        matrix.loc[usable, usable] = computed
    cells = []
    for a in matrix.index:
        for b in matrix.columns:
            method = request.metodo
            if request.modalidad == 'mixtas':
                if a in categorical and b in categorical:
                    method = 'V de Cramér'
                elif (a in categorical) != (b in categorical):
                    method = 'Eta (η)'
            cells.append({'variable_1': a, 'variable_2': b, 'valor': matrix.loc[a, b], 'metodo': method})
    return serializar({'celdas': cells, 'columnas': list(matrix.columns), 'constantes': constants,
                      'filas_utilizadas': len(frame), 'filas_excluidas': len(original) - len(frame), 'dummies': dummy_info})


def matriz(frame, config, scaler=None):
    if not config.columnas or not set(config.columnas).issubset(frame.columns):
        raise ValueError(f'Faltan entradas requeridas: {", ".join(c for c in config.columnas if c not in frame.columns)}')
    values = frame[config.columnas].apply(pd.to_numeric, errors='raise')
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError('Las entradas contienen nulos o valores infinitos.')
    if scaler is not None:
        values = scaler.transform(values)
    return np.asarray(values, dtype=np.float32)


def ventana(historial, registro, config, scaler=None):
    frame = pd.DataFrame(historial)
    if frame.empty or config.fecha not in frame or config.fecha not in registro:
        raise ValueError('Se requiere historial y fecha de referencia.')
    reference = pd.Timestamp(registro[config.fecha])
    frame[config.fecha] = pd.to_datetime(frame[config.fecha], errors='raise')
    if config.grupo:
        if config.grupo not in frame or config.grupo not in registro:
            raise ValueError(f'Falta el grupo temporal: {config.grupo}')
        frame = frame.loc[frame[config.grupo].astype(str) == str(registro[config.grupo])]
    frame = frame.loc[frame[config.fecha] <= reference].sort_values(config.fecha).tail(config.ventana)
    if len(frame) != config.ventana or frame[config.fecha].iloc[-1] != reference:
        raise ValueError(f'Se requieren {config.ventana} semanas hasta la fecha seleccionada.')
    if not frame[config.fecha].diff().dropna().eq(pd.Timedelta(days=7)).all():
        raise ValueError('El historial debe tener semanas consecutivas, sin duplicados.')
    return matriz(frame, config, scaler)[None, :, :], reference
