# ESTADO_ARCHIVO: EN_MODIFICACION
import pandas as pd
import streamlit as st
from app.client import ErrorAPI
from app.datasets import selector_dataset, cargar_imagenes, filas, mapear_entradas
from app.results import mostrar_evaluacion
from app.ui import cabecera, api_activa

cabecera('evaluacion', '05', 'Resultados con contexto.', 'Importa métricas de tus notebooks o evalúa con etiquetas reales. Inicio utiliza la evaluación seleccionada de cada modelo.')
api = api_activa()
state = api.request('GET', '/configuracion')
slot = st.selectbox('Modelo', ['cnn', 'rnn', 'ann'], format_func=str.upper)
cfg = state['configuracion'][slot]
if slot not in state['identidades']:
    st.info('Este modelo está pendiente. Configura su artefacto antes de importar o calcular métricas.')
import_tab, calculate_tab, saved_tab = st.tabs(['Importar CSV', 'Calcular evaluación', 'Resultados guardados'])
with import_tab:
    st.caption('Columnas requeridas: modelo (cnn/rnn/ann), conjunto, metrica, valor. Los valores se asocian a los artefactos activos.')
    uploaded = st.file_uploader('CSV de métricas', type=['csv'], max_upload_size=20, key='csv_metricas')
    if st.button('Importar métricas', type='primary', disabled=uploaded is None):
        try:
            dataset = api.subir(uploaded)
            api.request('POST', '/evaluaciones/importar', {'filas': dataset['filas'], 'revision': state['revision']})
            st.success('Métricas importadas y seleccionadas para el resumen de Inicio.')
        except ErrorAPI as exc:
            st.error(str(exc))
with calculate_tab:
    name, frame = selector_dataset(api, 'evaluar_' + slot, 'Conjunto etiquetado')
    conjunto = st.text_input('Nombre del conjunto de evaluación', value=name or '', key='conjunto_' + slot)
    objective = 'recomendacion'
    images = []
    image_column = 'imagen'
    history = None
    if frame is not None:
        if slot in ('cnn', 'ann'):
            expected = 'especie' if slot == 'cnn' else 'recomendacion'
            objective = st.selectbox('Columna de etiqueta real', list(frame.columns), index=list(frame.columns).index(expected) if expected in frame else None, key='etiqueta_' + slot)
        if slot == 'cnn':
            image_column = st.selectbox('Columna de nombre de imagen', list(frame.columns), index=list(frame.columns).index('imagen') if 'imagen' in frame else None, key='eval_img')
            images = st.file_uploader('Imágenes etiquetadas', type=['jpg', 'jpeg', 'png'], accept_multiple_files=True, max_upload_size=20, key='imagenes_eval')
        elif slot == 'ann':
            frame = mapear_entradas(frame, cfg['columnas'], 'eval_ann')
        else:
            st.caption('El historial debe incluir las semanas reales del horizonte evaluado. Las entradas del modelo solo usan fechas anteriores o iguales a la referencia.')
            _, history = selector_dataset(api, 'eval_historial', 'Historial y valores reales')
            fields = cfg['columnas'] + [cfg['fecha'], cfg['objetivo']] + ([cfg['grupo']] if cfg['grupo'] else [])
            if history is not None:
                history = mapear_entradas(history, fields, 'eval_hist')
            frame = mapear_entradas(frame, [cfg['fecha']] + ([cfg['grupo']] if cfg['grupo'] else []), 'eval_ref')
            if frame is not None:
                options = list(range(len(frame)))
                positions = st.multiselect('Observaciones de referencia que se evaluarán', options,
                                          format_func=lambda i: f"Fila {i + 1} · {frame.iloc[i][cfg['fecha']]}" + (f" · {frame.iloc[i][cfg['grupo']]}" if cfg['grupo'] else ''))
                frame = frame.iloc[positions]
        if st.button('Calcular métricas', type='primary', disabled=slot not in state['identidades'] or frame is None or frame.empty or not conjunto.strip()):
            try:
                if slot in ('cnn', 'ann') and objective is None:
                    raise ValueError('Selecciona la etiqueta real.')
                if slot == 'cnn' and image_column is None:
                    raise ValueError('Selecciona la columna de imagen.')
                with st.spinner('Evaluando datos etiquetados…'):
                    api.request('POST', '/evaluate', {'modelo': slot, 'conjunto': conjunto, 'objetivo': objective or '',
                                'registros': filas(frame), 'imagenes': cargar_imagenes(images), 'columna_imagen': image_column or 'imagen',
                                'historial': filas(history) if history is not None else [], 'revision': state['revision']}, timeout=300)
                st.success('Evaluación calculada y seleccionada para Inicio.')
            except (ErrorAPI, ValueError) as exc:
                st.error(str(exc))
with saved_tab:
    saved = api.request('GET', '/evaluaciones')
    identity = state['identidades'].get(slot)
    results = [r for r in saved['resultados'] if r['modelo'] == slot and r['identidad'] == identity]
    if not results:
        st.info('Sin evaluación cargada para el artefacto activo.')
    else:
        selected_id = saved['seleccion'].get(slot)
        default = next((i for i, r in enumerate(results) if r['id'] == selected_id), len(results) - 1)
        selected = st.selectbox('Evaluación', results, index=default, format_func=lambda r: f"{r['conjunto']} · {r['origen']} · {r['fecha'][:19]}")
        if st.button('Mostrar esta evaluación en Inicio'):
            api.request('POST', '/evaluaciones/seleccionar', {'modelo': slot, 'id': selected['id']})
            st.success('Resumen de Inicio actualizado.')
        mostrar_evaluacion(selected)
