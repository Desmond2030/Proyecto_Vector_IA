# ESTADO_ARCHIVO: EN_MODIFICACION
import datetime
import streamlit as st
from app.client import ErrorAPI
from app.datasets import selector_dataset, cargar_imagenes, filas, mapear_entradas
from app.results import mostrar_resultado
from app.ui import cabecera, api_activa

cabecera('prediccion', '03', 'De la imagen a la acción.', 'Ingresa una observación o utiliza cada modelo por separado. El pronóstico requiere historial semanal.')
api = api_activa()
state = api.request('GET', '/configuracion')
config = state['configuracion']
mode = st.selectbox('Operación', ['flujo', 'mosquito', 'casos', 'accion'], format_func=lambda v: {'flujo': 'Flujo integrado', 'mosquito': 'Solo identificar imagen', 'casos': 'Solo pronosticar casos', 'accion': 'Solo recomendar acción'}[v])
history = None
if mode in ('flujo', 'casos'):
    with st.expander('Historial semanal para el pronóstico', expanded=True):
        _, history = selector_dataset(api, 'historial_individual', 'Serie histórica')
        if history is not None:
            cfg = config['rnn']
            history = mapear_entradas(history, cfg['columnas'] + [cfg['fecha']] + ([cfg['grupo']] if cfg['grupo'] else []), 'hist_ind')
st.html('<div class="steps"><b>01 · Imagen y contexto</b><b>02 · Pronóstico de casos</b><b>03 · Acción recomendada</b></div>')
image_col, context_col, result_col = st.columns([1, 1.15, 1.2], gap='large')
image = None
record = {}
with image_col:
    with st.container(border=True):
        st.subheader('Imagen del mosquito')
        if mode in ('flujo', 'mosquito'):
            image = st.file_uploader('Seleccionar imagen', type=['jpg', 'jpeg', 'png'], max_upload_size=20)
            if image:
                st.image(image, width='stretch')
                record['imagen'] = image.name
            else:
                st.caption('JPG o PNG · La imagen se procesará con la CNN seleccionada.')
        else:
            st.caption('Esta operación utiliza únicamente datos tabulares.')
with context_col:
    with st.container(border=True):
        st.subheader('Contexto de la observación')
        if mode in ('flujo', 'casos'):
            cfg = config['rnn']
            last_date = datetime.date.today()
            if history is not None:
                import pandas as pd
                parsed = pd.to_datetime(history[cfg['fecha']], errors='coerce').dropna()
                if not parsed.empty:
                    last_date = parsed.max().date()
            reference = st.date_input('Fecha de referencia', last_date)
            record[cfg['fecha']] = reference.isoformat()
            if cfg['grupo']:
                groups = sorted(history[cfg['grupo']].dropna().astype(str).unique()) if history is not None else []
                record[cfg['grupo']] = st.selectbox('Ubicación', groups) if groups else st.text_input('Ubicación', 'Costa Rica')
        if mode in ('flujo', 'accion'):
            auto = {'casos_pred_4sem', 'aegypti_detectado'} if mode == 'flujo' else set()
            labels = {'casos_umbral_endemico': 'Umbral endémico · casos / 4 semanas', 'lluvia_14d_mm': 'Lluvia de 14 días · mm',
                      'temp_max': 'Temperatura máxima · °C', 'temp_min': 'Temperatura mínima · °C', 'humedad': 'Humedad · %',
                      'semana_epi': 'Semana epidemiológica', 'casos_pred_4sem': 'Casos previstos en cuatro semanas', 'aegypti_detectado': 'Aegypti detectado · 0 o 1'}
            for column in config['ann']['columnas']:
                if column in auto:
                    continue
                if column == 'aegypti_detectado':
                    record[column] = st.selectbox(labels[column], [0, 1])
                elif column == 'semana_epi':
                    record[column] = st.number_input(labels[column], min_value=1, max_value=53, value=1)
                else:
                    value = st.number_input(labels.get(column, column), value=None, placeholder='Ingresa un valor', key='individual_' + column)
                    record[column] = value
            st.caption('Los campos vacíos no se sustituyen por ceros. Usa las unidades del entrenamiento.')
        run = st.button('Generar predicción', type='primary', width='stretch')
if run:
    try:
        with st.spinner('Procesando observación…'):
            response = api.request('POST', '/predict/' + mode, {'registros': [record], 'imagenes': cargar_imagenes([image]) if image else [],
                                   'historial': filas(history) if history is not None else [], 'revision': state['revision']})
        st.session_state['prediccion_individual'] = {'revision': state['revision'], 'resultado': response['resultados'][0]}
    except ErrorAPI as exc:
        st.error(str(exc))
with result_col:
    with st.container(border=True):
        st.subheader('Resultado')
        saved = st.session_state.get('prediccion_individual')
        if saved and saved['revision'] == state['revision']:
            st.caption('Última ejecución. Vuelve a generar al cambiar los datos.')
            mostrar_resultado(saved['resultado'])
        else:
            st.caption('El resultado aparecerá aquí después de ejecutar. Los módulos sin modelo se mostrarán como pendientes.')
