# ESTADO_ARCHIVO: EN_MODIFICACION
import copy
from pathlib import Path
import pandas as pd
import streamlit as st
from app.client import ErrorAPI
from app.datasets import selector_imagen, procedencia
from app.results import mostrar_resultado
from app.ui import cabecera, api_activa

cabecera('prediccion', '03', 'De la imagen a la acción.',
         'Una observación, cuatro semanas de pronóstico y una recomendación conjunta.')
api = api_activa()
state = api.request('GET', '/configuracion')
config = state['configuracion']
inventory = api.request('GET', '/artefactos')
revision = state['revision']


def elegir(label, options, current, key):
    choices = [None] + options
    return st.selectbox(label, choices, index=choices.index(current) if current in choices else 0,
                        format_func=lambda v: str(Path(v).relative_to('models')) if v and v.startswith('models/') else (v or 'Sin seleccionar'),
                        key=key + revision)


with st.container(border=True):
    st.subheader('Modelos del flujo')
    updated = copy.deepcopy(config)
    with st.form('configurar_modelos'):
        for col, slot, title in zip(st.columns(3), ['cnn', 'rnn', 'ann'],
                                    ['01 · Identificación CNN', '02 · Pronóstico RNN', '03 · Recomendación ANN']):
            with col:
                st.markdown(f'**{title}**')
                cfg = updated[slot]
                options = [p for p in inventory['modelo'] if slot.upper() + '_Models/' in p
                           or not any(kind + '_Models/' in p for kind in ['CNN', 'RNN', 'ANN'])]
                cfg['archivo'] = elegir('Modelo ' + slot.upper(), options, cfg['archivo'], slot + '_modelo')
                if slot != 'cnn':
                    cfg['scaler'] = elegir('Escalador de entradas ' + slot.upper(), inventory['scaler'], cfg['scaler'], slot + '_scaler')
                if slot == 'rnn':
                    cfg['scaler_salida'] = elegir('Escalador de casos', inventory['scaler'], cfg['scaler_salida'], 'salida')
                    series = [p for p in inventory['datos'] if 'dengue' in Path(p).name]
                    cfg['serie'] = elegir('Serie del modelo', series, cfg['serie'], 'serie')
                    st.caption('Asocia el modelo y sus escaladores originales. Los nacionales y provinciales no son intercambiables.')
                if slot in state['identidades']:
                    st.caption('● Modelo listo')
                else:
                    st.caption('○ ' + state['errores'].get(slot, 'Pendiente de configuración'))
        with st.expander('Configuración técnica'):
            a, b = st.columns(2)
            with a:
                updated['cnn']['clases'] = [s.strip() for s in st.text_input('Clases CNN, en orden de salida', ', '.join(config['cnn']['clases'])).split(',') if s.strip()]
                updated['cnn']['clase_deteccion'] = st.text_input('Clase de detección', config['cnn']['clase_deteccion'])
                normalizations = ['interna', 'dividir_255', 'ninguna']
                updated['cnn']['normalizacion'] = st.selectbox('Normalización CNN', normalizations, index=normalizations.index(config['cnn']['normalizacion']))
                updated['ann']['clases'] = [s.strip() for s in st.text_input('Clases ANN, en orden de salida', ', '.join(config['ann']['clases'])).split(',') if s.strip()]
            with b:
                updated['rnn']['columnas'] = [s.strip() for s in st.text_area('Entradas RNN, en orden de entrenamiento', ', '.join(config['rnn']['columnas']), height=160).split(',') if s.strip()]
                st.caption('Ventana: 26 semanas · Horizonte: 4 semanas. ANN: siete entradas del flujo.')
        apply = st.form_submit_button('Aplicar selección de modelos', type='primary')
    if apply:
        try:
            if not updated['rnn']['serie']:
                raise ValueError('Selecciona la serie procesada de la RNN.')
            with st.spinner('Cargando y comprobando modelos…'):
                api.request('PUT', '/configuracion', updated)
            st.session_state.pop('prediccion_individual', None)
            st.rerun()
        except (ErrorAPI, ValueError) as exc:
            st.error(str(exc))
    with st.expander('Cargar un modelo o escalador'):
        kind = st.radio('Artefacto', ['modelo', 'scaler'], horizontal=True, format_func=lambda v: 'Modelo' if v == 'modelo' else 'Escalador')
        upload = st.file_uploader('Archivo del entrenamiento', type=['keras', 'h5'] if kind == 'modelo' else None,
                                  max_upload_size=500, key='artefacto_' + kind)
        st.caption('También se admiten escaladores sin extensión. Después de cargarlo, selecciónalo arriba.')
        if st.button('Cargar archivo', disabled=upload is None):
            try:
                api.subir(upload, kind)
                st.rerun()
            except ErrorAPI as exc:
                st.error(str(exc))

st.html('<div class="steps"><b>01 · Clasificar imagen</b><b>02 · Pronosticar cuatro semanas</b><b>03 · Recomendar acción</b></div>')
image_col, context_col = st.columns([1, 1.5], gap='large')
with image_col:
    with st.container(border=True):
        st.subheader('Imagen del mosquito')
        image = selector_imagen(api, 'prediccion')
with context_col:
    with st.container(border=True):
        st.subheader('Fecha, ubicación y clima')
        cfg = config['rnn']
        payload = api.request('GET', '/datos', params={'nombre': cfg['serie']})
        history = pd.DataFrame(payload['filas'])
        history[cfg['fecha']] = pd.to_datetime(history[cfg['fecha']])
        groups = sorted(history[cfg['grupo']].astype(str).unique()) if cfg['grupo'] else ['Serie completa']
        a, b = st.columns(2)
        location = a.selectbox('Ubicación', groups, key='ubicacion_' + cfg['serie'])
        selected = history.loc[history[cfg['grupo']].astype(str) == location] if cfg['grupo'] else history
        dates = selected[cfg['fecha']].sort_values().drop_duplicates().dt.date.tolist()
        reference = b.selectbox('Última semana observada', dates, index=len(dates) - 1, key='fecha_' + cfg['serie'] + location)
        st.caption('El pronóstico usa las 26 semanas hasta esta fecha. Los campos siguientes alimentan la recomendación ANN.')
        procedencia(cfg['serie'])
        with st.expander('Ver historial utilizado'):
            st.dataframe(selected.loc[selected[cfg['fecha']].dt.date <= reference].sort_values(cfg['fecha']).tail(26),
                         hide_index=True, width='stretch', height=220)
        a, b = st.columns(2)
        with a:
            threshold = st.number_input('Umbral endémico · casos / 4 semanas', min_value=0.0, value=None)
            rain = st.number_input('Lluvia de 14 días · mm', min_value=0.0, value=None)
            humidity = st.number_input('Humedad · %', min_value=0.0, max_value=100.0, value=None)
        with b:
            temp_max = st.number_input('Temperatura máxima · °C', value=None)
            temp_min = st.number_input('Temperatura mínima · °C', value=None)
            st.caption('La detección y los casos previstos se incorporan automáticamente desde CNN y RNN.')

context = dict(casos_umbral_endemico=threshold, lluvia_14d_mm=rain, temp_max=temp_max, temp_min=temp_min, humedad=humidity)
request = {'imagen': image, 'fecha': reference.isoformat(), 'ubicacion': location, 'contexto': context, 'revision': revision}
complete = image is not None and all(v is not None for v in context.values())
if st.button('Generar predicción conjunta', type='primary', disabled=not complete, width='stretch'):
    try:
        with st.spinner('Clasificando, pronosticando y preparando la recomendación…'):
            response = api.request('POST', '/predict/flujo', request)
        st.session_state['prediccion_individual'] = {'entrada': request, 'resultado': response}
    except ErrorAPI as exc:
        st.error(str(exc))
if not complete:
    st.caption('Selecciona una imagen y completa los cinco valores de contexto para ejecutar.')
st.divider()
st.subheader('Resultados de la observación')
saved = st.session_state.get('prediccion_individual')
if saved and saved['entrada'] == request:
    mostrar_resultado(saved['resultado'])
else:
    st.caption('Aquí aparecerán juntos los resultados de CNN, RNN y ANN. Ejecuta nuevamente después de cambiar una entrada.')
