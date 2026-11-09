# ESTADO_ARCHIVO: EN_MODIFICACION
import streamlit as st
from app.client import ErrorAPI
from app.datasets import selector_dataset, cargar_imagenes, filas, mapear_entradas
from app.results import tabla_predicciones, descargar
from app.ui import cabecera, api_activa

cabecera('lotes', '04', 'Una colección, un mismo flujo.', 'Vincula imágenes por nombre de archivo y conserva cada observación hasta la descarga de resultados.')
api = api_activa()
state = api.request('GET', '/configuracion')
mode = st.selectbox('Operación del lote', ['flujo', 'mosquito', 'casos', 'accion'], format_func=lambda v: {'flujo': 'Flujo integrado', 'mosquito': 'Clasificación de imágenes', 'casos': 'Pronóstico temporal', 'accion': 'Recomendación de acciones'}[v])
name, frame = selector_dataset(api, 'lotes', 'CSV de observaciones')
images = []
image_column = 'imagen'
if mode in ('flujo', 'mosquito'):
    images = st.file_uploader('Imágenes del lote · JPG o PNG', type=['jpg', 'jpeg', 'png'], accept_multiple_files=True, max_upload_size=20)
    if frame is not None:
        image_column = st.selectbox('Columna con el nombre de imagen', list(frame.columns), index=list(frame.columns).index('imagen') if 'imagen' in frame else None)
    if images:
        st.caption(f'{len(images)} imágenes seleccionadas. Se vinculan por nombre exacto, nunca por posición.')
history = None
if mode in ('flujo', 'casos'):
    with st.expander('Historial para construir las ventanas', expanded=True):
        _, history = selector_dataset(api, 'historial_lotes', 'Serie histórica')
        cfg = state['configuracion']['rnn']
        fields = cfg['columnas'] + [cfg['fecha']] + ([cfg['grupo']] if cfg['grupo'] else [])
        if history is not None:
            history = mapear_entradas(history, fields, 'hist_lote')
if frame is not None:
    cfg = state['configuracion']
    required = []
    if mode in ('flujo', 'casos'):
        required += [cfg['rnn']['fecha']] + ([cfg['rnn']['grupo']] if cfg['rnn']['grupo'] else [])
    if mode in ('flujo', 'accion'):
        required += [c for c in cfg['ann']['columnas'] if mode == 'accion' or c not in ('casos_pred_4sem', 'aegypti_detectado')]
    if required:
        frame = mapear_entradas(frame, required, 'lote')
    if frame is not None:
        st.dataframe(frame.head(100), hide_index=True, width='stretch')
        st.caption(f'{len(frame)} registros. La vista previa muestra hasta 100; la ejecución procesa todos.')
if st.button('Procesar lote', type='primary', disabled=frame is None and not (mode == 'mosquito' and images)):
    try:
        if mode in ('flujo', 'mosquito') and frame is not None and not image_column:
            raise ValueError('Selecciona la columna que identifica las imágenes.')
        with st.spinner('Procesando lote…'):
            response = api.request('POST', '/predict/' + mode, {'registros': filas(frame) if frame is not None else [],
                                   'imagenes': cargar_imagenes(images), 'historial': filas(history) if history is not None else [],
                                   'columna_imagen': image_column or 'imagen', 'revision': state['revision']}, timeout=300)
        st.session_state['resultado_lote'] = {'revision': state['revision'], 'resultados': response['resultados']}
    except (ErrorAPI, ValueError) as exc:
        st.error(str(exc))
saved = st.session_state.get('resultado_lote')
if saved and saved['revision'] == state['revision']:
    st.subheader('Resultados del último lote')
    results = saved['resultados']
    failed = sum(bool(r['errores']) for r in results)
    st.caption(f'{len(results)} registros · {failed} con alguna etapa pendiente o inválida')
    output = tabla_predicciones(results)
    st.dataframe(output, hide_index=True, width='stretch')
    descargar(output, 'Descargar datos con predicciones', 'predicciones_vectoria.csv')
