# ESTADO_ARCHIVO: EN_MODIFICACION
import copy
import streamlit as st
from app.client import ErrorAPI
from app.ui import cabecera, api_activa

cabecera('modelos', '06', 'Cada modelo, su configuración.', 'Selecciona artefactos del proyecto o carga los tuyos. Confirma el contrato utilizado durante el entrenamiento.')
api = api_activa()
try:
    inventory = api.request('GET', '/artefactos')
    state = api.request('GET', '/configuracion')
except ErrorAPI as exc:
    st.error(str(exc))
    st.stop()
with st.expander('Subir modelo o escalador'):
    st.caption('Archivos propios de confianza. Los nuevos artefactos se guardan dentro de api/archivos/.')
    kind = st.radio('Tipo', ['modelo', 'scaler'], horizontal=True)
    file = st.file_uploader('Artefacto', type=['keras', 'h5'] if kind == 'modelo' else ['pkl', 'joblib'], max_upload_size=500)
    if st.button('Subir archivo', disabled=file is None):
        try:
            response = api.subir(file, kind)
            st.session_state['subida_modelo'] = response['mensaje']
            st.rerun()
        except ErrorAPI as exc:
            st.error(str(exc))
if st.session_state.get('subida_modelo'):
    st.info(st.session_state.pop('subida_modelo'))

def elegir(label, values, actual, key):
    options = [None] + list(dict.fromkeys(values + ([actual] if actual else [])))
    return st.selectbox(label, options, index=options.index(actual), key=key, format_func=lambda x: x or 'Sin seleccionar')

config = copy.deepcopy(state['configuracion'])
with st.form('configurar_modelos'):
    tabs = st.tabs(['CNN · Imagen', 'RNN · Pronóstico', 'ANN · Acción'])
    for tab, slot in zip(tabs, ['cnn', 'rnn', 'ann']):
        with tab:
            cfg = config[slot]
            cfg['archivo'] = elegir('Modelo Keras', inventory['modelo'], cfg['archivo'], slot + '_archivo')
            if slot in ('cnn', 'ann'):
                cfg['clases'] = [v.strip() for v in st.text_area('Clases, una por línea y en el orden entrenado', '\n'.join(cfg['clases']), key=slot + '_clases').splitlines() if v.strip()]
                st.caption('El orden mostrado es una propuesta de configuración. Debe coincidir con los índices del modelo elegido.')
            if slot == 'cnn':
                cfg['normalizacion'] = st.selectbox('Normalización', ['interna', 'dividir_255', 'ninguna'], index=['interna', 'dividir_255', 'ninguna'].index(cfg['normalizacion']))
                cfg['clase_deteccion'] = st.text_input('Clase que activa aegypti_detectado', cfg['clase_deteccion'])
                st.caption('Alto, ancho y canales se leen del modelo. Se reutiliza la preparación de imágenes de src/.')
            else:
                cfg['columnas'] = [v.strip() for v in st.text_area('Entradas, una por línea y en el orden entrenado', '\n'.join(cfg['columnas']), key=slot + '_columnas').splitlines() if v.strip()]
                cfg['scaler'] = elegir('Escalador de entrada', inventory['scaler'], cfg['scaler'], slot + '_scaler')
                if slot == 'rnn':
                    a, b = st.columns(2)
                    cfg['ventana'] = a.number_input('Semanas de historial', 1, 520, int(cfg['ventana']))
                    cfg['horizonte'] = b.number_input('Semanas a pronosticar', 1, 52, int(cfg['horizonte']))
                    cfg['fecha'] = st.text_input('Columna de fecha', cfg['fecha'])
                    cfg['grupo'] = st.text_input('Columna de ubicación (vacía para una sola serie)', cfg['grupo'])
                    cfg['objetivo'] = st.text_input('Columna de casos reales', cfg['objetivo'])
                    cfg['scaler_salida'] = elegir('Escalador de salida', inventory['scaler'], cfg['scaler_salida'], 'rnn_salida')
    submitted = st.form_submit_button('Validar y aplicar configuración', type='primary')
if submitted:
    try:
        with st.spinner('Cargando y validando artefactos…'):
            api.request('PUT', '/configuracion', config, timeout=300)
        for key in ['prediccion_individual', 'resultado_lote']:
            st.session_state.pop(key, None)
        st.success('Configuración aplicada. Los módulos sin artefacto permanecen pendientes.')
    except ErrorAPI as exc:
        st.error(str(exc))
st.caption('No se entrenan modelos ni se ajustan escaladores desde esta página.')
