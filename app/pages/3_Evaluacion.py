# ESTADO_ARCHIVO: EN_MODIFICACION
import pandas as pd
import streamlit as st
from app.client import ErrorAPI
from app.results import mostrar_evaluacion
from app.ui import cabecera, api_activa
from app.datasets import procedencia

cabecera('evaluacion', '04', 'Resultados con contexto.',
         'Evalúa cada modelo con etiquetas reales. La última evaluación calculada actualiza Inicio automáticamente.')
api = api_activa()
state = api.request('GET', '/configuracion')
slot = st.selectbox('Modelo que se evaluará', ['cnn', 'rnn', 'ann'],
                    format_func=lambda v: {'cnn': 'CNN · Especie', 'rnn': 'RNN · Cuatro semanas', 'ann': 'ANN · Acción recomendada'}[v])
cfg = state['configuracion'][slot]
ready = slot in state['identidades']
if not ready:
    st.info(state['errores'].get(slot, 'Selecciona el modelo en Predicción.'))
    st.page_link('pages/2_Prediccion.py', label='Configurar modelos', icon=':material/tune:')
else:
    st.caption('Modelo activo: ' + cfg['archivo'])

calculate, saved_tab = st.tabs(['Evaluar modelo', 'Historial de resultados'])
with calculate:
    request = {'modelo': slot, 'revision': state['revision']}
    if slot == 'cnn':
        a, b = st.columns(2)
        request['conjunto'] = a.selectbox('Conjunto de imágenes', ['val', 'test'], format_func=lambda v: 'Validación' if v == 'val' else 'Prueba final')
        request['limite'] = b.number_input('Máximo de imágenes por especie', min_value=1, max_value=1000, value=20, step=1)
        st.caption('Se toman las primeras imágenes por nombre de cada especie, hasta el máximo indicado. Se mostrará el tamaño de la muestra evaluada.')
    elif slot == 'ann':
        st.write('Dataset de acciones procesado · último 20% de fechas.')
        st.caption('Todos los escenarios de una misma fecha permanecen juntos. El escalador original se utiliza sin volver a ajustarlo.')
        procedencia('vectoria_acciones.parquet')
    else:
        source = api.request('GET', '/datos', params={'nombre': cfg['serie']})
        frame = pd.DataFrame(source['filas'])
        a, b = st.columns(2)
        request['ubicacion'] = a.selectbox('Ubicación', sorted(frame[cfg['grupo']].astype(str).unique())) if cfg['grupo'] else None
        request['limite'] = b.number_input('Máximo de ventanas a evaluar', min_value=1, max_value=1000, value=20, step=1)
        st.caption('Último 20% de fechas de la serie elegida. Cada ventana usa 26 semanas y se compara con las cuatro semanas posteriores. Se evalúan las últimas ventanas disponibles.')
        procedencia(cfg['serie'])
    if st.button('Calcular y actualizar métricas', type='primary', disabled=not ready):
        try:
            with st.spinner('Calculando métricas con el modelo seleccionado…'):
                result = api.request('POST', '/evaluate', request, timeout=600)
            st.session_state['evaluacion_reciente'] = result
            st.success('Evaluación guardada. Las métricas de Inicio están actualizadas.')
        except ErrorAPI as exc:
            st.error(str(exc))
    recent = st.session_state.get('evaluacion_reciente')
    if recent and recent['modelo'] == slot and recent['identidad'] == state['identidades'].get(slot):
        mostrar_evaluacion(recent, key='reciente')

with saved_tab:
    saved = api.request('GET', '/evaluaciones')
    results = [r for r in reversed(saved['resultados']) if r['modelo'] == slot and r.get('origen') == 'Calculada']
    if not results:
        st.info('Todavía no hay evaluaciones calculadas para este tipo de modelo.')
    else:
        selected = st.selectbox('Evaluación guardada', results,
                                format_func=lambda r: f"{r['fecha'][:19]} · {r['archivo_modelo'].split('/')[-1]} · {r['conjunto']}")
        if selected['identidad'] != state['identidades'].get(slot):
            st.caption('Esta evaluación corresponde a otra configuración. Se conserva como historial y no se utiliza en Inicio.')
        mostrar_evaluacion(selected, key='historial')
