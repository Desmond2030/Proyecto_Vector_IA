# ESTADO_ARCHIVO: EN_MODIFICACION
"""Estado de sesión y controles; las transformaciones se delegan a src a través de la API."""
import base64
import json
import pandas as pd
import streamlit as st
from app.client import ErrorAPI


def filas(frame):
    return json.loads(frame.to_json(orient='records', date_format='iso'))


def registrar_dataset(payload):
    frame = pd.DataFrame(payload['filas'], columns=payload['columnas'])
    if '_fila_origen' not in frame:
        frame.insert(0, '_fila_origen', range(1, len(frame) + 1))
    st.session_state.setdefault('datasets', {})[payload['fuente']] = frame
    st.session_state['dataset_activo'] = payload['fuente']


def selector_dataset(api, key='datos', label='Dataset', cargar=True):
    library = st.session_state.setdefault('datasets', {})
    if cargar:
        with st.expander('Abrir datos del proyecto o subir un archivo', expanded=not library):
            left, right = st.columns(2)
            with left:
                try:
                    names = api.request('GET', '/datos')
                    selected = st.selectbox('Archivo del proyecto', names, index=None, key=key + '_local')
                    if st.button('Abrir archivo', key=key + '_abrir', disabled=not selected):
                        registrar_dataset(api.request('GET', '/datos', params={'nombre': selected}))
                        st.rerun()
                except ErrorAPI as exc:
                    st.error(str(exc))
            with right:
                upload = st.file_uploader('CSV, Excel o Parquet', type=['csv', 'xlsx', 'parquet'], max_upload_size=20, key=key + '_upload')
                if st.button('Cargar archivo', key=key + '_cargar', disabled=upload is None):
                    try:
                        registrar_dataset(api.subir(upload))
                        st.rerun()
                    except ErrorAPI as exc:
                        st.error(str(exc))
    if not library:
        st.info('Abre un dataset para continuar. No necesitas modelos para explorarlo.')
        return None, None
    names = list(library)
    active = st.session_state.get('dataset_activo')
    name = st.selectbox(label, names, index=names.index(active) if active in names else 0, key=key + '_seleccion')
    st.caption(f'Fuente: {name} · Copia de trabajo en esta sesión')
    if 'semanal.csv' in name:
        st.caption('Datos sintéticos; no representan cifras oficiales.')
    elif 'acciones' in name:
        st.caption('Datos mixtos: contexto real y escenarios/etiquetas simulados.')
    return name, library[name].copy()


def editor_dataset(api, name, frame):
    st.caption('Edita celdas o elimina filas. Los cambios solo se aplican al pulsar Guardar preparación.')
    st.caption('Guardar conserva esta selección como copia de trabajo de la sesión. El archivo original permanece intacto.')
    revision = st.session_state.get('edicion_revision', 0)
    edited = st.data_editor(frame, num_rows='dynamic', disabled=['_fila_origen'], width='stretch', height=350,
                            key=f'editor_{name}_{revision}')
    with st.expander('Tipos, nombres, categorías y dummies'):
        a, b = st.columns(2)
        with a:
            rename = st.text_area('Renombrar columnas (JSON)', '{}', key='renombrar_' + name,
                                  help='Ejemplo: {"fecha": "fecha_inicio_semana"}')
            types = st.text_area('Cambiar tipos (JSON)', '{}', key='tipos_' + name,
                                 help='Ejemplo: {"lluvia_mm": "float64", "provincia": "category"}')
        with b:
            maps = st.text_area('Mapas de categorías (JSON)', '{}', key='mapas_' + name,
                                help='Ejemplo: {"deteccion": {"Sí": 1, "No": 0}}')
            dummies = st.multiselect('Generar dummies', [c for c in frame if c != '_fila_origen'], key='dummy_' + name)
        remove_null = st.checkbox('Eliminar filas con valores faltantes', key='nulos_' + name)
        remove = st.multiselect('Eliminar columnas', [c for c in frame if c != '_fila_origen'], key='eliminar_' + name)
    if st.button('Guardar preparación', type='primary'):
        try:
            edited = edited.drop(columns=remove)
            renames = json.loads(rename)
            if '_fila_origen' in renames and renames['_fila_origen'] != '_fila_origen':
                raise ValueError('Conserva _fila_origen para mantener la correspondencia de los registros.')
            if edited['_fila_origen'].isna().any():
                start = int(frame['_fila_origen'].max()) + 1
                mask = edited['_fila_origen'].isna()
                edited.loc[mask, '_fila_origen'] = range(start, start + int(mask.sum()))
            response = api.request('POST', '/datasets/prepare', {'filas': filas(edited), 'renombrar': renames,
                                   'tipos': json.loads(types), 'mapas': json.loads(maps), 'dummies': dummies,
                                   'eliminar_nulos': remove_null})
            st.session_state['datasets'][name] = pd.DataFrame(response['filas'])
            st.session_state['dataset_activo'] = name
            st.session_state['preparacion_info'] = response['dummies']
            st.session_state['edicion_revision'] = revision + 1
            st.rerun()
        except (ErrorAPI, ValueError, KeyError, TypeError) as exc:
            st.error(str(exc))
    if st.session_state.get('preparacion_info'):
        st.dataframe(st.session_state['preparacion_info'], hide_index=True, width='stretch')


def cargar_imagenes(files):
    return [{'nombre': f.name, 'contenido': base64.b64encode(f.getvalue()).decode('ascii')} for f in files]


def mapear_entradas(frame, columns, key):
    """Solo selecciona correspondencias; no altera el CSV ni inventa valores."""
    mapping = {}
    with st.expander('Mapear columnas a las entradas del modelo', expanded=not set(columns).issubset(frame.columns)):
        available = list(frame.columns)
        for column in dict.fromkeys(columns):
            source = st.selectbox(column, available, index=available.index(column) if column in available else None,
                                  key=key + '_' + column, placeholder='Selecciona una columna')
            mapping[column] = source
    if any(value is None for value in mapping.values()):
        st.info('Completa el mapeo de columnas para habilitar el envío.')
        return None
    result = frame.copy()
    for target, source in mapping.items():
        result[target] = frame[source]
    return result
