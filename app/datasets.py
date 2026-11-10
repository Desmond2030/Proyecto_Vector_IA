# ESTADO_ARCHIVO: EN_MODIFICACION
"""Copias de exploración y selección de muestras; los originales se leen por API."""
import base64
import json
from pathlib import Path
import pandas as pd
import streamlit as st
from app.client import ApiClient, ErrorAPI


def filas(frame):
    return json.loads(frame.to_json(orient='records', date_format='iso'))


def procedencia(name):
    if 'dengue_semanal' in name:
        st.caption('Serie provincial sintética; no representa cifras oficiales.')
    elif 'acciones' in name:
        st.caption('Acciones: contexto real y escenarios, predicciones y etiquetas simulados.')
    elif 'nacional_real' in name:
        st.caption('Serie nacional: casos y clima del dataset académico procesado.')


def selector_dataset(api, key='datos', label='Dataset procesado'):
    names = api.request('GET', '/datos')
    if not names:
        st.info('No hay archivos en la carpeta de datos procesados.')
        return None, None
    name = st.selectbox(label, names, format_func=lambda p: Path(p).stem.replace('vectoria_', '').replace('_', ' ').capitalize(), key=key + '_seleccion')
    library = st.session_state.setdefault('datasets', {})
    if name not in library:
        payload = api.request('GET', '/datos', params={'nombre': name})
        frame = pd.DataFrame(payload['filas'], columns=payload['columnas'])
        frame.insert(0, '_fila_origen', range(1, len(frame) + 1))
        library[name] = frame
    procedencia(name)
    if st.button('Restablecer copia desde el original', key=key + '_restablecer'):
        library.pop(name, None)
        st.session_state.pop('analisis', None)
        st.session_state.pop('preparacion_info', None)
        st.session_state['edicion_revision'] = st.session_state.get('edicion_revision', 0) + 1
        st.rerun()
    return name, library[name].copy()


@st.cache_data(show_spinner=False, max_entries=48)
def contenido_imagen(url, ruta):
    return base64.b64decode(ApiClient(url).request('GET', '/imagen', params={'nombre': ruta})['contenido'])


def elegir_muestras(api, key, limite):
    a, b = st.columns(2)
    conjunto = a.selectbox('Conjunto', ['val', 'train', 'test'], key=key + '_conjunto',
                          format_func=lambda v: {'val': 'Validación', 'train': 'Entrenamiento', 'test': 'Prueba'}[v])
    meta = api.request('GET', '/imagenes', params={'conjunto': conjunto, 'limite': 1})
    especie = b.selectbox('Especie del dataset', meta['clases'], key=key + '_especie')
    total = meta['conteos'][especie]
    pages = max(1, (total + limite - 1) // limite)
    page = st.number_input('Página de muestras', min_value=1, max_value=pages, value=1,
                           key=key + conjunto + especie + '_pagina')
    st.caption(f'{total:,} imágenes de {especie} · página {page} de {pages}')
    return api.request('GET', '/imagenes', params={'conjunto': conjunto, 'especie': especie, 'limite': limite, 'pagina': page - 1})['imagenes']


def selector_imagen(api, key):
    source = st.radio('Origen de la imagen', ['Muestra del proyecto', 'Subir imagen'], horizontal=True, key=key + '_origen')
    if source == 'Subir imagen':
        upload = st.file_uploader('Imagen JPG o PNG', type=['jpg', 'jpeg', 'png'], max_upload_size=20, key=key + '_imagen')
        if upload is None:
            return None
        st.image(upload, width='stretch')
        return {'nombre': upload.name, 'contenido': base64.b64encode(upload.getvalue()).decode()}
    images = elegir_muestras(api, key, 12)
    if not images:
        st.info('No hay imágenes en esta selección.')
        return None
    selected = st.selectbox('Imagen', images, format_func=lambda v: v['nombre'], key=key + '_muestra')
    st.image(contenido_imagen(api.url, selected['ruta']), width='stretch')
    st.caption('La etiqueta del dataset no se envía como resultado: la CNN clasificará la imagen.')
    return {'ruta': selected['ruta']}


def galeria_imagenes(api):
    images = elegir_muestras(api, 'galeria', 9)
    columns = st.columns(3)
    for i, sample in enumerate(images):
        with columns[i % 3]:
            st.image(contenido_imagen(api.url, sample['ruta']), caption=sample['nombre'], width='stretch')

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
