# ESTADO_ARCHIVO: EN_MODIFICACION
import html
from pathlib import Path
import streamlit as st
from app.client import ApiClient, ErrorAPI

APP = Path(__file__).resolve().parent
COLORES = {'inicio': '#087F72', 'explorar': '#9B570A', 'prediccion': '#315BC7',
           'evaluacion': '#7952A8'}
AVISO = 'Herramienta de apoyo a la vigilancia; no diagnostica dengue en personas.'


def estilo(pagina):
    st.html(APP / 'assets' / 'styles.css')
    st.html(f'<style>:root {{ --accent: {COLORES[pagina]}; }}</style>')



def navegacion(pages, actual):
    """Enlaces nativos dentro del diseño, sin depender de la barra superior."""
    icons = [':material/home:', ':material/analytics:', ':material/online_prediction:', ':material/assessment:']
    with st.container(key='vectoria_navigation'):
        for index, (column, page, icon) in enumerate(zip(st.columns(4), pages, icons)):
            with column:
                with st.container(key=f'vectoria_nav_{index}'):
                    st.page_link(page, label=page.title, icon=icon, width='stretch')
            if page.url_path == actual.url_path:
                st.html(f'<style>.st-key-vectoria_nav_{index} [data-testid="stPageLink-NavLink"] {{'
                        'background: var(--accent) !important; color: #fff !important; '
                        'border-color: var(--accent) !important;}</style>')


def cabecera(pagina, numero, titulo, descripcion):
    estilo(pagina)
    st.html(f'<section class="page-heading"><div class="eyebrow">{numero} / {html.escape(pagina.upper())}</div>'
            f'<h1>{html.escape(titulo)}</h1><p>{html.escape(descripcion)}</p></section>')


def api_activa():
    api = ApiClient()
    try:
        api.salud()
    except ErrorAPI as exc:
        st.info(str(exc))
        st.code('.Vector_IA_wsl/bin/python app/run.py', language='bash')
        st.stop()
    return api


def mensaje_error(exc):
    st.error(str(exc))


def resumen_evaluacion(api):
    try:
        evaluations = api.request('GET', '/evaluaciones')['vigentes']
    except ErrorAPI:
        st.info('Conecta la API para consultar las métricas guardadas.')
        return
    for column, slot, name in zip(st.columns(3), ['cnn', 'rnn', 'ann'],
                                  ['Identificación · CNN', 'Pronóstico · RNN', 'Acciones · ANN']):
        with column:
            with st.container(border=True):
                st.markdown(f'**{name}**')
                result = evaluations.get(slot)
                if not result:
                    st.caption('Sin evaluación calculada para el modelo activo.')
                    continue
                values = result['metricas']
                labels = [('mae', 'MAE · casos'), ('rmse', 'RMSE · casos')] if slot == 'rnn' else [('accuracy', 'Accuracy'), ('f1_macro', 'F1 macro')]
                for metric, label in labels:
                    number = values.get(metric)
                    st.metric(label, '—' if number is None else (f'{number:,.2f}' if slot == 'rnn' else f'{number:.2%}'))
                st.caption(result['archivo_modelo'].split('/')[-1])
                st.caption(f"{result['fecha'][:19]} UTC · {result['conjunto']}")
