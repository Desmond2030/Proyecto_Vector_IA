# ESTADO_ARCHIVO: EN_MODIFICACION
import html
from pathlib import Path
import streamlit as st
from app.client import ApiClient, ErrorAPI

APP = Path(__file__).resolve().parent
COLORES = {'inicio': '#087F72', 'explorar': '#B86A16', 'prediccion': '#315BC7',
           'lotes': '#C45545', 'evaluacion': '#7952A8', 'modelos': '#526477'}
AVISO = 'Herramienta de apoyo a la vigilancia; no diagnostica dengue en personas.'


def estilo(pagina):
    st.html(APP / 'assets' / 'styles.css')
    st.html(f'<style>:root {{ --accent: {COLORES[pagina]}; }}</style>')


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
        evaluations = {}
    for column, slot, name in zip(st.columns(3), ['cnn', 'rnn', 'ann'], ['Identificación · CNN', 'Pronóstico · RNN', 'Acciones · ANN']):
        with column:
            st.markdown(f'**{name}**')
            result = evaluations.get(slot)
            if not result:
                st.caption('— Sin evaluación cargada')
            else:
                preferred = ['mae', 'rmse'] if slot == 'rnn' else ['accuracy', 'f1_macro']
                metrics = result['metricas']
                keys = [k for k in preferred if k in metrics] or list(metrics)[:2]
                st.write(' · '.join(f'{k}: {metrics[k]:.4g}' for k in keys))
                st.caption(f"{result['conjunto']} · {result['origen']}")
