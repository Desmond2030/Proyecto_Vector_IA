# ESTADO_ARCHIVO: EN_MODIFICACION
"""Entrada: streamlit run app/Home.py, desde la raíz del proyecto."""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import streamlit as st
import altair as alt
from app.client import ApiClient, ErrorAPI
from app.ui import estilo, AVISO

st.set_page_config(page_title='VectorIA · Vigilancia del dengue', page_icon='◈', layout='wide')
alt.data_transformers.disable_max_rows()
estilo('inicio')
pages = [st.Page('pages/0_Inicio.py', title='Inicio', default=True),
         st.Page('pages/4_Visualizaciones.py', title='Explorar'),
         st.Page('pages/1_Prediccion.py', title='Predicción'),
         st.Page('pages/2_Analisis.py', title='Lotes'),
         st.Page('pages/3_Metricas.py', title='Evaluación'),
         st.Page('pages/5_Modelos.py', title='Modelos')]
page = st.navigation(pages, position='top')
left, right = st.columns([4, 1])
with left:
    st.html('<div class="masthead"><div class="wordmark">VectorIA</div><span>Costa Rica / Inteligencia artificial aplicada / 2026</span></div>')
with right:
    try:
        health = ApiClient().salud()
        st.caption('● API conectada · ' + str(sum(health['modelos'].values())) + '/3 modelos')
    except ErrorAPI:
        st.caption('○ API sin conexión')
page.run()
st.html(f'<footer class="footer">{AVISO}</footer>')
