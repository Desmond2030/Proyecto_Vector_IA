# ESTADO_ARCHIVO: EN_MODIFICACION
import html
import json
import streamlit as st
from app.client import ApiClient
from app.ui import APP, estilo, resumen_evaluacion

estilo('inicio')
project = json.loads((APP / 'proyecto.json').read_text(encoding='utf-8'))
left, right = st.columns([.95, 1.25], gap='large')
with left:
    st.html('<div class="eyebrow">01 / UN PROYECTO, TRES PERSPECTIVAS</div><div class="hero-title">Observar.<br>Anticipar.<br>Actuar.</div>')
    st.html(f'<p class="hero-copy">{html.escape(project["descripcion"])}</p>')
    for n, text in [('01', 'Identificar especies · CNN'), ('02', 'Pronosticar casos · SimpleRNN / GRU'), ('03', 'Recomendar acciones · ANN multiclase')]:
        st.html(f'<div class="chapter"><b>{n}</b><span>{text}</span></div>')
with right:
    st.image(str(APP / 'assets' / 'portada.png'), width='stretch')
    st.caption('VectorIA / Costa Rica / 2026 · Ilustración conceptual')
for col, key, title in zip(st.columns(4), ['estudiante', 'institucion', 'curso', 'profesor'], ['Estudiante', 'Institución', 'Curso', 'Profesor']):
    with col:
        st.html(f'<div class="credits"><small>{title}</small><p>{html.escape(project[key])}</p></div>')
st.divider()
st.subheader('Resultados de evaluación')
resumen_evaluacion(ApiClient())
st.page_link('pages/3_Evaluacion.py', label='Ver evaluación', icon=':material/arrow_forward:')
