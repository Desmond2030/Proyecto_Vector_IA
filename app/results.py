# ESTADO_ARCHIVO: EN_MODIFICACION
import json
import altair as alt
import pandas as pd
import streamlit as st
from app.ui import COLORES


def descargar(frame, label='Descargar CSV', nombre='vectoria.csv', key=None):
    st.download_button(label, frame.to_csv(index=False).encode('utf-8-sig'), nombre, 'text/csv', key=key)


def matriz_interactiva(cells):
    frame = pd.DataFrame(cells)
    if frame.empty:
        st.info('No hay columnas suficientes para una matriz.')
        return
    frame['detalle'] = frame['valor'].map(lambda v: 'No definido' if pd.isna(v) else f'{v:.3f}')
    base = alt.Chart(frame).encode(x=alt.X('variable_2:N', title=None, sort=None, axis=alt.Axis(labelLimit=240, labelAngle=-45)),
                                  y=alt.Y('variable_1:N', title=None, sort=None, axis=alt.Axis(labelLimit=240)),
                                  tooltip=['variable_1', 'variable_2', 'metodo', 'detalle'])
    color = alt.condition('isValid(datum.valor)', alt.Color('valor:Q', scale=alt.Scale(domain=[-1, 0, 1], range=['#386c9b', '#f3f1e7', '#b86a16']), title='Coeficiente'), alt.value('#d9ddd7'))
    chart = base.mark_rect(stroke='#f7f6f2', strokeWidth=1).encode(color=color)
    if frame.variable_1.nunique() <= 12:
        chart += base.mark_text(fontSize=11).encode(text=alt.Text('valor:Q', format='.2f'), color=alt.value('#232827'))
    st.altair_chart(chart.properties(height=max(260, frame.variable_1.nunique() * 27)), width='stretch')
    st.dataframe(frame, hide_index=True, width='stretch', height=230)
    descargar(frame, 'Descargar coeficientes', 'correlaciones.csv')


def mostrar_resultado(result):
    cnn, rnn, ann = result.get('cnn'), result.get('rnn'), result.get('ann')
    if cnn:
        st.subheader(cnn['especie'].replace('_', ' ').capitalize())
        st.caption(f"aegypti_detectado = {cnn['aegypti_detectado']}")
    if rnn:
        st.markdown('**Pronóstico de casos**')
        frame = pd.DataFrame(rnn['semanas'])
        chart = alt.Chart(frame).mark_line(point=True, color=COLORES['prediccion']).encode(
            x=alt.X('fecha:T', title='Semana'), y=alt.Y('casos:Q', title='Casos de dengue', scale=alt.Scale(zero=False)), tooltip=['fecha', 'casos'])
        st.altair_chart(chart.properties(height=190), width='stretch')
        st.metric('Total del horizonte', f"{rnn['total']:,.1f} casos")
    if ann:
        st.markdown('**Acción sugerida por ANN**')
        st.success(ann['accion'].replace('_', ' ').capitalize())
    for slot, error in result.get('errores', {}).items():
        st.info(f'{slot.upper()}: {error}')
    if result.get('artefactos'):
        with st.expander('Modelos y datos utilizados'):
            st.json({'modelos': result['artefactos'], 'entradas': result.get('entrada_modelos', {})})


def tabla_predicciones(results):
    rows = []
    for result in results:
        row = dict(result['entrada'])
        additions = {'fila_resultado': result['fila'], 'especie_predicha': (result['cnn'] or {}).get('especie'),
                     'deteccion_predicha': (result['cnn'] or {}).get('aegypti_detectado'),
                     'casos_pronosticados': (result['rnn'] or {}).get('total'),
                     'accion_recomendada': (result['ann'] or {}).get('accion'),
                     'errores': json.dumps(result['errores'], ensure_ascii=False)}
        additions.update({f'modelo_{k}': v for k, v in result['artefactos'].items()})
        for i, point in enumerate((result.get('rnn') or {}).get('semanas', []), 1):
            additions[f'casos_h{i}'] = point['casos']
            additions[f'fecha_h{i}'] = point['fecha']
        for key, value in additions.items():
            while key in row:
                key += '_resultado'
            row[key] = value
        rows.append(row)
    return pd.DataFrame(rows)


def mostrar_evaluacion(result):
    st.caption(f"{result['conjunto']} · {result['origen']} · {result['archivo_modelo']}")
    metrics = pd.DataFrame([{'metrica': k, 'valor': v} for k, v in result['metricas'].items()])
    st.dataframe(metrics, hide_index=True, width='stretch')
    if result.get('matriz'):
        matrix = pd.DataFrame(result['matriz'])
        real = matrix.columns[0]
        melted = matrix.melt(id_vars=[real], var_name='Predicha', value_name='Cantidad').rename(columns={real: 'Real'})
        chart = alt.Chart(melted).mark_rect().encode(x='Predicha:N', y='Real:N', color=alt.Color('Cantidad:Q', scale=alt.Scale(scheme='purples')), tooltip=['Real', 'Predicha', 'Cantidad'])
        st.altair_chart(chart.properties(height=290), width='stretch')
    if result.get('por_clase'):
        st.dataframe(pd.DataFrame(result['por_clase']), hide_index=True, width='stretch')
    if result.get('predicciones'):
        points = pd.DataFrame(result['predicciones'])
        if {'real', 'prediccion'}.issubset(points):
            chart = alt.Chart(points).mark_circle(color=COLORES['evaluacion']).encode(x='real:Q', y='prediccion:Q', tooltip=list(points.columns))
            st.altair_chart(chart.properties(height=240), width='stretch')
    for key in ['modelo', 'conjunto']:
        metrics[key] = result[key]
    descargar(metrics, 'Descargar métricas CSV', 'metricas.csv', key='metricas_' + result['id'])
