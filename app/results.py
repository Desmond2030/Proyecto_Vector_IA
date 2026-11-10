# ESTADO_ARCHIVO: EN_MODIFICACION
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
    titles = ['01 · Identificación CNN', '02 · Pronóstico RNN', '03 · Recomendación ANN']
    for column, slot, title in zip(st.columns(3), ['cnn', 'rnn', 'ann'], titles):
        with column:
            with st.container(border=True):
                st.markdown(f'**{title}**')
                value = result.get(slot)
                if value is None:
                    st.info(result.get('errores', {}).get(slot, 'Sin resultado disponible.'))
                elif slot == 'cnn':
                    st.subheader(value['especie'].replace('_', ' ').capitalize())
                    st.caption('Aegypti detectado: ' + ('Sí' if value['aegypti_detectado'] else 'No'))
                elif slot == 'rnn':
                    st.metric('Total de las próximas 4 semanas', f"{value['total']:,.1f}")
                    st.caption('Casos previstos · suma sin redondear para la entrada ANN')
                    frame = pd.DataFrame(value['semanas']).rename(columns={'fecha': 'Semana', 'casos': 'Casos previstos'})
                    st.dataframe(frame, hide_index=True, width='stretch',
                                 column_config={'Casos previstos': st.column_config.NumberColumn(format='%.2f')})
                else:
                    st.subheader(value['accion'].replace('_', ' ').capitalize())
                    st.caption('Calculada con la detección, el pronóstico y el contexto ingresado.')
    if rnn:
        frame = pd.DataFrame(rnn['semanas'])
        chart = alt.Chart(frame).mark_line(point=True, color=COLORES['prediccion']).encode(
            x=alt.X('fecha:T', title='Semana futura', axis=alt.Axis(format='%d/%m/%Y')),
            y=alt.Y('casos:Q', title='Casos previstos', scale=alt.Scale(zero=False)),
            tooltip=['fecha', alt.Tooltip('casos:Q', format='.2f')])
        st.altair_chart(chart.properties(height=240), width='stretch')
    summary = {'Fecha de referencia': result['fecha'], 'Ubicación': result['ubicacion'],
               'Especie': cnn['especie'] if cnn else 'Pendiente',
               'Detección aegypti': cnn['aegypti_detectado'] if cnn else None,
               'Casos en 4 semanas': rnn['total'] if rnn else None,
               'Recomendación': ann['accion'] if ann else 'Pendiente'}
    if rnn:
        summary.update({f'Semana {i + 1}': point['casos'] for i, point in enumerate(rnn['semanas'])})
    with st.expander('Resumen conjunto y entradas utilizadas'):
        st.dataframe(pd.DataFrame([summary]), hide_index=True, width='stretch')
        st.dataframe(pd.DataFrame([result['entrada_modelos']]), hide_index=True, width='stretch')
        for slot, path in result['artefactos'].items():
            st.caption(f'{slot.upper()}: {path}')
        if ann and ann.get('probabilidades'):
            st.dataframe(pd.DataFrame([ann['probabilidades']]), hide_index=True, width='stretch')
        descargar(pd.DataFrame([summary]), 'Descargar resultado', 'prediccion_vectoria.csv')


def mostrar_evaluacion(result, key='evaluacion'):
    st.caption(f"{result['conjunto']} · {result['fecha'][:19]} UTC · {result.get('muestras', '—')} muestras")
    st.caption(result['archivo_modelo'])
    metrics = pd.DataFrame([{'Métrica': k, 'Valor': v} for k, v in result['metricas'].items()])
    st.dataframe(metrics, hide_index=True, width='stretch')
    if result.get('matriz'):
        matrix = pd.DataFrame(result['matriz'])
        real = matrix.columns[0]
        melted = matrix.melt(id_vars=[real], var_name='Predicha', value_name='Cantidad').rename(columns={real: 'Real'})
        chart = alt.Chart(melted).mark_rect().encode(
            x=alt.X('Predicha:N', sort=None), y=alt.Y('Real:N', sort=None),
            color=alt.Color('Cantidad:Q', scale=alt.Scale(scheme='purples')), tooltip=['Real', 'Predicha', 'Cantidad'])
        st.altair_chart(chart.properties(height=290), width='stretch', key=key + result['id'] + '_matriz')
    if result.get('por_clase'):
        st.dataframe(pd.DataFrame(result['por_clase']), hide_index=True, width='stretch')
    if result.get('por_horizonte'):
        st.dataframe(pd.DataFrame(result['por_horizonte']), hide_index=True, width='stretch')
    if result.get('predicciones'):
        points = pd.DataFrame(result['predicciones'])
        if {'real', 'prediccion'}.issubset(points):
            chart = alt.Chart(points).mark_circle(color=COLORES['evaluacion']).encode(
                x=alt.X('real:Q', title='Casos observados'), y=alt.Y('prediccion:Q', title='Casos previstos'),
                tooltip=list(points.columns))
            st.altair_chart(chart.properties(height=240), width='stretch', key=key + result['id'] + '_puntos')
    descargar(metrics, 'Descargar métricas', 'metricas.csv', key=key + result['id'])
