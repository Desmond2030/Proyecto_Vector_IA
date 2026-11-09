# ESTADO_ARCHIVO: EN_MODIFICACION
import altair as alt
import pandas as pd
import streamlit as st
from app.client import ErrorAPI
from app.datasets import selector_dataset, editor_dataset, filas
from app.results import descargar, matriz_interactiva
from app.ui import cabecera, api_activa, COLORES

cabecera('explorar', '02', 'Los datos cuentan la historia.', 'Explora distribuciones, compara variables y prepara una copia de tus datos.')
api = api_activa()
name, frame = selector_dataset(api, 'explorar')
if frame is not None:
    a, b, c = st.columns(3)
    a.metric('Registros', f'{len(frame):,}')
    b.metric('Variables', len(frame.columns) - int('_fila_origen' in frame))
    c.metric('Valores faltantes', int(frame.isna().sum().sum()))
    with st.expander('Filtrar la vista'):
        filter_col = st.selectbox('Filtrar por variable', [c for c in frame if c != '_fila_origen'], index=None)
        if filter_col:
            if pd.api.types.is_numeric_dtype(frame[filter_col]):
                if frame[filter_col].dropna().empty:
                    st.info('Esta variable solo contiene valores faltantes.')
                else:
                    lower, upper = st.columns(2)
                    low = lower.number_input('Desde', value=float(frame[filter_col].min()), key='min_' + filter_col)
                    high = upper.number_input('Hasta', value=float(frame[filter_col].max()), key='max_' + filter_col)
                    frame = frame.loc[frame[filter_col].between(low, high)]
            else:
                selected = st.multiselect('Valores', frame[filter_col].dropna().unique(), default=frame[filter_col].dropna().unique())
                frame = frame.loc[frame[filter_col].isin(selected)]
    distributions, relations, table = st.tabs(['Distribución', 'Correlaciones y asociaciones', 'Tabla y limpieza'])
    with distributions:
        available = [c for c in frame if c != '_fila_origen']
        numeric = frame[available].select_dtypes('number').columns.tolist()
        if not available:
            st.info('El dataset necesita al menos una variable de análisis.')
            st.stop()
        controls, plot = st.columns([1, 3], gap='large')
        with controls:
            default_field = next((c for c in ['casos_dengue', 'casos_pred_4sem'] if c in available), available[0])
            field = st.selectbox('Variable', available, index=available.index(default_field))
            kind = st.selectbox('Gráfico', ['Barras', 'Histograma', 'Cajas', 'Dispersión'], index=1 if default_field in numeric else 0)
            group = st.selectbox('Agrupar por', available, index=None)
            other = st.selectbox('Segunda variable numérica', numeric, index=None) if kind == 'Dispersión' else None
        with plot:
            if frame.empty:
                st.info('No hay registros en la selección.')
            elif kind == 'Barras':
                if group and field in numeric:
                    data = frame.groupby(group, dropna=False)[field].sum().reset_index()
                    chart = alt.Chart(data).mark_bar(color=COLORES['explorar']).encode(x=alt.X(f'{group}:N', title=group), y=alt.Y(f'{field}:Q', title=f'Suma de {field}'), tooltip=[group, field])
                else:
                    data = frame[field].value_counts(dropna=False).rename_axis(field).reset_index(name='Cantidad')
                    chart = alt.Chart(data).mark_bar(color=COLORES['explorar']).encode(x=alt.X(f'{field}:N', sort='-y'), y='Cantidad:Q', tooltip=[field, 'Cantidad'])
                st.altair_chart(chart.properties(height=340), width='stretch')
            elif field not in numeric:
                st.info('Este gráfico necesita una variable numérica.')
            elif kind == 'Histograma':
                chart = alt.Chart(frame).mark_bar(color=COLORES['explorar']).encode(x=alt.X(f'{field}:Q', bin=True), y=alt.Y('count():Q', title='Frecuencia'), tooltip=['count():Q'])
                st.altair_chart(chart.properties(height=340), width='stretch')
            elif kind == 'Cajas':
                chart = alt.Chart(frame).mark_boxplot(color=COLORES['explorar']).encode(y=alt.Y(f'{field}:Q'))
                if group:
                    chart = chart.encode(x=alt.X(f'{group}:N'))
                st.altair_chart(chart.properties(height=340), width='stretch')
            elif other:
                chart = alt.Chart(frame).mark_circle(color=COLORES['explorar'], opacity=.55).encode(x=f'{field}:Q', y=f'{other}:Q', tooltip=[field, other])
                st.altair_chart(chart.properties(height=340).interactive(), width='stretch')
            else:
                st.info('Selecciona una segunda variable.')
    with relations:
        st.caption('Numéricas, categóricas y dummies: elige el tratamiento según el significado de tus variables.')
        selected = st.multiselect('Variables para el análisis', [c for c in frame if c != '_fila_origen'], default=[c for c in frame if c != '_fila_origen'])
        cats = st.multiselect('Variables categóricas (incluye códigos numéricos si representan categorías)', selected,
                              default=[c for c in selected if not pd.api.types.is_numeric_dtype(frame[c])])
        a, b = st.columns(2)
        mode = a.selectbox('Modalidad', ['mixtas', 'numericas', 'dummies'], format_func=lambda v: {'mixtas': 'Asociaciones mixtas', 'numericas': 'Solo numéricas', 'dummies': 'Numéricas + dummies'}[v])
        method = b.selectbox('Método numérico', ['pearson', 'spearman'], format_func=str.capitalize)
        st.caption('Mixtas: Pearson/Spearman, V de Cramér y η. Las medidas no tienen todas signo ni la misma interpretación; una asociación no prueba causalidad.')
        if st.button('Calcular asociaciones', type='primary', disabled=len(selected) < 2 or frame.empty):
            try:
                with st.spinner('Calculando sobre la selección…'):
                    result = api.request('POST', '/datasets/correlaciones', {'filas': filas(frame), 'columnas': selected, 'categoricas': cats, 'modalidad': mode, 'metodo': method})
                st.session_state['analisis'] = {'resultado': result, 'fuente': name, 'modalidad': mode, 'metodo': method}
            except ErrorAPI as exc:
                st.error(str(exc))
        saved = st.session_state.get('analisis')
        if saved:
            result = saved['resultado']
            st.caption(f"Último cálculo: {saved['fuente']} · {saved['modalidad']} · {saved['metodo']}. Recalcula después de cambiar datos o filtros.")
            st.write(f"{result['filas_utilizadas']} filas utilizadas · {result['filas_excluidas']} excluidas por nulos")
            if result['constantes']:
                st.info('No definido para variables constantes: ' + ', '.join(result['constantes']))
            if result['dummies']:
                st.dataframe(result['dummies'], hide_index=True, width='stretch')
            matriz_interactiva(result['celdas'])
    with table:
        editor_dataset(api, name, frame)
        descargar(frame, 'Descargar selección', 'datos_preparados.csv')
