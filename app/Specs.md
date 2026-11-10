<!-- ESTADO_ARCHIVO: EN_MODIFICACION -->
# App VectorIA

Ejecutar desde la raíz WSL: .Vector_IA_wsl/bin/python -B app/run.py.
App en http://127.0.0.1:8501 y API en http://127.0.0.1:8000/docs.
También se puede iniciar Streamlit por separado y configurar VECTORIA_API_URL.

## Cuatro páginas

- Inicio: presentación, datos académicos de proyecto.json y métricas vigentes consultadas a la API.
- Exploración: Parquet procesados, filtros, tablas, barras, histogramas, cajas, dispersión, correlaciones Pearson/Spearman, dummies y asociaciones mixtas; galería de imágenes por conjunto y especie.
- Predicción: selección/carga de modelos y escaladores, serie del modelo RNN, imagen del proyecto o cargada, fecha, ubicación y cinco campos manuales de clima/umbral. Un botón ejecuta el flujo y presenta los tres resultados juntos.
- Evaluación: cálculo por modelo e historial consultable. La última evaluación del modelo activo actualiza Inicio automáticamente.

Paleta: verde, ámbar, azul y violeta, respectivamente. Tipografía y espaciado comunes; formularios arriba y resultados con el ancho disponible.

La navegación utiliza cuatro enlaces nativos de Streamlit dentro del encabezado de la app: una fila en escritorio y dos columnas en pantallas pequeñas. La página activa se destaca con su color; los enlaces conservan contraste propio independientemente del tema.

## Datos y modelos

Las copias editables de Exploración viven en la sesión, pueden descargarse y restablecerse. No modifican el historial que utiliza Predicción, que siempre procede del Parquet original procesado leído por la API.

Los selectores de modelos y escaladores se aplican explícitamente. La configuración técnica está plegada. Escaladores sin extensión pueden cargarse; deben ser los originales del modelo correspondiente. La ANN permanece pendiente si falta el suyo, conservando los resultados disponibles de CNN y RNN.

Los campos manuales son umbral endémico de cuatro semanas, lluvia de 14 días, temperatura máxima, temperatura mínima y humedad. La detección CNN y la suma de las cuatro semanas RNN se incorporan automáticamente. No se sustituyen campos vacíos por cero. Al cambiar entradas o configuración debe ejecutarse de nuevo la predicción.

## Métricas

Inicio presenta accuracy y F1 macro para clasificación y MAE/RMSE en casos para pronóstico. Sin evaluación correspondiente se muestra un estado vacío. Los valores no están escritos en la interfaz: proceden de api/evaluaciones.json y sobreviven al reinicio.

Evaluación informa muestra y partición utilizada. Consultar una evaluación antigua no cambia el resumen del Inicio. No hay tests ni entrenamiento desde la app.
