<!-- ESTADO_ARCHIVO: EN_MODIFICACION -->
# API VectorIA

FastAPI ejecuta las clases del proyecto; Streamlit solo envía entradas y muestra resultados.
Ejecutar desde la raíz con .Vector_IA_wsl/bin/python -B app/run.py o usar Uvicorn con un worker.

## Rutas

- GET /health: estado independiente de CNN, RNN y ANN.
- GET /artefactos: modelos, escaladores (también sin extensión) y Parquet procesados.
- GET/PUT /configuracion: selección de cada modelo, escaladores y serie RNN. Devuelve revision, identidades y errores por modelo. Una dependencia ausente deja solo ese modelo pendiente.
- GET /datos: inventario; con nombre devuelve columnas y filas, incluyendo la fecha del índice.
- GET /imagenes: conjunto (train/val/test), especie, limite y pagina; devuelve rutas y etiquetas de las muestras.
- GET /imagen: nombre con ruta del inventario; devuelve contenido base64 para visualizar.
- POST /files/{modelo|scaler}?nombre=archivo: cuerpo binario, hasta 500 MB. Guardado solo en api/archivos.
- POST /datasets/prepare y /datasets/correlaciones: copias de exploración. Pearson, Spearman, dummies y asociaciones mixtas (V de Cramér y eta para las parejas categóricas/mixtas).
- POST /predict/flujo: una observación con imagen, fecha, ubicacion, contexto y revision.
- POST /evaluate: modelo, revision; CNN usa conjunto val/test y limite por especie; RNN usa ubicacion y limite de ventanas; ANN utiliza la validación temporal de acciones.
- GET /evaluaciones: historial y última evaluación calculada de cada artefacto activo.

## Flujo individual

Imagen: un objeto con ruta del inventario, o nombre y contenido base64 para una imagen cargada.
Contexto: casos_umbral_endemico, lluvia_14d_mm, temp_max, temp_min, humedad.

La API ejecuta CNN → RNN → ANN. CNN aporta aegypti_detectado. RNN toma de la serie configurada las 26 semanas hasta la fecha y ubicación elegidas, respeta las 29 columnas entrenadas, desescala los cuatro resultados y aporta su suma como casos_pred_4sem. ANN recibe esas dos salidas y los cinco campos manuales, en sus siete columnas entrenadas.

La respuesta contiene cnn, rnn, ann, errores, artefactos y entrada_modelos. Una etapa fallida no borra resultados anteriores; RNN puede completar su salida aunque CNN falle, pero ANN requiere ambas. No hay predicción por lotes ni resultados de relleno.

## Clases y artefactos

CNN reutiliza img_prediction y evaluar. ANN reutiliza Modelo_ANN.make_prediction para inferencia y evaluación. SimpleRNN/GRU reutilizan su modelo cargado y sus métodos generar_ventanas_dengue y evaluar. No se entrena ni se ajustan escaladores desde la API.

Las entradas RNN son población, cuatro variables climáticas y sus seis rezagos. Casos es el objetivo, no una entrada adicional. Cada ventana corresponde a una única ubicación. La asociación del modelo, su serie y escaladores originales es explícita; comprobar dimensiones no demuestra la procedencia del entrenamiento.

ANN requiere su escalador original de siete variables. No se reconstruye automáticamente. El scaler.pkl heredado de 20 variables queda excluido.

Los originales se leen de data/proceced/archivos_parquet y data/raw/Imagenes. Los archivos subidos, configuración, evaluaciones y temporales permanecen dentro de api/. No se modifica src, models ni data.

## Evaluación

CNN compara etiquetas de carpetas con la muestra seleccionada. ANN conserva el último 20% de fechas y sus escenarios juntos. RNN genera ventanas de 26+4 semanas íntegramente dentro del último 20% de fechas de la ubicación, y evalúa las últimas ventanas solicitadas. No se ajusta ningún escalador.

Accuracy/F1 macro para CNN y ANN; MAE/RMSE globales y por horizonte para RNN. Las evaluaciones guardan fecha, conjunto, tamaño e identidad (configuración y contenido del modelo y escaladores). Inicio muestra solo la última calculada que corresponda a esa identidad. El historial permanece consultable; no hay importación de métricas ni selección manual del resumen.
