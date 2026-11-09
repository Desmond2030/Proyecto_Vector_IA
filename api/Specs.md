<!-- ESTADO_ARCHIVO: EN_MODIFICACION -->
# API VectorIA

Servicio local y de un solo usuario/worker. Desde la raíz WSL:

```bash
.Vector_IA_wsl/bin/python -B -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Documentación interactiva: http://127.0.0.1:8000/docs. `python app/run.py` inicia también Streamlit.

## Responsabilidades

- `loaders.py`: reutiliza `TabularDataProcessor` y `TabularDataLoader` para cargar datos y escaladores. Datos originales de `data/` y artefactos de `models/` son de solo lectura.
- `datasets.py`: transforma con las clases de `src/`; coordina Pearson/Spearman, dummies y asociaciones mixtas Dython. No ajusta escaladores.
- `model_adapter.py`: reutiliza `CNN_Model_Clasification` para imágenes y clasificación. El único ajuste de imagen adicional es dividir entre 255 cuando el perfil entrenado lo requiere y el modelo no incluye Rescaling.
- `predict.py`: coordina las tres redes y conserva errores por registro. ANN/RNN usan un adaptador de inferencia Keras con entradas, clases y parámetros temporales configurables.
- `evaluation.py`: usa `CNN_Model_Clasification.evaluar(mostrar=False)` para CNN; calcula métricas ANN/RNN donde no existe implementación compatible en src.

## Rutas

| Método y ruta | Entrada / resultado |
|---|---|
| GET `/health` | Estado y disponibilidad independiente de cnn/rnn/ann. |
| GET `/artefactos` | Rutas relativas disponibles de modelos, scalers y datos. |
| GET/PUT `/configuracion` | Perfiles por modelo; PUT valida todo antes de reemplazar la configuración activa. |
| GET `/datos` | Inventario; `?nombre=data/raw/archivo.csv` devuelve tabla. |
| POST `/files/{tipo}?nombre=...` | Cuerpo binario. Tipo: datos/modelo/scaler. Datos hasta 20 MB; artefactos hasta 500 MB. |
| POST `/datasets/prepare` | filas, renombrar, tipos, mapas, dummies, eliminar_nulos. |
| POST `/datasets/correlaciones` | filas, columnas, categoricas, modalidad (numericas/dummies/mixtas), metodo (pearson/spearman). |
| POST `/predict/{modo}` | mosquito/casos/accion/flujo. registros, imagenes, historial, columna_imagen, revision. |
| GET `/evaluaciones` | Historial y selección vigente por identidad del artefacto. |
| POST `/evaluaciones/importar` | filas con modelo, conjunto, metrica, valor y revision de configuración. |
| POST `/evaluaciones/seleccionar` | modelo e id de una evaluación del artefacto activo. |
| POST `/evaluate` | Contrato de inferencia más modelo, conjunto y objetivo. |

Modelos subidos: `api/archivos/`. Configuración: `api/configuracion.json`. Resultados: `api/evaluaciones.json`. Temporales de imágenes/datos: `api/.tmp/`, eliminados al terminar la petición. No se escribe en `models/`, `data/`, `src/` o notebooks. Solo subir artefactos propios de confianza, especialmente pickle/joblib.

## Contratos de modelos

Cada perfil configura archivo, columnas ordenadas, clases ordenadas y scaler opcional. Los perfiles iniciales no seleccionan archivos. Las clases iniciales ANN son una propuesta: confirmar el orden entrenado antes de usar un modelo.

- CNN: un tensor de entrada con dimensiones fijas `(N, alto, ancho, canales)` y salida `(N, clases)`. RGB o gris. El preprocesamiento utiliza la clase del proyecto y respeta la normalización indicada. La predicción individual devuelve la etiqueta que expone `img_prediction`, sin repetir inferencia para obtener probabilidades; la evaluación sí devuelve las salidas de `evaluar`.
- ANN: `(N, variables)` → `(N, clases)`. Entradas numéricas según el contrato. La acción es el argmax; no se sustituye la ANN por reglas.
- RNN: `(N, ventana, variables)` → `(N, horizonte)` o `(N, horizonte, 1)`. Inicialmente 26 semanas de entrada y 4 futuras. Scaler de salida con 1 variable o una por horizonte. Se exigen fechas semanales consecutivas y un único grupo por ventana.

Los modelos deben utilizar capas Keras disponibles; no se ejecuta código personalizado subido. La identidad de cada perfil incluye configuración y contenido de modelo/scalers. Un cambio de artefacto no hereda las métricas del anterior.

## Imágenes y flujo

`imagenes` es una lista de `{nombre, contenido}`, con contenido base64, JPG/PNG (hasta 100 imágenes, 20 MB cada una, 100 MB total decodificado). La columna `imagen`, configurable, identifica el nombre exacto. Cada fila conserva su entrada y posición en el resultado. No se empareja por orden de archivos.

CNN → `aegypti_detectado`; RNN → suma de cuatro semanas como `casos_pred_4sem`; ambas salidas y contexto → ANN. El flujo no utiliza valores antiguos de detección/pronóstico para ocultar una etapa fallida. Cada etapa fallida queda explicada; las anteriores se conservan. Las operaciones individuales también están disponibles.

## Asociaciones

Se trabaja sobre copias y se excluyen filas incompletas en las variables seleccionadas, mostrando el conteo. Las categóricas numéricas deben marcarse explícitamente. Dummies usa `get_dummies()` existente: omite la última categoría observada y se informa la referencia. Las dummies de exploración no se aplican automáticamente a inferencia.

Mixtas usa Dython con Pearson/Spearman, V de Cramér corregida y eta. Variables constantes y valores no finitos se muestran como no definidos. Se devuelve una tabla por pareja con método y coeficiente, para renderizar con Altair.

## Evaluación y límites

CNN/ANN requieren clases reales por nombre. RNN requiere las observaciones de referencia y el historial que contiene los valores reales del horizonte; la ventana de entrada solo utiliza fechas hasta la referencia. No hay entrenamiento ni división automática de conjuntos. Los CSV importados se marcan como importados, no como resultados recalculados.
