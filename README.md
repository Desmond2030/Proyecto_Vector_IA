# Proyecto 2: VectorIA - Asistente Inteligente para la Vigilancia del Dengue en Costa Rica

**Curso:** BD-151 Inteligencia Artificial Aplicada - Colegio Universitario de Cartago
**Profesor:** Osvaldo González Chaves
**Año:** 2026

## Integrantes

| Nombre | Carné | Correo |
|---|---|---|
| Desmond Akin Bermúdez Rodríguez | 305270891 | 305270891@cuc.cr |

## Descripción del problema

VectorIA busca apoyar a las autoridades de salud y a las comunidades mediante tres funciones: identificar mosquitos en fotografías con una CNN, pronosticar los casos semanales de dengue de las próximas cuatro semanas con una RNN/LSTM y recomendar acciones de control vectorial con una ANN multiclase.

La aplicación integrará fotografía, semana seleccionada, clima, pronóstico y recomendación con su motivo, junto con un mapa de calor cantonal. Debe mostrar permanentemente: **«Herramienta de apoyo a la vigilancia; no diagnostica dengue en personas».**

El proyecto deriva de «Proyecto ANN en WSL» (`Proyecto_Abandono_Clientes`). Actualmente contiene EDA de imágenes, una CNN propia y artefactos de experimentos. La comparación con MobileNetV2, los módulos temporal y de acciones y la demo integrada están pendientes. La API y el dashboard existentes conservan lógica del proyecto predecesor.

Requisitos: **VectorIA.pdf**, proporcionado por el desarrollador, páginas 2-10, contrastado con `data/raw/LEEME_datos.md`. El PDF es una referencia externa, no un archivo incluido actualmente en el repositorio.

## Dataset

El paquete académico **VectorIA_datos.zip** aporta los CSV y el diccionario. Los cuatro CSV están presentes en `data/raw/`; las cantidades se comprobaron por lectura, sin modificar originales.

| Archivo | Naturaleza | Contenido y uso |
|---|---|---|
| `vectoria_dengue_nacional_real.csv` | Real | 574 semanas de casos y clima (2014-2024); RNN/LSTM. Casos de OPS/PAHO vía OpenDengue y clima de NASA POWER. |
| `vectoria_acciones.csv` | Mixto | 2.060 filas: 412 semanas por 5 escenarios (2017-2024); ANN. Casos, clima y canal endémico reales; predicción, detección y etiquetas simuladas. |
| `vectoria_casos_cantonal_anual_real.csv` | Real | 422 filas, 71 cantones (2013-2018); mapa de calor. Usar proporciones o rankings: la suma no coincide con el total nacional. |
| `vectoria_dengue_semanal.csv` | Sintético, opcional | 4.382 filas por provincia (2014-2025). No representa cifras oficiales y contiene faltantes deliberados. |

**Imágenes:** combinar las fuentes del enunciado y documentar procedencia y etiquetas: IEEE DataPort (Aedes/Culex, 1.404 imágenes según el PDF), Mendeley Vector Mosquito Images, Mosquito Alert 2023 y Kaggle (aegypti/albopictus, unas 4.120 imágenes según el PDF). Estas cifras describen fuentes propuestas, no el inventario local. MLMI-2024 (800 fotos de microscopio) es un bonus para larvas.

Las clases objetivo son **Aedes aegypti, Aedes albopictus y Culex**. Los notebooks actuales usan Aedes Aegypti, Anopheles Stephensi y Culex Quinquefasciatus: hay que adecuar el conjunto de entrenamiento sin renombrar Anopheles como albopictus ni cambiar etiquetas sin evidencia.

Conservar los originales en `data/raw/`. Registrar limpieza, duplicados, tamaños, canales, distribución de clases y procedencia; guardar derivados en `data/processed/` cuando se implemente el flujo. Consultar el diccionario antes de unir datasets o interpretar columnas.

## Modelos

- **Módulo 1 - CNN:** implementar una CNN propia desde cero y compararla con **MobileNetV2 mediante transfer learning**. Reportar F1 de ambas, con meta de **F1 macro >= 85 %**. Abordar el desbalance combinando datasets y usando pesos de clase. La arquitectura 128x128, Conv2D(32/64/128), MaxPool, GlobalAveragePooling, Dense(128), Dropout(0.5), Softmax(3) es una sugerencia del PDF, no una configuración actual obligatoria.
- **Módulo 2 - RNN/LSTM:** usar fechas, lluvia, temperaturas máxima y mínima, humedad y rezagos para pronosticar **las próximas 4 semanas**. Comparar **SimpleRNN** con **LSTM o GRU**, reportando RMSE de ambas y MAE, además del baseline estacional de la misma semana del año anterior. Preparar ventanas de **26 semanas**, aplicar MinMaxScaler ajustado solo con entrenamiento y dividir temporalmente los conjuntos. Se sugieren SimpleRNN(32) -> Dense(4) y LSTM/GRU(64/32) -> Dropout(0.2) -> Dense(4).
- **Módulo 3 - ANN:** clasificador de `monitoreo_normal`, `eliminacion_criaderos`, `fumigacion_focalizada` y `alerta_sanitaria`, con meta de **precision y recall >= 85 %**. Reportar métricas por clase y declarar el promedio utilizado; el PDF no especifica uno. Separar por `fecha_inicio_semana`, manteniendo juntos sus cinco escenarios, con particiones agrupadas o temporales, nunca con reparto aleatorio de filas.

Las reglas que originan las etiquetas ANN se aplican en este orden:

1. `casos_pred_4sem > 2 * casos_umbral_endemico`: `alerta_sanitaria`.
2. `casos_pred_4sem > casos_umbral_endemico` y `aegypti_detectado == 1`: `fumigacion_focalizada`.
3. `lluvia_14d_mm > 80` y `temp_max > 27`: `eliminacion_criaderos`.
4. En otro caso: `monitoreo_normal`.

Son reglas del dataset académico, no recomendaciones sanitarias validadas. El diccionario declara ruido aleatorio en el 3 % de las etiquetas. Los nombres reales del CSV (`casos_umbral_endemico`, `aegypti_detectado`) difieren de los abreviados del PDF; conservar un mapeo explícito al integrar modelos.

Lineamientos de entrenamiento:

- Separar entrenamiento, validación y prueba antes de ajustar transformaciones. Reservar prueba para evaluación final, sin elegir con ella arquitectura, hiperparámetros o umbrales.
- En imágenes, evitar duplicados o variantes del mismo origen entre conjuntos, normalizar según cada arquitectura y aumentar solo entrenamiento. Calcular class weights con entrenamiento.
- En series, evitar información futura en rezagos, escalado y objetivos, incluidos los límites entre particiones. Evaluar modelos y baseline sobre las mismas fechas y horizontes.
- En acciones, mantener todos los escenarios de una semana en una partición y distinguir entradas simuladas de futuras salidas reales de CNN/LSTM.
- Registrar semillas, orden de clases, hiperparámetros, callbacks, métricas y artefactos. El notebook CNN actual utiliza EarlyStopping y ModelCheckpoint.

## API REST

La API es **opcional**. Los endpoints siguientes pertenecen al alcance previsto del PDF; sus contratos aún deben implementarse para VectorIA.

| Método propuesto | Endpoint | Respuesta prevista |
|---|---|---|
| POST | `/predict/mosquito` | Especie y probabilidades |
| POST | `/predict/casos` | Pronóstico de cuatro semanas |
| POST | `/predict/accion` | Acción preventiva sugerida y motivo |

`api/` contiene una implementación heredada de abandono de clientes; no proporciona todavía estos servicios de VectorIA. FastAPI expone la documentación automática en `http://localhost:8000/docs` al iniciar el servidor; su contenido depende de la implementación disponible.

## Estructura del proyecto

Estructura resumida con componentes actuales y previstos, identificados expresamente. No se renombraron notebooks ni crearon módulos durante la actualización documental.

```text
Proyecto_VectorIA/
│
├── README.md                         ← Propósito, requisitos, estado y uso
├── AGENTS.md                         ← Instrucciones para agentes
├── requirements.txt                  ← Dependencias actuales
├── .gitignore
├── .Vector_IA_wsl/                    ← Entorno local WSL, excluido de Git
├── USO_DE_IA.md                       ← Registro acumulativo de intervenciones
│
├── data/
│   ├── raw/                          ← CSV e imágenes originales
│   │   ├── LEEME_datos.md
│   │   ├── vectoria_dengue_nacional_real.csv
│   │   ├── vectoria_acciones.csv
│   │   ├── vectoria_casos_cantonal_anual_real.csv
│   │   ├── vectoria_dengue_semanal.csv
│   │   └── Imagenes/
│   ├── proceced/                     ← Directorio existente con esta grafía
│   └── processed/                    ← Previsto; aún no existe
│
├── notebooks/
│   ├── 1_EDA.ipynb                   ← EDA actual, centrado en imágenes
│   ├── 2_Modelo_CNN.ipynb            ← CNN propia
│   ├── 03_LSTM_Casos.ipynb           ← Previsto
│   └── 04_ANN_Acciones.ipynb          ← Previsto
│
├── src/
│   ├── __init__.py
│   ├── Modelo_CNN.py                 ← Procesamiento visual y CNN
│   ├── Modelo_ANN.py                 ← Base ANN heredada
│   ├── Tabular_Data_Loader.py
│   ├── Tabular_Data_Processor.py
│   ├── Graficos.py
│   ├── data_prep.py                  ← Previsto por el enunciado
│   └── train/                       ← Previsto: cnn.py, lstm.py y ann.py
│
├── Proyect_Context/
│   ├── About_Folder.md               ← Reglas del historial
│   └── General_Context_File.md       ← Evolución y antecedentes
│
├── models/
│   ├── CNN_Models/Img_classify/      ← Modelos 1 y 2; checkpoints 1, 2 y 3
│   ├── configuracion_inferencia.json ← Configuración heredada de la API
│   └── scaler.pkl                   ← Artefacto heredado; verificar compatibilidad
│
├── api/                             ← Base heredada; adaptación opcional pendiente
│   ├── main.py
│   ├── schemas.py
│   └── predict.py
│
└── app/                             ← Base Streamlit; adaptación pendiente
    ├── Home.py
    ├── run.py
    ├── pages/                       ← Páginas heredadas
    └── assets/                      ← Previsto para recursos de la demo
```

El PDF propone `01_EDA.ipynb` y `02_CNN_Mosquitos.ipynb`; se conservan los nombres actuales hasta que el desarrollador decida reorganizarlos. El árbol no enumera todos los auxiliares existentes de API y app.

## Instalación

Entorno principal: **WSL2, Ubuntu-24.04**, raíz `/home/hadoop/CUC/IA_Aplicada/Proyecto_VectorIA`. Desde Windows corresponde a `\\wsl.localhost\Ubuntu-24.04\home\hadoop\CUC\IA_Aplicada\Proyecto_VectorIA`; `/mnt/c/home/...` es otra ubicación.

El entorno existente `.Vector_IA_wsl` fue creado con **Python 3.12.3**, según su configuración. El PDF requiere Python sin fijar versión; el comentario Python 3.11 de `requirements.txt` es heredado y está pendiente de conciliación. Se requiere Git para versionado.

Desde WSL, para preparar dependencias cuando corresponda:

```bash
cd /home/hadoop/CUC/IA_Aplicada/Proyecto_VectorIA
source .Vector_IA_wsl/bin/activate
python --version
python -m pip install -r requirements.txt
```

El stack del PDF incluye TensorFlow/Keras, OpenCV, pandas, Matplotlib, Seaborn, Plotly y Streamlit; contempla Gradio como alternativa, FastAPI como opción y Docker como extra. **OpenCV y Plotly aún no están declarados en `requirements.txt`**. Su incorporación corresponde a la implementación; esta actualización no modifica dependencias ni instala paquetes.

Para notebooks se declaran `ipykernel` y `nbformat`. En VS Code conectado a WSL, seleccionar el intérprete o kernel del proyecto; los notebooks guardan el nombre `Vector_IA_wsl(GPU)`. El nombre no confirma GPU disponible: verificar TensorFlow y CUDA en la sesión utilizada. Reutilizar el entorno existente sin recrearlo automáticamente.

## Ejecución

1. Abrir en VS Code conectado a WSL los notebooks desde `notebooks/`, seleccionar el kernel y ejecutar en orden `1_EDA.ipynb` y `2_Modelo_CNN.ipynb`. Revisar antes rutas y categorías, que conservan referencias de experimentos anteriores. Los notebooks LSTM y ANN de acciones están pendientes.
2. Una vez adaptada la API opcional y disponibles los artefactos, iniciarla desde la raíz:
   ```bash
   .Vector_IA_wsl/bin/python -m uvicorn api.main:app --reload
   ```
3. Una vez adaptada la demo, iniciar Streamlit desde la raíz:
   ```bash
   .Vector_IA_wsl/bin/python -m streamlit run app/Home.py
   ```

Los comandos de servidor corresponden a puntos de entrada existentes, pero hoy ejecutan componentes heredados y no garantizan una demo VectorIA funcional.

El flujo objetivo es subir foto, elegir semana, ver pronóstico de cuatro semanas contra el umbral endémico, obtener acción y motivo y explorar proporciones cantonales. Los controles «¿qué pasaría si?» para lluvia o casos son un extra. Mantener visible el aviso de apoyo a la vigilancia y ausencia de diagnóstico en personas.

## Entregables

- [ ] README con propósito, instalación y uso verificables.
- [ ] EDA de imágenes, series y acciones, con calidad de datos y procedencia.
- [ ] CNN propia y MobileNetV2 comparadas mediante F1 macro, meta >= 85 %.
- [ ] SimpleRNN y LSTM/GRU con RMSE y MAE contra baseline estacional.
- [ ] ANN de acciones con partición por semana y precision/recall, meta >= 85 %.
- [ ] Modelos en `.h5` o `.keras` y transformaciones necesarias para inferencia.
- [ ] Interfaz Streamlit con los tres módulos y mapa cantonal.
- [ ] Aviso ético fijo: no diagnostica dengue en personas.
- [ ] Informe técnico con diseño, resultados, métricas y recomendaciones.
- [ ] Código publicado en GitHub por el desarrollador.
- [ ] Opcional: API para mosquito, casos y acción.
- [ ] Extras: larvas MLMI-2024, pronóstico provincial sintético, escenarios interactivos y Docker.

**Checkpoint:** 2 de noviembre (EDA y primer modelo). **Entrega:** 9 de noviembre. Fechas del PDF en el contexto del curso 2026; no se indica hora.

**Rúbrica:** modelos 40 %, producto 30 %, documentación 20 %, innovación 10 %. Las casillas representan validación final, no ausencia de trabajo parcial.

## Resultados

| Modelo o comparación | Métricas requeridas | Meta o referencia | Estado |
|---|---|---|---|
| CNN propia / MobileNetV2 | F1 macro de ambas | >= 85 % | Artefactos CNN propios; comparación y adecuación de clases pendientes |
| SimpleRNN / LSTM o GRU | RMSE y MAE | Baseline estacional del año anterior | Pendiente |
| ANN de acciones | Precision y recall, detalle por clase | >= 85 % | Pendiente |

**Modelo seleccionado y justificación:**

Pendiente de evaluación comparable sobre conjuntos reservados. Hay dos modelos CNN guardados y tres checkpoints, pero su existencia y las salidas históricas no prueban cumplimiento del alcance nuevo. Las métricas de abandono de clientes de la plantilla no corresponden a VectorIA.

## Conclusiones y recomendaciones

La base actual permite continuar la experimentación con imágenes. Antes de validar la CNN, adecuar las clases objetivo y comparar con MobileNetV2. Completar el pronóstico temporal, la ANN de acciones y la demo, manteniendo trazabilidad entre datos, transformaciones y artefactos.

Distinguir cifras reales, escenarios simulados y datos sintéticos. Interpretar el mapa como comparación relativa y las recomendaciones como salidas académicas de apoyo; no presentar métricas ni reglas del dataset como evidencia de eficacia sanitaria o diagnóstico individual.
