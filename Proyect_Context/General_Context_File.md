# ESTADO_ARCHIVO: ESTABLE
# General_Context_File
## Elemento fijo de archivos de contexto:
- Este archivo resguarda el contexto del proyecto; En cada proceso del proyecto, para mantener un flujo coherente por cada conversacion, quedara como un historial del **"como evoluciono el proyecto"**, **manteniendo la version mas compacta** del **como se tomo una decicion sobre el proyecto**, del **como se van implementando cada cambio**, del **porque se realiza la implementacion** y en tal caso el **porque se presindio de un recurso ya implementado en el proyecto**, **integrando las explicaciones por parte del agente y del desarrolador en cuestion, <u>el de este ultimo es el que rige en mayor medida el proyecto</u>**, citando de forma resumida, segun que tan extensa sea la converzacion sobre una decicion,  o literal integrando los mensajes mas importantes segun lo comentado por parte del desarrolador en el chat.

##############################################################

# Context from here:


-------------------------------------------------------------
# Historia_01:
## **Inicio de VectorIA en WSL y evolución desde el proyecto ANN hacia clasificación de imágenes con CNN**.

**Fecha:** 2026-11-02 (America/Costa_Rica).

**Notas_Importantes:** La rama actual es `feature/Modelo_CNN` y todavía no tiene commits. Esta historia documenta el contenido existente y sus antecedentes; las versiones anteriores se identifican mediante conversaciones, registros heredados y artefactos, no como versiones publicadas de VectorIA. El archivo de contexto no contenía historias previas. Se conserva íntegro el elemento fijo y se añade esta nota después de `Context from here`.

Agente:**Humano**-> Desmond
- Identifica como predecesor el proyecto «Proyecto ANN en WSL», trabajado en `Proyecto_Abandono_Clientes`, y establece como raíz principal `/home/hadoop/CUC/IA_Aplicada/Proyecto_VectorIA` en Ubuntu-24.04.
- Define que el desarrollador realiza los cambios de código y que el agente solo interviene cuando recibe autorización. Las consultas y revisiones de lectura no deben generar entradas en `USO_DE_IA.md`; las modificaciones autorizadas sí deben registrarse.
- Solicita: «no realizaras el commit, yo lo realizare, solo pasa el mensaje de lo que integra nuevo el proyecto en la rama», junto con la incorporación de esta historia y el registro de la intervención.

Agente:ChatGPT (Codex)
- **Base heredada:** el proyecto ANN aportó clases de carga, procesamiento tabular, gráficos y modelos ANN, así como una API FastAPI y un dashboard Streamlit orientados a abandono de clientes. Los registros heredados describen particiones, escalado, clasificación y regresión, integración de la página y modelos de referencia. Estos componentes permanecen como base reutilizada; su presencia no demuestra una adaptación de la API o la página a clasificación de mosquitos. README y AGENTS.md todavía describen el alcance anterior y quedan pendientes de actualización por el desarrollador.
- **Antecedentes de WSL:** en el chat «Investiga el error en WSL» se identificó un conflicto entre bibliotecas CUDA del sistema y del entorno `.IA_Abandono_wsl`, junto con errores en PATH. El historial confirma detección de GPU tras un ajuste temporal y el registro de un kernel específico; también señala la necesidad de seleccionarlo en VS Code. Este antecedente no constituye una validación del entorno actual. Los notebooks de VectorIA guardan el nombre de kernel `Vector_IA_wsl(GPU)` y `.gitignore` excluye `.Vector_IA_wsl/`.
- **Evolución de las clases:** el chat «Revisa Modelo_CNN.py» recoge revisiones sucesivas de carga de imágenes, tokenización, corpus, predicción individual y por lotes, umbrales y evaluación. El desarrollador aclaró que la preparación imagen-texto se destina a modelos distintos de la CNN clasificadora; no se documenta un modelo de difusión terminado. Las propuestas de aquel chat no se consideran automáticamente implementadas.
- **Contenido actual de VectorIA:** `Vision_Model_Procesor` reúne carga por carpetas, rutas y etiquetas, preparación imagen-texto, visualización de muestras, extracción de rutas y dimensiones con `set_df_img()` y traslado por categoría con `move_files_in_img_paths()`. `CNN_Model_Clasification` configura convoluciones, normalización, aumento de datos, pooling, transición GAP/Flatten y capas densas; integra entrenamiento con prefetch, callbacks, tiempo de ejecución e historial, predicción y evaluación con métricas y matrices de confusión.
- **Cambios observados durante esta conversación:** la visualización distingue datasets directos de diccionarios; la salida del notebook CNN usa `len(img_classify.categories)` en lugar del valor fijo 4; el recorrido de imágenes usa la subcarpeta y rutas individuales; las claves de dimensiones coinciden con las columnas y se crea un procesador por especie. Estas correcciones se observaron en las versiones guardadas por el desarrollador; el agente realizó diagnósticos de solo lectura.
- **Notebooks y versiones:** actualmente existen `notebooks/1_EDA.ipynb` y `notebooks/2_Modelo_CNN.ipynb`; en revisiones anteriores se consultó `3_Modelo_CNN.ipynb`. El EDA guardado explora rutas de imágenes y el notebook CNN configura entrenamiento, callbacks, evaluación y guardado opcional. Las llamadas de particionado y movimiento vistas en capturas no figuran en el EDA guardado consultado, por lo que no se afirma su persistencia ni ejecución completa.
- **Artefactos anteriores y actuales:** se verificó mediante lectura de `config.json` que los modelos `CNN_Vector_Clasification_1.keras` y `CNN_Vector_Clasification_2.keras` tienen bloques densos de 38/22/13/3 y 32/16/3 unidades, respectivamente. Existen checkpoints de ambos y un tercer checkpoint con bloque denso 64/3. Los cinco artefactos tienen convoluciones de 32/64 filtros y tres salidas. No se cargaron para inferencia ni se comparó su calidad; los números de sus nombres no equivalen a versiones Git.
- **Límites pendientes:** el movimiento de imágenes actualiza el DataFrame solo al terminar y no controla colisiones de nombres ni recuperación ante fallos parciales. Se distinguió `/mnt/c/home/...` (Windows) de `/home/hadoop/...` (raíz WSL). Persisten comprobaciones y adaptación documental pendientes; no se afirma que el proyecto completo sea reproducible ni que la API y el dashboard sirvan al modelo CNN.
- **Intervención actual:** consulta de los dos chats referenciados, código, notebooks, metadatos de modelos, instrucciones y Git; adición de esta historia y del registro de uso de IA. El mensaje propuesto sigue la convención histórica `features/Título: descripción`, observada también en commits del predecesor. No se ejecutan entrenamientos, movimientos de imágenes, instalaciones, commit ni push.

**Fuentes de contexto:**
- [Revisa Modelo_CNN.py](thread://01a161c3-b647-7a41-9cd7-6fbe36e2b996?hostId=local).
- [Investiga el error en WSL](thread://01a15c24-24e0-7760-9d6f-6e4a59f77024?hostId=local).
- Registro heredado `USO_DE_IA.md`, archivos actuales y Git del predecesor: `features/Base del modelo ANN` (c1623ec), `feature/pagina` (428dfb0), `features/Modelos de referencia` (8f88b85). Estos hashes pertenecen al predecesor, no a VectorIA.

Archivos_manipulados_por_el_agente: `Proyect_Context/General_Context_File.md` y `USO_DE_IA.md` (solo adiciones). El código, notebooks, modelos, datos y documentos heredados restantes se consultaron sin modificarlos.
-------------------------------------------------------------
