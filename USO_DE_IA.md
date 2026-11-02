# Uso de IA

## Intervención 1 — Documento de contexto del dataset

**Herramienta:** Claude Code (Claude Sonnet 5.5, Anthropic).

**Tareas:**
- Búsqueda y lectura de fuentes sobre el dataset (IBM Community/Cognos Analytics, metadatos de Kaggle, Mendeley Data).
- Verificación de cifras sobre `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv` (distribuciones, valores en blanco, relaciones entre variables).
- Redacción de `data/raw/docs/Telco_Data_Context.md`.
- Actualización de `AGENTS.md` (seguimiento de archivos y tabla de archivos controlados).

**Prompt utilizado (resumen):** Generar en `data/raw/docs` el documento `Telco_Data_Context.md` con el máximo contenido verídico y respaldado por fuentes confiables sobre el dataset de Kaggle `blastchar/telco-customer-churn`; incluir tablas con el nombre de la variable, su significado y su renombrado al español (captura de las columnas renombradas); ser conciso sin recortar contenido importante; no crear otras carpetas ni archivos.

**Reflexión sobre cómo la IA ayudó en el aprendizaje:** 
Ayudo a comprender la data sin tener que filtar los sitos a mano de antemano

**Modificaciones realizadas por el desarrollador al código/análisis generado:** 
Ninguna

## Intervención 2 — Revisión de cambios y redacción del mensaje de commit

**Fecha:** 2026-10-10.

**Herramienta:** ChatGPT (Codex).

**Tareas:**
- Consulta del estado de Git, diferencias pendientes e historial de commits.
- Lectura de documentación, módulos de `src/` y celdas de los notebooks para resumir los cambios existentes.
- Redacción de un mensaje de commit en español con el formato `features/Título: descripción`, observado en el historial disponible; los proyectos vecinos consultados no tenían repositorio Git propio.
- Identificación de que el CSV original y el Parquet procesado siguen apareciendo como archivos no versionados pese a las entradas de `.gitignore`.
- Sin ejecución de notebooks, instalación de dependencias ni creación del commit; el usuario realizará el commit.

**Prompt utilizado:** «Genera un mensaje tipo commit que contenga todos los cambios realizados en el proyecto, yo realizare el commit pero tu indicaras los cambios mediante el mensaje, mantén una convención con los otros proyectos».

**Reflexión sobre cómo la IA ayudó en el aprendizaje:** 
Mantener estructura en la convencion de commits del proyecto

**Modificaciones realizadas por el desarrollador al código/análisis generado:** 
Ninguno

## Intervención 3 — Mensaje de commit para particiones y escalado

**Fecha:** 2026-10-10.

**Herramienta:** ChatGPT (Codex).

**Tareas:**
- Revisión del estado de Git y de las diferencias respecto al commit bfb8d28, incluidos archivos nuevos y comparación de las celdas de los notebooks renombrados.
- Redacción del mensaje con la convención existente `features/Título: descripción`, centrado en numeración de notebooks, particiones estratificadas, escalado, persistencia y exclusiones de Git.
- Identificación mediante lectura del código de un problema pendiente en la exportación `.npy`: `save_data` usa `self.array`, mientras el notebook asigna los subconjuntos a `data`; además, el setter de `array` se llama a sí mismo.
- Sin cambios al código o a los notebooks, sin ejecutarlos y sin crear el commit.

**Prompt utilizado:** «Genera un mensaje tipo commit que contenga todos los cambios realizados en el proyecto, yo realizare el commit pero tu indicaras los cambios mediante el mensaje, mantén una convención con los otros proyectos».

**Reflexión sobre cómo la IA ayudó en el aprendizaje:** 
Mantener estructura en la convencion de commits del proyecto

**Modificaciones realizadas por el desarrollador al código/análisis generado:** 
Ninguno

## Intervención 4 — Ajuste de dependencias del proyecto

**Fecha:** 2026-10-11.

**Herramienta:** ChatGPT (Codex).

**Prompt utilizado:** «Pon el archivo [requirements.txt](requirements.txt) en regla con el proyecto».

**Tareas:**
- Revisión de las instrucciones, README, estado de Git, imports de los módulos y notebooks actuales, y metadatos del entorno Python 3.11.9 existente.
- Incorporación de scipy, dython, pyarrow, keras y nbformat como dependencias explícitas; declaración de ipykernel para los notebooks.
- Establecimiento de matplotlib>=3.10 por el argumento orientation de boxplot y keras>=3 en consonancia con TensorFlow>=2.16; conservación de las dependencias previstas para API y dashboard, identificadas mediante comentarios.
- Consulta de documentación oficial: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.boxplot.html y https://pandas.pydata.org/docs/reference/api/pandas.read_parquet.html.
- Verificación de sintaxis, ausencia de duplicados, cobertura de imports externos y cumplimiento de restricciones por las versiones instaladas. pip check no encontró conflictos en el entorno existente.
- Comprobación satisfactoria de imports de las dependencias añadidas y del kernel, TensorFlow/Keras, lectura/escritura Parquet en memoria con datos ficticios y creación de un boxplot horizontal sin guardar archivos.
- Jupyter y Notebook no están instalados en el entorno local; jupyter se conserva en requirements.txt. No se instalaron paquetes ni se validó una instalación desde cero. El archivo declara dependencias directas y no bloquea todas sus versiones.
- El notebook nuevo del modelo ANN está vacío; se revisaron los otros tres notebooks y los módulos de src. AGENTS.md conserva una descripción anterior del proyecto. No se modificaron código, notebooks, datos ni cambios previos del usuario, ni se ejecutó entrenamiento.

**Reflexión sobre cómo la IA ayudó en el aprendizaje:** Ayudo a mantener una mejor reproductivilidad del proyecto de forma ordenada sin escribir tanto 

**Modificaciones realizadas por el desarrollador al código/análisis generado:** 
Ninguno

## Intervención 5 — Mensaje de commit para la base ANN y persistencia de datos

**Fecha:** 2026-10-11.

**Herramienta:** ChatGPT (Codex).

**Tareas:**
- Revisión del estado de Git, historial y diferencias respecto al commit cb884d3, incluidos los archivos nuevos y las celdas modificadas del notebook de particiones.
- Redacción del mensaje con la convención `features/Título: descripción`, abarcando la base de clases ANN, exportación de particiones en CSV, manejo del escalador, utilidades, dependencias y exclusiones.
- Comprobación de que el notebook del primer modelo está vacío; la implementación ANN se describe como desarrollo inicial, sin afirmar entrenamiento ni resultados validados.
- Solo se añadió este registro al documento; no se modificaron código, notebooks, dependencias ni datos. No se ejecutaron pruebas, notebooks o entrenamientos, ni se creó el commit.

**Prompt utilizado:** «Genera un mensaje tipo commit que contenga todos los cambios realizados en el proyecto, yo realizare el commit pero tu indicaras los cambios mediante el mensaje, mantén una convención con los otros proyectos».

**Reflexión sobre cómo la IA ayudó en el aprendizaje:** 
Mantener estructura en la convencion de commits del proyecto

**Modificaciones realizadas por el desarrollador al código/análisis generado:** 
Ninguno

## Intervención 6 — Mensaje de commit del proyecto principal en WSL

**Fecha:** 2026-10-12.

**Herramienta:** ChatGPT (Codex).

**Tareas:**
- Revisión del estado de Git, historial y diferencias pendientes exclusivamente en el proyecto principal de WSL, incluidos archivos nuevos, fuentes de las celdas y artefactos de modelos.
- Redacción del mensaje con la convención existente `features/Título: descripción`, abarcando los dos notebooks ANN, configuración y evaluación de modelos, partición 80/20, escalador, modelos y checkpoints, dependencias, exclusiones y documentación.
- Distinción entre actualizaciones de salidas/metadatos y cambios de código en los notebooks; los nombres de los modelos guardados no se consideran métricas verificadas.
- Solo se añadió este registro. No se ejecutaron notebooks, entrenamientos ni pruebas, ni se creó el commit. No se modificó la copia de Windows.

**Prompt utilizado:** «Genera un mensaje tipo commit que contenga todos los cambios realizados solo en el proyecto principal de WSL, yo realizare el commit pero tu indicaras los cambios mediante el mensaje, mantén una convención con los otros proyectos».

**Reflexión sobre cómo la IA ayudó en el aprendizaje:** 
Mantener estructura en la convencion de commits del proyecto

**Modificaciones realizadas por el desarrollador al código/análisis generado:** 
Ninguno


## Intervención 7 — Resolución del merge en WSL

**Fecha:** 2026-10-12.

**Herramienta:** ChatGPT (Codex).

**Prompt utilizado:** «Revisa los conflictos en la carpeta de WSL, arregla los merge para yo despues realizar el pull». Continuación: «sigue con lo que estabas».

**Tareas:**
- Comparación de la base común, la versión local y la remota del merge pendiente en Feature/Modelos; identificación de cuatro archivos en conflicto.
- Conservación de la versión local de src/Modelo_ANN.py, compatible con los dos notebooks ANN actuales: preparación de experimentos, callbacks, batch_size, activaciones, reducción de neuronas y Regression_Model.
- Conservación del notebook de particiones local con test_size=0.20 y sus salidas asociadas; la única diferencia de código frente al remoto era test_size=0.15.
- Conservación de tensorflow[and-cuda]>=2.16 sin comillas y de la selección local de dependencias para notebooks (ipykernel y nbformat).
- Conservación íntegra de las seis intervenciones anteriores, incluida la intervención 6 que no estaba en la versión remota, y adición de este registro.
- Respaldo temporal de los cuatro archivos antes de resolver; validación estructural de notebooks, sintaxis de Python y requisitos, compatibilidad de los argumentos usados por los notebooks y ausencia de marcadores de conflicto.
- Preparación del commit local de merge para desbloquear el siguiente pull. Sin ejecutar notebooks ni entrenamientos, instalar dependencias, modificar datos/modelos o trabajar sobre la copia de Windows; sin pull ni push.

**Reflexión sobre cómo la IA ayudó en el aprendizaje:** 
facilita la resolucion de conflicto para no tener que corregir a mano

**Modificaciones realizadas por el desarrollador al código/análisis generado:** 
Ninguno

## Intervención — Copia de los modelos de referencia a WSL

**Fecha:** 2026-10-12.
**Herramienta:** ChatGPT (Codex) y PowerShell.
**Prompt utilizado:** «Copia esos modelos».

**Tareas realizadas:** Copia desde models/ de la carpeta secundaria de Windows a models/ del proyecto principal WSL de modelo_74_accur.keras y regression_model.keras. Ambos destinos estaban ausentes; se verificó la igualdad SHA-256 de cada copia con su origen. Se comprobó además que scaler.pkl ya era idéntico en ambos proyectos, por lo que no se reemplazó. Los modelos de referencia ya están disponibles en WSL, actualizando la limitación registrada al integrar app/ y api/. No se entrenaron ni alteraron redes, no se cambió la configuración de inferencia y no se realizó commit ni push en esta intervención.

**Reflexión sobre cómo la IA ayudó en el aprendizaje:** 
Mantener estructura limpia del proyecto con nuevas adiciones y verificacion de que son las mismas

**Modificaciones realizadas por el desarrollador al código/análisis generado:** 
Ninguno

## Intervención — Commit de modelos de referencia

**Fecha:** 2026-10-12.
**Herramienta:** ChatGPT (Codex), Git y PowerShell.
**Prompt utilizado:** «realiza el commit siguiendo la convencion».
**Tareas realizadas:** Revisión de los cambios pendientes en feature/pagina y de los mensajes recientes. Preparación del commit con el formato observado features/Área: Se describe el cambio, incluyendo únicamente los dos modelos de referencia copiados y su registro en USO_DE_IA.md. Se normalizan a LF los finales de línea del registro pendiente agregado por el agente. No se modifican redes ni configuración y no se realiza push.
**Reflexión sobre cómo la IA ayudó en el aprendizaje:** 
Mantener estructura en la convencion de commits del proyecto

**Modificaciones realizadas por el desarrollador al código/análisis generado:** 
Ninguno

## Intervención — Comparación del avance entre WSL y Windows

**Fecha:** 2026-10-12.
**Herramienta:** ChatGPT (Codex), Git y PowerShell.
**Prompt utilizado:** «cual es el proyecto mas avanzado».
**Tareas realizadas:** Comparación del estado de Git, archivos de código, notebooks, API, dashboard y hashes de modelos de ambas copias. WSL, en feature/pagina y commit 8f88b85, reúne el segundo notebook ANN, la clase Regression_Model, la API y la página integradas y los modelos de referencia. La API coincide con Windows al normalizar finales de línea; los modelos compartidos y el escalador son idénticos. Windows conserva cambios sin commit y una configuración local de inferencia que no existe en WSL. La conclusión se refiere al avance e integración de los archivos; no se ejecutaron aplicaciones, pruebas ni entrenamientos. Solo se añadió este registro en WSL, sin commit.
**Reflexión sobre cómo la IA ayudó en el aprendizaje:** 
Mantener estructura entre la zona del desarrollador y el agente para tratar dos lineas de desarrollo, la agentica y la manual
**Modificaciones realizadas por el desarrollador al código/análisis generado:** 
Ninguno solo revicion

## Intervención — Recarga automática de Streamlit

**Fecha:** 2026-10-12.
**Herramienta:** ChatGPT (Codex), PowerShell y Python del entorno WSL existente.
**Prompt utilizado:** «hot reload streamlit».
**Tareas realizadas:** Activación de --server.runOnSave true y --server.fileWatcherType poll en app/run.py. Actualización de app/Specs.md con el comando equivalente para iniciar solo Streamlit. Se verificó la instancia existente mediante PID, directorio y argumentos; se reinició únicamente Streamlit con las opciones activas. Comprobación HTTP /_stcore/health satisfactoria y confirmación de ambos argumentos en el proceso nuevo. Sin cambios a API, artefactos o configuración de modelos. No se realizó commit ni push.
**Reflexión sobre cómo la IA ayudó en el aprendizaje:** 
Ayuda en la busqueda rapida de comandos

**Modificaciones realizadas por el desarrollador al código/análisis generado:** 
Solo consulta
## Intervención — Revisión de Modelo_CNN.py sin cambios de código

**Fecha:** 2026-10-21.
**Herramienta:** ChatGPT (Codex), PowerShell, Python del entorno existente .IA_Abandono_wsl y documentación oficial de Keras/TensorFlow.
**Prompt utilizado:** «revisa Modelo_CNN.py y comprueba por deficincias sin cambiar codigo».
**Tareas realizadas:** Revisión de src/Modelo_CNN.py, README, dependencias y llamadas del notebook CNN. Comprobación de sintaxis y reproducciones puntuales en memoria, con datos ficticios y sin entrenar: nombres no definidos en carga de imágenes, atributo ausente en predicción por lotes, conversión inválida de argmax a DataFrame, setter que reemplaza el modelo y reducción de neuronas sin efecto. Revisión adicional de tamaños/canales y codificación de etiquetas, uso de prueba como validación, conexión de datasets al entrenamiento y prefetch descartado. Se detectaron llamadas desactualizadas en el notebook. El entorno documentado .IA_Abonado no existe; las verificaciones utilizaron el entorno WSL ya existente .IA_Abandono_wsl (Python 3.12). No se modificó código, notebooks, datos ni modelos, ni se instalaron paquetes. Solo se añadió esta entrada; sin commit ni push. No se ejecutó el flujo completo ni se midió calidad predictiva.


## Intervención — Historia inicial de VectorIA y propuesta de mensaje de commit

**Fecha:** 2026-11-02 (America/Costa_Rica).

**Herramienta:** ChatGPT (Codex), herramientas de lectura de chats, PowerShell, Python y Git en WSL.

**Prompt utilizado:** «Comparteme por el chat un mensaje tipo commit siguendo convenciones de proyectos pasados, no realizaras el commit, yo lo realizare, solo pasa el mensaje de lo que integra nuevo el proyecto en la rama». Se solicitó además: «Integra una historia en la carpeta de contexto que mencione los cambios que se realizaron en este proyecto y las versiones anterioes y el proyecto predecesor "Proyecto ANN en WSL" he integra la intervencion» en `USO_DE_IA.md`, tomando como raíz el proyecto principal en Ubuntu-24.04 y como referencias los chats «Revisa Modelo_CNN.py» e «Investiga el error en WSL».

**Tareas realizadas:**
- Lectura de los chats referenciados, instrucciones de contexto, README, dependencias, estado de Git y registros heredados; contraste con la convención de commits del proyecto predecesor.
- Revisión estática del código y notebooks guardados y lectura de metadatos de los dos modelos CNN y tres checkpoints, sin cargar redes ni ejecutar inferencia.
- Incorporación de `Historia_01` al final de `Proyect_Context/General_Context_File.md`, después de `Context from here`, conservando íntegro el contenido previo. El historial estaba vacío, por lo que no existían dos historias anteriores que consultar.
- Documentación de la base ANN heredada, antecedentes de WSL/GPU, evolución de CNN y procesamiento de imágenes, correcciones observadas, versiones de artefactos y limitaciones pendientes. Se distinguen cambios del desarrollador, propuestas del agente y resultados guardados no reproducidos en esta intervención.
- Preparación para el chat de un mensaje con la convención `features/Título: descripción`. La rama `feature/Modelo_CNN` aún no tiene commits: se describe la incorporación inicial de su contenido, no una diferencia contra un commit inexistente.
- Comprobación de que ambas adiciones preservan los bytes originales y quedan al final de sus respectivos documentos. No se modifican código, notebooks, datos, modelos, configuración, README ni AGENTS.md. Sin ejecución de entrenamientos, movimientos de imágenes, instalaciones, staging, commit o push.

**Archivos modificados:** `Proyect_Context/General_Context_File.md` y `USO_DE_IA.md`, exclusivamente mediante adiciones.

**Reflexión sobre cómo la IA ayudó en el aprendizaje:** Pendiente de completar por el desarrollador.

**Modificaciones realizadas por el desarrollador al código/análisis generado:** Pendiente de completar por el desarrollador.


## Intervención — Adaptación de README y AGENTS a los requisitos de VectorIA

**Fecha:** 2026-11-02 (America/Costa_Rica).

**Herramienta:** ChatGPT (Codex), habilidad PDF, pypdf, Poppler, PowerShell, Python y Git/WSL.

**Prompt utilizado:** «Estos son los requerimientos del muevo proyecto, actualiza el readme.md y adapta el agents.md, manten la linea de estos documentos como plantilla, solo cambia el contenido», con VectorIA.pdf como documento adjunto de requisitos.

**Tareas realizadas:**
- Lectura de las diez páginas del PDF y comprobación visual de sus páginas renderizadas; separación de requisitos, arquitecturas sugeridas y extras opcionales.
- Contraste con código, notebooks, entorno, dependencias, carpetas y diccionario de datos. Conteo de los cuatro CSV sin modificar originales ni ejecutar scripts de generación.
- Actualización de README.md conservando sus secciones, tablas, árbol y orientación de instalación/uso. Se documentan CNN propia frente a MobileNetV2, SimpleRNN frente a LSTM/GRU, ANN de acciones, datos, particiones, métricas, demo, API opcional, fechas y rúbrica. Los resultados heredados de abandono se sustituyen por el estado pendiente de validación de VectorIA, sin inventar métricas.
- Adaptación de AGENTS.md conservando las once secciones numeradas y todas sus subsecciones. Se actualizan objetivo, estado, WSL, comandos y convenciones, manteniendo restricciones de edición y registro acumulativo; se precisa que las consultas y lecturas no generan intervenciones. README conserva Only_Dev con autorización puntual para esta actualización.
- Identificación explícita de diferencias pendientes: categorías CNN, API/dashboard heredados, módulos faltantes, OpenCV/Plotly no declarados y comentario Python 3.11 frente al entorno creado con 3.12.3.
- Validación de conservación de encabezados, delimitadores Markdown, presencia de requisitos esenciales, referencias actuales y retiro de instrucciones obsoletas de churn. Se preserva el contenido anterior de este registro mediante adición. Se guardaron respaldos temporales de los dos documentos fuera del proyecto.
- No se modifican código, notebooks, modelos, datasets, dependencias ni historial contextual. Sin entrenamientos, instalaciones, movimientos de imágenes, staging, commit o push.

**Archivos modificados:** README.md, AGENTS.md y USO_DE_IA.md (este último solo mediante adición).

**Reflexión sobre cómo la IA ayudó en el aprendizaje:** Pendiente de completar por el desarrollador.

**Modificaciones realizadas por el desarrollador al código/análisis generado:** Pendiente de completar por el desarrollador.


## Intervención — Incorporación de la licencia MIT del remoto

**Fecha:** 2026-11-02 (America/Costa_Rica).

**Herramienta:** ChatGPT (Codex), conector GitHub, Git, Python y PowerShell/WSL.

**Prompt utilizado:** «revisa el git de"\\wsl.localhost\Ubuntu-24.04\home\hadoop\CUC\IA_Aplicada\Proyecto_VectorIA" y el remoto para verificar que la licecia en el repositorio se integre dentro del proyecto actual».

**Tareas realizadas:**
- Revisión de la rama local feature/Modelo_CNN, del estado limpio del árbol de trabajo y de origin (Desmond2030/Proyecto_Vector_IA).
- Consulta directa del remoto: main contiene LICENSE con licencia MIT y copyright 2026 de Desmond Akin Bermúdez Rodríguez. El archivo no estaba en la rama local.
- Comprobación de que el commit local d406552 y origin/main (0536cf4) no comparten ancestro; se distingue la incorporación del archivo de una unión de historiales.
- Copia exacta a la raíz del archivo LICENSE cuyo blob remoto es 9b8684053ad8f3331ceb8946bca36012fe3c654d, verificando coincidencia con el objeto ya disponible en Git local y preservando íntegramente texto y autoría.
- Verificación de igualdad de bytes y de blob, conservación del contenido previo de este registro y ausencia de cambios en HEAD e índice. Quedan LICENSE sin seguimiento y esta entrada documental pendientes de incorporar por el desarrollador.
- Sin modificar código, datos, modelos, README o AGENTS; sin merge, rebase, staging, commit ni push. Los historiales siguen independientes.

**Archivos modificados:** LICENSE (nuevo) y USO_DE_IA.md (solo adición).

**Reflexión sobre cómo la IA ayudó en el aprendizaje:** Pendiente de completar por el desarrollador.

**Modificaciones realizadas por el desarrollador al código/análisis generado:** Pendiente de completar por el desarrollador.


## Intervención — Unificación de historiales local y remoto

**Fecha:** 2026-11-02 (America/Costa_Rica).

**Herramienta:** ChatGPT (Codex), Git, Python y PowerShell/WSL.

**Prompt utilizado:** «unifica los historiales».

**Tareas realizadas:**
- Actualización de referencias de origin mediante fetch; comprobación de feature/Modelo_CNN (d406552) y origin/main (0536cf4), con raíces independientes.
- Revisión de cambios pendientes: únicamente la licencia previamente copiada y su entrada de uso de IA. Respaldo temporal de LICENSE y los registros antes del merge.
- Preparación sin conflictos del merge de origin/main mediante --allow-unrelated-histories, preservando los dos historiales originales y el contenido local del proyecto.
- Incorporación de Historia_02 al contexto, conservación de todos los registros previos y adición de esta intervención.
- Creación del commit local de merge con dos padres, comprobación de ascendencia de ambas raíces, licencia idéntica a origin/main y ausencia de cambios ajenos a LICENSE y los dos documentos de registro.
- Sin cambios de código, datos, modelos o notebooks; sin entrenamiento, rebase, push ni modificación de main en el remoto.

**Archivos modificados:** LICENSE, Proyect_Context/General_Context_File.md y USO_DE_IA.md. Se incluyen en el commit de merge los cambios pendientes de la intervención anterior relativos a la licencia.

**Reflexión sobre cómo la IA ayudó en el aprendizaje:** Pendiente de completar por el desarrollador.

**Modificaciones realizadas por el desarrollador al código/análisis generado:** Pendiente de completar por el desarrollador.
