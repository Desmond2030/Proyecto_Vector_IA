<!-- ESTADO_ARCHIVO: EN_MODIFICACION -->
# VectorIA · aplicación Streamlit

Desde `/home/hadoop/CUC/IA_Aplicada/Proyecto_VectorIA`:

```bash
.Vector_IA_wsl/bin/python -B app/run.py
```

App: http://127.0.0.1:8501 · API: http://127.0.0.1:8000/docs. Ctrl+C cierra los procesos iniciados por el lanzador. Para usar otra API, iniciar Streamlit por separado con `VECTORIA_API_URL` en el entorno. No se instala ninguna dependencia nueva.

## Páginas

- **Inicio:** portada editorial, ilustración, datos de `app/proyecto.json` y evaluaciones vigentes. No contiene métricas inventadas.
- **Explorar:** abrir CSV/Excel/Parquet desde el proyecto o subirlo; editar y preparar una copia, filtrar y graficar con Altair. Numéricas Pearson/Spearman, numéricas más dummies y asociaciones mixtas (Dython). Seleccionar tipos semánticos, no solo tipos físicos.
- **Predicción:** imagen, observación y contexto; flujo completo o CNN, RNN y ANN por separado. Para RNN, cargar historial y elegir fecha y ubicación.
- **Lotes:** CSV de observaciones e imágenes vinculadas por nombre; historial separado para pronóstico. Exporta columnas originales y resultados sin sobrescribir nombres existentes.
- **Evaluación:** importar CSV de métricas, calcular sobre etiquetas reales y elegir qué evaluación aparece en Inicio.
- **Modelos:** elegir `.keras`/`.h5` del inventario o subir artefactos propios; configurar orden de clases/entradas y scalers. Sin modelos cargados, explorar datos sigue funcionando.

## Datos y edición

La biblioteca de datasets vive en `st.session_state`: se conserva entre páginas y se pierde al cerrar/reiniciar la sesión. Las copias preparadas pueden descargarse. No se sobrescriben los CSV originales. `_fila_origen` permite seguir la procedencia de filas editadas. Guardar preparación utiliza `TabularDataProcessor` a través del API.

Los campos avanzados aceptan JSON, por ejemplo:

```json
{"fecha": "fecha_inicio_semana"}
```

Para tipos: `{"lluvia_mm": "float64", "provincia": "category"}`. Para mapas: `{"deteccion": {"Sí": 1, "No": 0}}`.

Las asociaciones muestran el número de filas completas utilizadas, las excluidas, la categoría omitida en dummies y el método por pareja. Pearson/Spearman tienen signo; V de Cramér y eta expresan intensidad sin dirección. El análisis no demuestra causalidad. Las categorías codificadas 0/1 pueden marcarse como categóricas en el modo mixto.

## Ajustes habituales

- Créditos y descripción: `proyecto.json`.
- Tipografía y disposición: `assets/styles.css`.
- Colores por página: `ui.py`.
- Contratos entrenados: página Modelos, persistidos en `api/configuracion.json`.
- Añadir o cambiar datos: cargar archivo y mapear entradas, sin editar código si el contrato entrenado no cambia.
- Correlaciones: `api/datasets.py`; presentación Altair en `results.py`.

El diseño usa fondo marfil, tipografía editorial y navegación superior. Inicio verde, Explorar ámbar, Predicción azul, Lotes coral, Evaluación violeta y Modelos pizarra. La ilustración de portada es conceptual; los resultados mostrados por la app proceden exclusivamente de inferencia o evaluaciones identificadas.

## Estado y límites

No se entrenan redes. Los módulos sin artefactos compatibles permanecen pendientes. La evaluación CNN reutiliza `evaluar()` de `src`; el API conecta la inferencia de la ANN multiclase de acciones y la RNN de pronóstico mediante sus perfiles configurados.

El registro en contexto y Uso de IA se realizará únicamente por indicación posterior del desarrollador.

## Ilustración

`assets/portada.png` generada con la herramienta incorporada imagegen. Prompt: ilustración editorial científica de un mosquito en grafito, papel marfil con cuadrícula muy tenue, acento verde discreto, sin textos ni interfaz; proporción 4:3. Es un recurso decorativo, no una clave de identificación taxonómica.
