# ESTADO_ARCHIVO: EN_MODIFICACION
"""Contratos de VectorIA. Los nombres y el orden entrenado viven en Configuracion."""
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

Slot = Literal['cnn', 'rnn', 'ann']


class Contrato(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)


class Modelo(Contrato):
    archivo: str | None = None
    columnas: list[str] = Field(default_factory=list)
    clases: list[str] = Field(default_factory=list)
    scaler: str | None = None
    scaler_salida: str | None = None
    ventana: int = Field(default=26, ge=1, le=520)
    horizonte: int = Field(default=4, ge=1, le=52)
    fecha: str = 'fecha_inicio_semana'
    grupo: str = 'provincia'
    objetivo: str = 'casos_dengue'
    normalizacion: Literal['interna', 'dividir_255', 'ninguna'] = 'interna'
    clase_deteccion: str = 'aegypti'

    @model_validator(mode='after')
    def nombres_unicos(self):
        for values in (self.columnas, self.clases):
            if len(set(values)) != len(values) or any(not v.strip() for v in values):
                raise ValueError('Los nombres deben ser únicos y no vacíos.')
        return self


class Configuracion(Contrato):
    cnn: Modelo = Field(default_factory=lambda: Modelo(clases=['aegypti', 'albopictus', 'anopheles', 'culex']))
    rnn: Modelo = Field(default_factory=lambda: Modelo(columnas=['casos_dengue', 'lluvia_mm', 'temp_max_c', 'temp_min_c', 'humedad_rel_pct']))
    ann: Modelo = Field(default_factory=lambda: Modelo(
        columnas=['casos_pred_4sem', 'casos_umbral_endemico', 'lluvia_14d_mm', 'temp_max', 'temp_min', 'humedad', 'aegypti_detectado', 'semana_epi'],
        clases=['monitoreo_normal', 'eliminacion_criaderos', 'fumigacion_focalizada', 'alerta_sanitaria']))


class Tabla(Contrato):
    filas: list[dict[str, Any]] = Field(min_length=1, max_length=10000)


class Preparacion(Tabla):
    renombrar: dict[str, str] = Field(default_factory=dict)
    tipos: dict[str, Literal['float64', 'string', 'category']] = Field(default_factory=dict)
    mapas: dict[str, dict[str, int]] = Field(default_factory=dict)
    dummies: list[str] = Field(default_factory=list)
    eliminar_nulos: bool = False


class Analisis(Tabla):
    columnas: list[str] = Field(min_length=2)
    categoricas: list[str] = Field(default_factory=list)
    modalidad: Literal['numericas', 'dummies', 'mixtas'] = 'mixtas'
    metodo: Literal['pearson', 'spearman'] = 'pearson'


class Imagen(Contrato):
    nombre: str
    contenido: str


class Inferencia(Contrato):
    registros: list[dict[str, Any]] = Field(default_factory=list, max_length=10000)
    imagenes: list[Imagen] = Field(default_factory=list, max_length=100)
    historial: list[dict[str, Any]] = Field(default_factory=list, max_length=10000)
    columna_imagen: str = 'imagen'
    revision: str | None = None


class Evaluacion(Inferencia):
    modelo: Slot
    conjunto: str = Field(min_length=1, max_length=200)
    objetivo: str = 'recomendacion'


class Importacion(Tabla):
    revision: str | None = None


class SeleccionEvaluacion(Contrato):
    modelo: Slot
    id: str
