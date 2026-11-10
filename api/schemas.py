# ESTADO_ARCHIVO: EN_MODIFICACION
"""Contratos pequeños para exploración, una observación y evaluación."""
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

Slot = Literal['cnn', 'rnn', 'ann']
SERIE_NACIONAL = 'data/proceced/archivos_parquet/vectoria_dengue_nacional_real.parquet'
COLUMNAS_RNN = ['poblacion', 'lluvia_mm', 'temp_max_c', 'temp_min_c', 'humedad_rel_pct'] + [
    f'{variable}_lag{lag}' for variable in ['lluvia_mm', 'temp_max_c', 'temp_min_c', 'humedad_rel_pct']
    for lag in range(6, 0, -1)]
COLUMNAS_ANN = ['casos_pred_4sem', 'casos_umbral_endemico', 'lluvia_14d_mm',
                'temp_max', 'temp_min', 'humedad', 'aegypti_detectado']


class Contrato(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)


class Modelo(Contrato):
    archivo: str | None = None
    columnas: list[str] = Field(default_factory=list)
    clases: list[str] = Field(default_factory=list)
    scaler: str | None = None
    scaler_salida: str | None = None
    serie: str = SERIE_NACIONAL
    ventana: Literal[26] = 26
    horizonte: Literal[4] = 4
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
    rnn: Modelo = Field(default_factory=lambda: Modelo(columnas=COLUMNAS_RNN))
    ann: Modelo = Field(default_factory=lambda: Modelo(columnas=COLUMNAS_ANN,
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
    ruta: str | None = None
    nombre: str | None = None
    contenido: str | None = None

    @model_validator(mode='after')
    def origen_unico(self):
        if self.ruta:
            if self.nombre or self.contenido:
                raise ValueError('Elige una muestra del proyecto o una imagen cargada.')
        elif not self.nombre or not self.contenido:
            raise ValueError('Falta la imagen.')
        return self


class Contexto(Contrato):
    casos_umbral_endemico: float = Field(ge=0)
    lluvia_14d_mm: float = Field(ge=0)
    temp_max: float
    temp_min: float
    humedad: float = Field(ge=0, le=100)

    @model_validator(mode='after')
    def temperaturas(self):
        if self.temp_min > self.temp_max:
            raise ValueError('La temperatura mínima no puede superar la máxima.')
        return self


class Inferencia(Contrato):
    imagen: Imagen
    fecha: str
    ubicacion: str
    contexto: Contexto
    revision: str


class Evaluacion(Contrato):
    modelo: Slot
    revision: str
    conjunto: Literal['val', 'test'] = 'val'
    limite: int = Field(default=20, ge=1, le=1000)
    ubicacion: str | None = None
