# ESTADO_ARCHIVO: EN_MODIFICACION
"""Adaptación de las clases del proyecto, sin entrenamiento."""
import hashlib
import numpy as np
from sklearn.utils.validation import check_is_fitted
from api.loaders import cargar_archivo, ruta_permitida, datos_procesados


class ModeloInferencia:
    def __init__(self, slot, config):
        self.config, self.slot = config, slot
        if slot in ('rnn', 'ann') and not config.scaler:
            raise ValueError('Selecciona el escalador original de entradas del entrenamiento.')
        if slot == 'rnn' and not config.scaler_salida:
            raise ValueError('Selecciona el escalador original de casos de salida.')
        self.scaler = self._scaler(config.scaler, len(config.columnas))
        self.scaler_salida = self._scaler(config.scaler_salida)
        if self.scaler is not None and hasattr(self.scaler, 'feature_names_in_') and list(self.scaler.feature_names_in_) != config.columnas:
            raise ValueError('El orden de columnas no coincide con el escalador.')
        from tensorflow.keras.models import load_model
        path = ruta_permitida(config.archivo)
        self.model = load_model(path, compile=False, safe_mode=True)
        shape, output = self.model.input_shape, self.model.output_shape
        if isinstance(shape, list) or isinstance(output, list):
            raise ValueError('Se admite una entrada y una salida por modelo.')
        if slot == 'cnn':
            from src.Modelo_CNN import CNN_Model_Clasification
            if len(shape) != 4 or any(s is None for s in shape[1:]) or shape[-1] not in (1, 3):
                raise ValueError('La CNN necesita dimensiones fijas (alto, ancho, 1 o 3 canales).')
            internal = any(layer.__class__.__name__ == 'Rescaling' for layer in self.model.layers)
            if config.normalizacion == 'interna' and not internal:
                raise ValueError('No se encontró Rescaling. Confirma la normalización entrenada.')
            if config.normalizacion != 'interna' and internal:
                raise ValueError('El modelo ya incluye Rescaling: selecciona normalización interna.')

            class CNNInferencia(CNN_Model_Clasification):
                def _cargar_imagen(instance, ruta, resize=True):
                    image = super()._cargar_imagen(ruta, resize=resize)
                    return image / 255.0 if config.normalizacion == 'dividir_255' else image

            self.instancia = CNNInferencia(16, shape[1], shape[2], shape[3])
            self.instancia.categories = config.clases
            if config.clase_deteccion not in config.clases:
                raise ValueError('La clase de detección no pertenece a las clases configuradas.')
        elif slot == 'ann':
            from src.Modelo_ANN import Modelo_ANN
            from api.schemas import COLUMNAS_ANN
            if config.columnas != COLUMNAS_ANN:
                raise ValueError('La ANN de acciones necesita las siete columnas del flujo en el orden entrenado.')
            if len(shape) != 2 or shape[-1] != len(config.columnas):
                raise ValueError('Las columnas ANN no coinciden con la entrada del modelo.')
            self.instancia = Modelo_ANN(tipo='multiclase')
            self.instancia.scaler = self.scaler
        else:
            from src.Modelo_RNN_GRU import Simple_RNN_Model, GRU_RNN_Model
            if len(shape) != 3 or shape[1:] != (26, len(config.columnas)) or tuple(output[1:]) != (4,):
                raise ValueError('La RNN debe recibir 26 semanas y devolver cuatro valores.')
            if self.scaler_salida.n_features_in_ != 1:
                raise ValueError('El escalador de salida debe corresponder a casos_dengue.')
            names = getattr(self.scaler_salida, 'feature_names_in_', [config.objetivo])
            if list(names) != [config.objetivo]:
                raise ValueError('El escalador de salida corresponde a otro objetivo.')
            frame = datos_procesados(config.serie)
            # Los checkpoints publicados identifican su ámbito en el nombre.
            # Para artefactos propios se conserva la asociación explícita del usuario.
            if '_nacional' in path.stem.lower() and 'nacional' not in config.serie.lower():
                raise ValueError('El checkpoint nacional requiere la serie nacional.')
            if '_provincial' in path.stem.lower() and 'dengue_semanal' not in config.serie.lower():
                raise ValueError('El checkpoint provincial requiere la serie provincial.')
            required = config.columnas + [config.fecha, config.objetivo] + ([config.grupo] if config.grupo else [])
            if not set(required).issubset(frame.columns):
                raise ValueError('La serie seleccionada no contiene las columnas requeridas.')
            kinds = {layer.__class__.__name__ for layer in self.model.layers}
            if 'GRU' in kinds:
                self.instancia = GRU_RNN_Model()
            elif 'SimpleRNN' in kinds:
                self.instancia = Simple_RNN_Model()
            else:
                raise ValueError('Selecciona un modelo SimpleRNN o GRU del proyecto.')
            self.instancia.scaler_dict = {'x_scaler': self.scaler, 'y_scaler': self.scaler_salida}
        if slot in ('cnn', 'ann') and (len(output) != 2 or output[-1] != len(config.clases) or len(config.clases) < 2):
            raise ValueError('Las clases deben coincidir con el orden y número de salidas.')
        self.instancia.model = self.model
        signature = hashlib.sha256(config.model_dump_json().encode())
        for artifact in [path] + [ruta_permitida(v, 'scaler') for v in (config.scaler, config.scaler_salida) if v]:
            with artifact.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    signature.update(chunk)
        self.identidad = signature.hexdigest()

    @staticmethod
    def _scaler(name, n_columns=None):
        if not name:
            return None
        scaler = cargar_archivo(ruta_permitida(name, 'scaler'), 'scaler')
        check_is_fitted(scaler)
        if not hasattr(scaler, 'transform') or not hasattr(scaler, 'inverse_transform'):
            raise ValueError('El escalador debe admitir transform e inverse_transform.')
        if n_columns is not None and scaler.n_features_in_ != n_columns:
            raise ValueError('El escalador no coincide con la cantidad de entradas.')
        return scaler

    def clasificar(self, path):
        self.instancia.img_prediction(path)
        label = self.instancia.predictions
        return {'especie': label, 'aegypti_detectado': int(label == self.config.clase_deteccion)}

    def recomendar(self, frame):
        self.instancia.make_prediction(data=frame[self.config.columnas], target='recomendacion',
                                       categorias=self.config.clases, graficar=False)
        return {'accion': self.instancia.predictions.iloc[0, 0],
                'probabilidades': self.instancia.probabilidades.iloc[0].to_dict()}

    def ejecutar(self, values):
        output = np.asarray(self.instancia.model(values, training=False))
        if not np.isfinite(output).all():
            raise ValueError('El modelo devolvió valores no finitos.')
        return output
