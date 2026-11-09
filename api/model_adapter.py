# ESTADO_ARCHIVO: EN_MODIFICACION
"""Puentes pequeños: CNN reutiliza src; ANN/RNN cargadas no necesitan entrenamiento."""
import hashlib
import numpy as np
from sklearn.utils.validation import check_is_fitted
from api.loaders import cargar_archivo, ruta_permitida


class ModeloInferencia:
    def __init__(self, slot, config):
        from tensorflow.keras.models import load_model
        self.config, self.slot = config, slot
        path = ruta_permitida(config.archivo)
        self.model = load_model(path, compile=False, safe_mode=True)
        self.scaler = self._scaler(config.scaler, len(config.columnas))
        self.scaler_salida = self._scaler(config.scaler_salida)
        if self.scaler is not None and hasattr(self.scaler, 'feature_names_in_') and list(self.scaler.feature_names_in_) != config.columnas:
            raise ValueError('El orden de columnas no coincide con el scaler.')
        shape, output = self.model.input_shape, self.model.output_shape
        if isinstance(shape, list) or isinstance(output, list):
            raise ValueError('Esta versión admite una entrada y una salida por modelo.')
        if slot == 'cnn':
            from src.Modelo_CNN import CNN_Model_Clasification
            if len(shape) != 4 or any(s is None for s in shape[1:]) or shape[-1] not in (1, 3):
                raise ValueError('La CNN necesita dimensiones fijas (alto, ancho, 1 o 3 canales).')
            internal = any(layer.__class__.__name__ == 'Rescaling' for layer in self.model.layers)
            if config.normalizacion == 'interna' and not internal:
                raise ValueError('No se encontró Rescaling. Confirma el preprocesamiento entrenado.')
            if config.normalizacion != 'interna' and internal:
                raise ValueError('El modelo ya incluye Rescaling: selecciona normalización interna.')

            class CNNInferencia(CNN_Model_Clasification):
                def _cargar_imagen(instance, ruta, resize=True):
                    image = super()._cargar_imagen(ruta, resize=resize)
                    return image / 255.0 if config.normalizacion == 'dividir_255' else image

            self.cnn = CNNInferencia(16, shape[1], shape[2], shape[3])
            self.cnn.model = self.model
            self.cnn.categories = config.clases
        elif slot == 'ann':
            if len(shape) != 2 or shape[-1] != len(config.columnas):
                raise ValueError('Las columnas ANN deben coincidir con la entrada del modelo.')
        else:
            if len(shape) != 3 or shape[-1] != len(config.columnas) or shape[1] not in (None, config.ventana):
                raise ValueError('La RNN debe recibir (lote, ventana, variables) según la configuración.')
            if tuple(output[1:]) not in {(config.horizonte,), (config.horizonte, 1)}:
                raise ValueError('La RNN debe devolver un valor por semana del horizonte configurado.')
        if slot in ('cnn', 'ann'):
            if len(output) != 2 or output[-1] != len(config.clases) or len(config.clases) < 2:
                raise ValueError('Define las clases en el orden exacto de las salidas del clasificador.')
            if slot == 'cnn' and config.clase_deteccion not in config.clases:
                raise ValueError('La clase de detección no pertenece a las clases configuradas.')
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
        self.cnn.img_prediction(path)
        label = self.cnn.predictions
        return {'especie': label, 'aegypti_detectado': int(label == self.config.clase_deteccion)}

    def ejecutar(self, values):
        output = np.asarray(self.model(values, training=False))
        if not np.isfinite(output).all():
            raise ValueError('El modelo devolvió valores no finitos.')
        return output
