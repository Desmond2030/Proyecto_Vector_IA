import sys
import os

# Agrega la carpeta raíz del proyecto al path de Python
sys.path.append(os.path.abspath(os.path.join('..')))
import random

import time
import shutil
from PIL import Image


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, PReLU, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dropout
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import classification_report,confusion_matrix, mean_squared_error,mean_absolute_error,explained_variance_score
from sklearn.utils.class_weight import compute_class_weight
from keras.metrics import Precision, Recall, AUC
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from abc import ABC, abstractmethod



from src.Tabular_Data_Processor import TabularDataProcessor


class Model_Base(TabularDataProcessor):
    def __init__(self):
        
        super().__init__()

        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None

        self.test = None
        

        self._partitions: dict = {}
        self._model = Sequential()
        
        self._predictions = None
    


    @property
    def partitions(self):
        return self._partitions

    @property
    def model(self):
        return self._model
    
    @model.setter
    def model(self, model):
        self._model = model
        
    @property
    def predictions(self):
        return self._predictions
    
    
    # 5. GENERAR VENTANAS DE 26 SEMANAS EN 3D Y HORIZONTE DE 4 SEMANAS
    def generar_ventanas_dengue(self, X_data, y_data, tamaño_ventana=26, horizonte=4):
        X_3D, y_2D = [], []
        
        # El límite se ajusta para que quepan la ventana y las 4 semanas a predecir
        for i in range(len(X_data) - tamaño_ventana - horizonte + 1):
            # Entrada: bloque de 26 semanas
            X_3D.append(X_data[i : i + tamaño_ventana])
            # Salida: los siguientes 4 valores consecutivos de casos
            y_2D.append(y_data[i + tamaño_ventana : i + tamaño_ventana + horizonte].flatten())
            
        return np.array(X_3D), np.array(y_2D)
        

    def set_scaled_partitions(self, 
                              save_scalers_path:str = '../models/scalers/',
                              variables_entrada:list=None, 
                              variables_salida:list=None, 
                              col_one_hot:list = None,
                              group_by = "provincia", 
                              tamaño_ventana = 26,
                              horizonte = 4):
        
        
        assert not self._data.isna().any().any(), "Hay valores nulos en el dataset."
        
        assert variables_entrada and variables_salida, 'Falta indicar las variables de entrada y salida'
        
        if col_one_hot is not None:
            assert (len(col_one_hot) > 0) , ' No se puede recorrer una lista vacia'
        
        columnas = [
            col for col in self._data.columns
            if col not in [*variables_entrada, *variables_salida]
        ]
        assert not columnas, f"Columnas no mencionadas: {columnas}"   
        
        
        self._data = self._data.sort_index()
        
        if col_one_hot is not None:
            assert (len(col_one_hot) > 0) , ' No se puede recorrer una lista vacia'
            for col in col_one_hot:

                dummies = pd.get_dummies(
                    self._data[col].groupby(group_by, sort=False),
                    prefix=col,
                    dtype=np.int8
                    )
                
                self._data = pd.concat([self._data, dummies], axis=1)
                

        fechas = self._data.index.unique().sort_values()
        fecha_corte = fechas[int(len(fechas) * 0.80)]
        
        
        train = self._data.loc[self._data.index < fecha_corte].copy()
        val = self._data.loc[self._data.index >= fecha_corte].copy()

        self.x_train = train[variables_entrada]
        self.y_train = train[variables_salida]

        self.x_val = val[variables_entrada]
        self.y_val = val[variables_salida]

        
        
        assert self.x_train.index.max() < self.x_val.index.min(), "¡Fuga de datos! Hay fechas superpuestas."
        
        assert list(self.x_train.columns) == list(self.x_val.columns), "¡Orden de columnas distinto!"


        x_scaler = MinMaxScaler()

        self.x_train = x_scaler.fit_transform(self.x_train.select_dtypes(include='number'))
        self.x_val = x_scaler.transform(self.x_val.select_dtypes(include='number'))
        
        y_scaler = MinMaxScaler()

        self.y_train = y_scaler.fit_transform(self.y_train.select_dtypes(include='number'))
        self.y_val = y_scaler.transform(self.y_val.select_dtypes(include='number'))
        
        
        self.x_train, self.y_train = self.generar_ventanas_dengue(self.x_train, self.y_train, tamaño_ventana=tamaño_ventana, horizonte=horizonte)
        
        self.x_val, self.y_val = self.generar_ventanas_dengue(self.x_val, self.y_val, tamaño_ventana=tamaño_ventana, horizonte=horizonte)
        
        instance_name = type(self).__name__.strip('\'')
        
        self.scaler_dict = {'x_scaler':x_scaler,
                            'y_scaler':y_scaler}

        self.save_load_scaler(str(save_scalers_path)+f'x_scaler_{instance_name}',scaler = x_scaler,  save = True)
        self.save_load_scaler(str(save_scalers_path)+f'y_scaler{instance_name}', scaler = y_scaler,  save = True)
        
        return self
    
    
    
                 
    def set_test_data_as_data(self):
        if self.test is not None:
            self._data = self.test
        else:
            raise ValueError("Sin data para validacion")
        return self


    
    def preparar_experimento(self,semilla:int = 42):
        
        # Fijar la semilla para reproducibilidad
        tf.config.experimental.enable_op_determinism()
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(semilla)
        
        return self
    
    def start_training(self,
                       epochs:int,
                       w_dict = None,
                       batch_size= None,
                       callbacks= []):
    
    

        
        diccionario_pesos = w_dict if w_dict else None

        print("🚀 Iniciando el entrenamiento...\n")
        
        tiempo_inicio = time.time()
        

        self._model.fit(
            x=self.x_train,
            y=self.y_train,
            validation_data=(self.x_val, 
                             self.y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=diccionario_pesos
        )

        tiempo_final = time.time()
        
        tiempo_total_segundos = tiempo_final - tiempo_inicio
        
        # Se conviertr a formato legible (Horas, Minutos, Segundos)
        horas = int(tiempo_total_segundos // 3600)
        minutos = int((tiempo_total_segundos % 3600) // 60)
        segundos = int(tiempo_total_segundos % 60)

        print("\n" + "="*40)
        print(f"⏱️ ENTRENAMIENTO FINALIZADO")
        print(f"Tiempo total: {horas}h {minutos}m {segundos}s ({tiempo_total_segundos:.2f} segundos totales)")
        print("="*40)
        

        return self
    
    
    def evaluar(self, y_scaler = None, x=None, y=None, graficar=True, y_baseline=None):
        
        y_scaler = y_scaler if y_scaler is not None else self.scaler_dict['y_scaler']
        
        """Evalúa un objetivo semanal en casos originales, globalmente y por horizonte.

        x e y son ventanas y objetivos escalados; al omitir ambos se usa validación.
        y_scaler debe estar ajustado con entrenamiento sobre una sola variable.
        y_baseline, si existe, ya está en casos originales y alineado con y.
        Para prueba final, pasar x e y explícitamente. El llamador garantiza el
        orden y la separación temporal: estos arrays no contienen fechas.
        Devuelve DataFrames de métricas, reales y predicciones, sin redondear.
        """
        if (x is None) != (y is None):
            raise ValueError("Debe proporcionar x e y juntos.")
        if x is None:
            x, y = self.x_val, self.y_val
        if x is None or y is None:
            raise ValueError("No hay ventanas u objetivos para evaluar.")

        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        if x.ndim != 3 or y.ndim != 2:
            raise ValueError("\'x\' debe ser 3D (muestras, semanas, variables) e \'y\' 2D (muestras, horizontes).")
        if x.size == 0 or y.size == 0 or x.shape[0] != y.shape[0]:
            raise ValueError("\'x\' e \'y\' deben tener muestras, dimensiones no vacías e igual número de muestras.")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("\'x\' e \'y\' deben contener únicamente valores finitos.")
        if getattr(y_scaler, "n_features_in_", None) != 1:
            raise ValueError("y_scaler debe estar ajustado sobre una sola variable objetivo.")

        if y_baseline is not None:
            y_baseline = np.asarray(y_baseline, dtype=float)
            if y_baseline.shape != y.shape or not np.isfinite(y_baseline).all():
                raise ValueError("y_baseline debe tener la misma forma que \'y\' ademas de valores finitos en casos originales.")

        predicciones = np.asarray(self.model.predict(x, verbose=0), dtype=float)
        if predicciones.shape != y.shape or not np.isfinite(predicciones).all():
            raise ValueError("Las predicciones deben tener la misma forma que \'y\' ademas de valores finitos.")

        # El escalador se ajustó a una variable, no a las H columnas de horizontes.
        try:
            reales = y_scaler.inverse_transform(y.reshape(-1, 1)).reshape(y.shape)
            predicciones = y_scaler.inverse_transform(predicciones.reshape(-1, 1)).reshape(y.shape)
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError("y_scaler no permite desescalar el objetivo de una sola variable.") from error
        if not np.isfinite(reales).all() or not np.isfinite(predicciones).all():
            raise ValueError("El desescalado produjo valores no finitos.")

        nombre_modelo = type(self).__name__
        columnas = [f"semana_{i + 1}" for i in range(y.shape[1])]
        candidatos = [(nombre_modelo, predicciones)]
        if y_baseline is not None:
            candidatos.append(("baseline", y_baseline))

        filas = []
        for nombre, estimaciones in candidatos:
            mae = mean_absolute_error(reales, estimaciones, multioutput="raw_values")
            mse = mean_squared_error(reales, estimaciones, multioutput="raw_values")
            filas.append({"modelo": nombre, "horizonte": "global",
                          "MAE": float(mae.mean()), "RMSE": float(np.sqrt(mse.mean()))})
            for i, horizonte in enumerate(columnas):
                filas.append({"modelo": nombre, "horizonte": horizonte,
                              "MAE": float(mae[i]), "RMSE": float(np.sqrt(mse[i]))})

        self._predictions = pd.DataFrame(predicciones, columns=columnas)
        resultado = {
            "metricas": pd.DataFrame(filas, columns=["modelo", "horizonte", "MAE", "RMSE"]),
            "reales": pd.DataFrame(reales, columns=columnas),
            "predicciones": self._predictions,
        }

        if graficar:
            n_columnas = min(2, len(columnas))
            n_filas = (len(columnas) + n_columnas - 1) // n_columnas
            figura, ejes = plt.subplots(n_filas, n_columnas, squeeze=False,
                                       figsize=(7 * n_columnas, 3.5 * n_filas))
            for i, horizonte in enumerate(columnas):
                eje = ejes.flat[i]
                eje.plot(reales[:, i], label="Observado")
                eje.plot(predicciones[:, i], label="Pronosticado")
                if y_baseline is not None:
                    eje.plot(y_baseline[:, i], label="Baseline", linestyle="--")
                eje.set(title=f"{nombre_modelo} - Semana futura {i + 1}",
                        xlabel="Índice de ventana", ylabel="Casos de dengue")
                eje.legend()
                eje.grid(alpha=0.3)
            for eje in list(ejes.flat)[len(columnas):]:
                eje.set_visible(False)
            figura.tight_layout()
            plt.show()

        
        self.resultado = resultado
        return self


class Simple_RNN_Model(Model_Base):
    def __init__(self):
        super().__init__()
        
        
    def model_flow_settings(self,
                            num_features = None,
                            timesteps:int = 26,
                            fun_act = 'tanh',
                            leakyrelu_negative_slope:float = None,
                            capas_ocultas:int = 2,
                            unidades:int = 64,
                            secuencia:int = 0.5,
                            dropout:float = 0.5,
                            fun_out = 'linear',
                            n_output:int = 4,
                            optimizer='adam',
                            loss='mse',
                            metrics=['mae']
                            ):
        
        self._model = Sequential()
        
        num_features = num_features if num_features is not None else self.x_train.shape[2]
        
        self._model.add(layers.Input(shape=(timesteps, num_features))) 
        
        if fun_act == 'LeakyReLU' and leakyrelu_negative_slope:
            fun_act = keras.layers.LeakyReLU(negative_slope=leakyrelu_negative_slope)
          
        
        while capas_ocultas > 1:
   
            self._model.add(layers.SimpleRNN(unidades, activation=fun_act, return_sequences=True))
            
            unidades = int(unidades * secuencia)
            if unidades < 1 or isinstance(unidades, float):
                raise ValueError("La reducción dejó menos de una unidad o es decimal.")
            capas_ocultas -= 1
            
        self._model.add(layers.SimpleRNN(unidades, activation=fun_act, return_sequences=False))
        
        self._model.add(layers.Dropout(dropout))
        
        self._model.add(layers.Dense(n_output, activation=fun_out))
        
        self._model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        return self
    
    
class GRU_RNN_Model(Model_Base):
    def __init__(self):
        super().__init__()
        
        
    def model_flow_settings(self,
                            num_features = None,
                            timesteps:int = 26,
                            fun_act = 'tanh',
                            leakyrelu_negative_slope:float = None,
                            capas_ocultas:int = 2,
                            unidades:int = 64,
                            secuencia:int = 0.5,
                            dropout:float = 0.5,
                            fun_out = 'linear',
                            n_output:int = 4,
                            optimizer='adam',
                            loss='mse',
                            metrics=['mae']
                            ):
        
        self._model = Sequential()
        
        num_features = num_features if num_features is not None else self.x_train.shape[2]
        
        self._model.add(layers.Input(shape=(timesteps, num_features))) 
        
        if fun_act == 'LeakyReLU' and leakyrelu_negative_slope:
            fun_act = keras.layers.LeakyReLU(negative_slope=leakyrelu_negative_slope)
          
        
        while capas_ocultas > 1:
   
            self._model.add(layers.GRU(unidades, activation=fun_act, return_sequences=True))
            
            unidades = int(unidades * secuencia)
            if unidades < 1 or isinstance(unidades, float):
                raise ValueError("La reducción dejó menos de una unidad o es decimal.")
            capas_ocultas -= 1
            
        self._model.add(layers.GRU(unidades, activation=fun_act, return_sequences=False))
        
        self._model.add(layers.Dropout(dropout))
        
        self._model.add(layers.Dense(n_output, activation=fun_out))
        
        self._model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        return self
    