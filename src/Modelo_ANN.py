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
from tensorflow.keras import layers
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, PReLU, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import classification_report,confusion_matrix, mean_squared_error,mean_absolute_error,explained_variance_score
from sklearn.utils.class_weight import compute_class_weight
from keras.metrics import Precision, Recall, AUC
import pickle
from abc import ABC, abstractmethod



from src.Tabular_Data_Processor import TabularDataProcessor



class Modelo_ANN(TabularDataProcessor, ABC):
    def __init__(self, tipo):
        
        
        TabularDataProcessor.__init__(self)
        self.tipo = tipo
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

    def set_scaled_partitions(self, 
                              variables_entrada:list=None, 
                              variables_salida:list=None, 
                              tamaño_ventana = 26,
                              horizonte = 4,
                              stratify = True):
        
        
        assert not self._data.isna().any().any(), "Hay valores nulos en el dataset."
        
        assert variables_entrada and variables_salida, 'Falta indicar las variables de entrada y salida'
        
        columnas = [
            col for col in self._data.columns
            if col not in [*variables_entrada, *variables_salida]
        ]
        assert not columnas, f"Columnas no mencionadas: {columnas}"   
        
        
        if stratify:
            raise ValueError("La división temporal requiere stratify=False para conservar juntas las fechas.")
        if not isinstance(self._data.index, pd.DatetimeIndex) or self._data.index.hasnans:
            raise ValueError("Se requiere un DatetimeIndex sin fechas nulas.")

        df = self._data.sort_index(kind='stable')
        fechas = df.index.unique().sort_values()
        corte = int(len(fechas) * 0.80)
        if corte == 0 or corte >= len(fechas):
            raise ValueError("No hay suficientes fechas para separar entrenamiento y validación.")
        fecha_corte = fechas[corte]
        train = df.loc[df.index < fecha_corte]
        val = df.loc[df.index >= fecha_corte]

        self.x_train = train[variables_entrada].copy()
        self.y_train = train[variables_salida].copy()
        self.x_val = val[variables_entrada].copy()
        self.y_val = val[variables_salida].copy()

        self.scaler.fit(self.x_train)
        
        self.x_train = self.scaler.transform(self.x_train)
        self.x_val = self.scaler.transform(self.x_val)
        
        self.save_load_scaler(path ='../models/scalers/ANN_scaler', save = True)

        
        return self
    
                 
    def set_val_data_as_data(self):
        if self.val_file_name:
            self._data = self.partitions[self.val_file_name]
        else:
            raise ValueError("Sin data para validacion")
        return self


    def preparar_experimento(self,semilla:int = 42):
        
        # Fijar la semilla para reproducibilidad
        tf.config.experimental.enable_op_determinism()
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(semilla)
        
        return self




    

    def model_flow_settings(self, 
                            capas:int, 
                            fun_act:str,
                            loss:str,
                            optimizer:str,
                            n_output:int,
                            leakyrelu_negative_slope: float = None,
                            n_down = None,
                            dropout:float= None,
                            fun_out:str= None,
                            metrics = [tf.keras.metrics.AUC(curve='PR', name='prc')]):
        
        self._model = Sequential()
            
        num_neuronas = self.x_train.shape[1]
        
        
         
              
        if fun_act == 'LeakyReLU' and leakyrelu_negative_slope:
            fun_act = keras.layers.LeakyReLU(negative_slope=leakyrelu_negative_slope)
          
          
        self._model.add(layers.Input(shape=(num_neuronas,)))
        
        
        neuron_decrease = num_neuronas
        while capas != 0:
            
            
            
            if n_down and int((neuron_decrease * n_down)) > n_output:
                neuron_decrease = int((neuron_decrease * n_down))
        
            
            if fun_act == 'PReLU' :
                self._model.add(Dense(units=num_neuronas))
                self._model.add(PReLU())
            else:
                self._model.add(Dense(units=neuron_decrease,activation=fun_act))
            
            if dropout and n_down:
                self._model.add(Dropout(dropout * n_down))
            elif dropout:
                self._model.add(Dropout(dropout))
                
            
    
    
            capas -= 1
    
        self._model.add(
            Dense(units=n_output,
                  activation= fun_out if fun_out else fun_act))
        
        self._model.compile(loss=loss, 
                            optimizer=optimizer, 
                            metrics = metrics
                            )
        
        return self
        

    
    def start_training(self,
                       epochs:int,
                       w_balance = True,
                       w_dict = None,
                       batch_size= None,
                       callbacks= []):
    
    
        diccionario_pesos = w_dict
        if w_balance and w_dict is None:
            etiquetas = np.asarray(self.y_train).reshape(-1)
            clases_unicas = np.unique(etiquetas)
            pesos_calculados = compute_class_weight(
                class_weight='balanced', classes=clases_unicas, y=etiquetas,
            )
            diccionario_pesos = dict(zip(clases_unicas, pesos_calculados))

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
        
                
        tiempo_total_segundos = time.time() - tiempo_inicio

        # Se conviertr a formato legible (Horas, Minutos, Segundos)
        horas = int(tiempo_total_segundos // 3600)
        minutos = int((tiempo_total_segundos % 3600) // 60)
        segundos = int(tiempo_total_segundos % 60)


        print("\n" + "="*40)
        print(f"⏱️ ENTRENAMIENTO FINALIZADO")
        print(f"Tiempo total: {horas}h {minutos}m {segundos}s ({tiempo_total_segundos:.2f} segundos totales)")
        print("="*40)
        
        
        return self
    
    
    def make_prediction(self,
                        data=None,
                        target=None,
                        new_data=None,
                        eval=False,
                        *,
                        y=None,
                        categorias=None,
                        umbral=0.8,
                        escalado=False,
                        graficar=True,
                        y_scaler=None):
        """Predice y, con eval=True, evalúa según self.tipo en una sola llamada.

        tipo se indica al instanciar: 'binaria', 'multiclase' o 'regresion'.
        Sin data usa x_val/y_val (entradas ya escaladas). Los datos externos se
        transforman con el escalador existente salvo escalado=True. target indica
        la(s) columna(s) objetivo del DataFrame; también se puede proporcionar y.
        No se modifican los datos originales ni se ajustan escaladores.

        En clasificación, categorias corresponde al orden de las salidas de la
        red y a los valores originales de y; por defecto se usan índices 0..C-1.
        Binaria espera una salida sigmoid y conserva el umbral anterior de 0.8;
        multiclase usa argmax. No se aceptan logits ni etiquetas one-hot.
        En regresión, y_scaler opcional desescala predicciones y objetivos.

        Ejemplos, después de configurar/entrenar o asignar la red:
            modelo = Modelo_ANN(tipo='multiclase')
            modelo.make_prediction(eval=True, categorias=orden_de_clases)
            modelo.make_prediction(data=df, target='recomendacion',
                                   categorias=orden_de_clases, eval=True)
        Devuelve self. predictions contiene un DataFrame; probabilidades y
        resultado contienen las probabilidades y el detalle de esta llamada.
        """
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support

        if self.tipo not in ('binaria', 'multiclase', 'regresion'):
            raise ValueError("tipo debe ser 'binaria', 'multiclase' o 'regresion'.")
        if new_data and data is None:
            raise ValueError('new_data=True requiere proporcionar data.')
        if self.tipo != 'regresion' and y_scaler is not None:
            raise ValueError('y_scaler solo se utiliza en regresión.')
        if self.tipo == 'binaria' and (not np.isfinite(umbral) or not 0 < umbral < 1):
            raise ValueError('umbral debe estar entre 0 y 1.')

        objetivos = [target] if isinstance(target, str) else (list(target) if target is not None else [])
        if len(objetivos) != len(set(objetivos)):
            raise ValueError('target contiene columnas repetidas.')
        externo = data is not None
        inputs = data if externo else self.x_val
        reales = y if y is not None else (self.y_val if not externo else None)
        if inputs is None:
            raise ValueError('Proporcione data o prepare x_val antes de predecir.')
        if isinstance(inputs, pd.DataFrame):
            if not inputs.columns.is_unique:
                raise ValueError('Las entradas contienen columnas duplicadas.')
            if reales is None and objetivos and all(c in inputs for c in objetivos):
                reales = inputs[objetivos]
            indice = inputs.index.copy()
            columnas = getattr(self.scaler, 'feature_names_in_', None)
            if columnas is not None:
                faltantes = [c for c in columnas if c not in inputs]
                if faltantes:
                    raise ValueError(f'Faltan entradas del modelo: {faltantes}.')
                inputs = inputs[list(columnas)]
            else:
                excluir = objetivos + [f'Prediccion_{c}' for c in objetivos] + [f'Prop_{c}' for c in objetivos]
                inputs = inputs.drop(columns=excluir, errors='ignore')
        else:
            indice = reales.index.copy() if not externo and isinstance(reales, (pd.Series, pd.DataFrame)) else None
        if eval and reales is None:
            raise ValueError('Para evaluar se requieren y o las columnas indicadas en target.')
        if isinstance(data, pd.DataFrame) and isinstance(reales, (pd.Series, pd.DataFrame)):
            if not data.index.equals(reales.index):
                raise ValueError('Los índices de data e y deben coincidir en el mismo orden.')
        if isinstance(reales, pd.DataFrame):
            if not reales.columns.is_unique:
                raise ValueError('Los objetivos contienen columnas duplicadas.')
            if objetivos:
                if any(c not in reales for c in objetivos):
                    raise ValueError('Faltan columnas de target en y.')
                reales = reales[objetivos]
            else:
                objetivos = list(reales.columns)
        elif isinstance(reales, pd.Series) and not objetivos and reales.name is not None:
            objetivos = [reales.name]

        valores = np.asarray(inputs)
        if valores.ndim != 2 or 0 in valores.shape or valores.dtype.kind not in 'biuf':
            raise ValueError('Las entradas deben ser una matriz numérica 2D no vacía.')
        valores = valores.astype(float)
        if not np.isfinite(valores).all():
            raise ValueError('Las entradas contienen NaN o infinitos.')
        if externo and not escalado:
            columnas = getattr(self.scaler, 'feature_names_in_', None)
            if columnas is not None:
                if len(columnas) != valores.shape[1]:
                    raise ValueError('El número de entradas no coincide con el escalador.')
                valores = self.scaler.transform(pd.DataFrame(valores, columns=columnas))
            else:
                valores = self.scaler.transform(valores)
        if not np.isfinite(valores).all():
            raise ValueError('El escalado produjo valores no finitos.')
        indice = pd.RangeIndex(len(valores)) if indice is None else indice
        if len(indice) != len(valores):
            raise ValueError('El índice no coincide con el número de entradas.')

        salidas = np.asarray(self.model.predict(valores, verbose=0), dtype=float)
        if salidas.ndim == 1:
            salidas = salidas[:, None]
        if salidas.ndim != 2 or salidas.shape[0] != len(valores) or salidas.shape[1] == 0:
            raise ValueError('La forma de las predicciones no coincide con las entradas.')
        if not np.isfinite(salidas).all():
            raise ValueError('Las predicciones contienen NaN o infinitos.')
        probabilidades = None
        resultado = {}

        if self.tipo == 'regresion':
            nombres = objetivos or [f'objetivo_{i + 1}' for i in range(salidas.shape[1])]
            if len(nombres) != salidas.shape[1]:
                raise ValueError('target no coincide con el número de salidas de regresión.')
            if eval:
                reales = np.asarray(reales, dtype=float)
                if reales.ndim == 1:
                    reales = reales[:, None]
                if reales.shape != salidas.shape or not np.isfinite(reales).all():
                    raise ValueError('Los objetivos deben tener la misma forma y valores finitos que las predicciones.')
            if y_scaler is not None:
                if getattr(y_scaler, 'n_features_in_', None) != salidas.shape[1]:
                    raise ValueError('y_scaler debe estar ajustado para todas las columnas objetivo.')
                salidas = np.asarray(y_scaler.inverse_transform(salidas), dtype=float)
                if eval:
                    reales = np.asarray(y_scaler.inverse_transform(reales), dtype=float)
                if not np.isfinite(salidas).all() or (eval and not np.isfinite(reales).all()):
                    raise ValueError('El desescalado produjo valores no finitos.')
            predicciones = pd.DataFrame(salidas, index=indice, columns=nombres)
            if eval:
                mae = mean_absolute_error(reales, salidas, multioutput='raw_values')
                mse = mean_squared_error(reales, salidas, multioutput='raw_values')
                filas = [{'alcance': 'global', 'objetivo': None,
                          'MAE': float(mae.mean()), 'RMSE': float(np.sqrt(mse.mean()))}]
                filas += [{'alcance': 'objetivo', 'objetivo': col,
                           'MAE': float(mae[i]), 'RMSE': float(np.sqrt(mse[i]))}
                          for i, col in enumerate(nombres)]
                resultado['metricas'] = pd.DataFrame(filas)
                resultado['reales'] = pd.DataFrame(reales, index=indice, columns=nombres)
        else:
            if self.tipo == 'binaria':
                if salidas.shape[1] != 1:
                    raise ValueError('Binaria requiere una salida sigmoid; para softmax utilice multiclase.')
                probabilidades_array = np.column_stack([1 - salidas[:, 0], salidas[:, 0]])
                codigos = (salidas[:, 0] > umbral).astype(int)
            else:
                if salidas.shape[1] < 2:
                    raise ValueError('Multiclase requiere al menos dos salidas.')
                probabilidades_array = salidas
                codigos = salidas.argmax(axis=1)
            if (np.any(probabilidades_array < 0) or np.any(probabilidades_array > 1)
                    or not np.allclose(probabilidades_array.sum(axis=1), 1, atol=1e-5, rtol=0)):
                raise ValueError('La clasificación requiere probabilidades válidas, no logits.')
            n_clases = probabilidades_array.shape[1]
            if isinstance(categorias, (str, bytes)):
                raise ValueError('categorias debe ser una lista en el orden de las salidas.')
            categorias = list(range(n_clases)) if categorias is None else list(categorias)
            etiquetas = pd.Index(categorias)
            if len(categorias) != n_clases or etiquetas.has_duplicates or etiquetas.isna().any():
                raise ValueError('categorias debe contener una etiqueta única y no nula por clase.')
            if len(objetivos) > 1:
                raise ValueError('Clasificación requiere una sola columna objetivo.')
            nombre = objetivos[0] if objetivos else 'clase'
            predicciones = pd.DataFrame({nombre: np.asarray(categorias, dtype=object)[codigos]}, index=indice)
            probabilidades = pd.DataFrame(probabilidades_array, index=indice, columns=categorias)
            if eval:
                reales_array = np.asarray(reales)
                if reales_array.ndim == 2 and reales_array.shape[1] == 1:
                    reales_array = reales_array[:, 0]
                if reales_array.ndim != 1 or len(reales_array) != len(valores):
                    raise ValueError('Las etiquetas deben ser un vector de la misma longitud que las entradas.')
                reales_codificados = pd.Series(reales_array).map({c: i for i, c in enumerate(categorias)})
                if reales_codificados.isna().any():
                    raise ValueError('Hay etiquetas desconocidas o nulas; proporcione categorias en el orden de salida.')
                y_true = reales_codificados.to_numpy(dtype=int)
                labels = np.arange(n_clases)
                precision, recall, f1, soporte = precision_recall_fscore_support(
                    y_true, codigos, labels=labels, zero_division=0)
                resultado['reporte_clasificacion'] = pd.DataFrame({
                    'clase': categorias, 'precision': precision, 'recall': recall,
                    'f1': f1, 'support': soporte,
                })
                resultado['matriz_confusion'] = pd.DataFrame(
                    confusion_matrix(y_true, codigos, labels=labels),
                    index=pd.Index(categorias, name='Real'), columns=pd.Index(categorias, name='Predicción'))
                filas = []
                for promedio in ('macro', 'weighted'):
                    precision, recall, f1, _ = precision_recall_fscore_support(
                        y_true, codigos, labels=labels, average=promedio, zero_division=0)
                    filas.append({'promedio': promedio, 'accuracy': accuracy_score(y_true, codigos),
                                  'precision': precision, 'recall': recall, 'f1': f1, 'support': len(y_true)})
                resultado['metricas'] = pd.DataFrame(filas)
                resultado['reales'] = pd.DataFrame({nombre: reales_array}, index=indice)

        resultado['predicciones'] = predicciones
        if probabilidades is not None:
            resultado['probabilidades'] = probabilidades
        self._predictions = predicciones
        self.probabilidades = probabilidades
        self.resultado = resultado

        if eval:
            print(resultado['metricas'].to_string(index=False))
            if self.tipo != 'regresion':
                print(resultado['reporte_clasificacion'].to_string(index=False))
            if graficar:
                if self.tipo == 'regresion':
                    fig, ejes = plt.subplots(1, len(predicciones.columns), squeeze=False,
                                             figsize=(5 * len(predicciones.columns), 4))
                    for i, nombre in enumerate(predicciones.columns):
                        real = resultado['reales'].iloc[:, i]
                        pred = predicciones.iloc[:, i]
                        minimo, maximo = min(real.min(), pred.min()), max(real.max(), pred.max())
                        if minimo == maximo:
                            minimo, maximo = minimo - 0.5, maximo + 0.5
                        ejes[0, i].scatter(real, pred)
                        ejes[0, i].plot([minimo, maximo], [minimo, maximo], 'r--')
                        ejes[0, i].set(title=str(nombre), xlabel='Real', ylabel='Predicción')
                else:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(resultado['matriz_confusion'], annot=True, fmt='d', cmap='Blues', ax=ax)
                    ax.set_title('Matriz de confusión')
                fig.tight_layout()
                plt.show()
        return self
