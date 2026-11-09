import sys
import os
from pathlib import Path
import time
import shutil
from PIL import Image


# Agrega la carpeta raíz del proyecto al path de Python
sys.path.append(os.path.abspath(os.path.join('..')))
import random

from typing import Literal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, PReLU, Dropout, Input, Flatten, StringLookup, TextVectorization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report,confusion_matrix, mean_squared_error,mean_absolute_error,explained_variance_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.metrics import Precision, Recall, AUC
import pickle
from abc import ABC, abstractmethod



from src.Tabular_Data_Processor import TabularDataProcessor


class Vision_Model_Procesor:
    def __init__(self, 
                 batch_size = 16,
                 altura_pxl = 550,
                 ancho_pxl = 550,
                 canales = 3
                 ):
        
        # Rutas
        self.train_data_folder = None
        self.test_data_folder = None
        self.val_data_folder = None
        
        
        # Data
        self.train_data = None 
        self.val_data = None     
        self.test_data = None
        self.df_img = None
         
        # Configuraciones generales de instancia
        self.altura_pxl = altura_pxl
        self.ancho_pxl = ancho_pxl
        self.batch_size = batch_size
        self.canales = canales
        self.buffer_size = None
        self.labels = None
        self.lab_mode = None
        self.color_mode = 'rgb' if self.canales==3 else 'grayscale'
        
        # Etiquetas
        self.categories:int = None
        
        # Tokenizador
        self._text_tokenizer = None
        self.corpus_data_folder = None
        self.max_vocabulary = 5000
        self.phrase_length = 20
        self.standardize = 'lower_and_strip_punctuation'
        self.split = 'whitespace'
        self.tkn_output = 'int'
        self.vocabulary = None

        # Internos
        self._dataset_final = None
        
        self._predictions = None
        
        self._model = Sequential()
        self._losses = None
        
        self._img_paths = None
        
        # Componentes
        self._tab_data_processor = TabularDataProcessor()
    


    @property
    def model(self):
        return self._model
    
    @model.setter
    def model(self, model):
        self._model = model
        
    @property
    def text_tokenizer(self):
        return self._text_tokenizer
    
    @property
    def losses(self):
        return self._losses
        
    @property
    def tab_data_processor(self):
        return self._tab_data_processor
    
    @tab_data_processor.setter
    def tab_data_processor(self, TabularDataProcessor):
        self._tab_data_processor = TabularDataProcessor
        
    @property
    def predictions(self):
        return self._predictions
    
    @property
    def img_paths(self):
        return self._img_paths
    
    @img_paths.setter
    def img_paths(self, img_paths):
        self._img_paths = img_paths
    
    def set_tokenizer_config(self, txt_xten = '*.txt'):
        
        
        # Capa para tokenizar textos
        self._text_tokenizer = TextVectorization(
            max_tokens=self.max_vocabulary,          # Tamaño máximo de tu diccionario
            standardize = self.standardize,
            output_mode=self.tkn_output,        # Como devuelve el texto
            split = self.split,
            output_sequence_length=self.phrase_length  # Fuerza a que todas las frases midan exactamente los mismos tokens (hace Padding)
        )
        
        self.set_corpus(txt_xten = txt_xten)
        
        return self
    
    
    # El tokenizador necesita "aprender" las palabras del dataset
    # Se extrae solo el texto para el entrenamiento de la capa 
        
    def set_corpus(self, txt_xten = '*.txt'):
        
        if self._text_tokenizer is None:
            self.set_tokenizer_config(txt_xten = txt_xten)
            return self
            
        self.vocabulary = []
        
        corpus_paths = sorted(str(path) for path in Path(self.corpus_data_folder).glob(txt_xten))
        
        
        for f in corpus_paths:
        
            txt_bytes = tf.io.read_file(f)
        
            texto = tf.strings.strip(txt_bytes) 
            
            self.vocabulary.append(texto)
            
        self._text_tokenizer.adapt(self.vocabulary)
        
        return self
            
        
            
            
    
    
    
    # Para entrenaminemto
    def classify_img_by_folder(self):
        
        assert self.train_data_folder or self.test_data_folder, "Debe de existir un directorio para entrenamiento o prueba"
        
        
        if self.train_data_folder:
            self.train_data = tf.keras.utils.image_dataset_from_directory(
                self.train_data_folder,
                labels= self.labels if self.labels else 'inferred',        # Infiere automáticamente basándose en los nombres de las carpetas
                label_mode= self.lab_mode if self.lab_mode else 'int',       # Como se tratan las categorias  
                batch_size=self.batch_size,
                image_size=(self.altura_pxl, self.ancho_pxl),
                color_mode= self.color_mode,
                shuffle=True, # Baraja las imagenes
                seed=42,
                validation_split= None if self.val_data_folder  else 0.2,
                subset=None if self.val_data_folder  else "training"
            )
            

            self.val_data = tf.keras.utils.image_dataset_from_directory(
                self.val_data_folder if self.val_data_folder else self.train_data_folder,
                label_mode= self.lab_mode if self.lab_mode else 'int',   
                batch_size=self.batch_size,
                image_size=(self.altura_pxl, self.ancho_pxl),
                color_mode= self.color_mode,
                shuffle=True,
                seed=42,
                validation_split= None if self.val_data_folder  else 0.2,
                subset= None if self.val_data_folder  else "validation"
            )
            
            
        if self.test_data_folder:
            self.test_data = tf.keras.utils.image_dataset_from_directory(
                self.test_data_folder,
                labels= self.labels if self.labels else 'inferred',        # Infiere automáticamente basándose en los nombres de las carpetas
                label_mode= self.lab_mode if self.lab_mode else 'int',       # Como se tratan las categorias  
                batch_size=self.batch_size,
                image_size=(self.altura_pxl, self.ancho_pxl),
                color_mode= self.color_mode,
                shuffle=True, # Baraja las imagenes
                seed=42
            )
            

        print(f'Categorias de instancia: {self.categories}')

        if self.train_data:    
            print(f'Train data: {self.train_data.class_names}')
            print(f'Validation data: {self.val_data.class_names}')
            
            assert self.train_data.class_names == self.val_data.class_names == self.categories, \
                "Las clases definidas y las de los conjuntos train y val no coinciden"

            
        if self.test_data:
            print(f'Test data: {self.test_data.class_names}')
            
            assert self.test_data.class_names == self.categories, \
                "Las clases definidas y las del conjunto test no coinciden"
        

        
        return self

            

    def _cargar_imagen(self, ruta, resize = True):
        
        # 1. Leer el archivo binario desde el almacenamiento local
        contenido = tf.io.read_file(ruta)

        # 2. Decodificar los bytes a un Tensor de imagen (JPEG o PNG)
        imagen = tf.image.decode_image(
            contenido,
            channels=self.canales,
            expand_animations=False,
        )
        

        imagen.set_shape([None, None, self.canales])
        
        if not resize:
            return imagen

        # 3. Normalizar o redimensionar según los requisitos del modelo     
        return tf.image.resize(
            imagen,
            [self.altura_pxl, self.ancho_pxl],
            method="bilinear",
            antialias=False
        )
        
                
            
    #flujo tentativo, falta tokenizar el texto
    def procesar_img_data(self, ruta_img, etiqueta, one_hot_category = None, ruta_txt = None):
        
        # 1. Leer el archivo binario desde el almacenamiento local
        # 2. Decodificar los bytes a un Tensor de imagen (JPEG o PNG)
        # 3. Normalizar o redimensionar según los requisitos de tu modelo convolucional o de difusión
        img = self._cargar_imagen(ruta_img)
        
        
        # ---- PROCESAMIENTO DEL TEXTO (ETIQUETA) ----
        if ruta_txt:
            txt_bytes = tf.io.read_file(etiqueta)
        
            # tf.strings.strip remueve saltos de línea (\n) o espacios en blanco extra
            etiqueta_texto = tf.strings.strip(txt_bytes) 
            
            token_vector = self._text_tokenizer(etiqueta_texto)

            return img, token_vector
        
        elif one_hot_category:
            
            if self.categories:
                
                etiqueta_one_hot = tf.one_hot(etiqueta, depth=len(self.categories))
                
                return img, etiqueta_one_hot
            else:
                raise ValueError('Falta definir las categorias')
        
        else:
            return img, etiqueta
    
    
    def crear_pd_tf_df(self, rutas, etiquetas, one_hot_category = None, ruta_txt=None):
        
        
        
        tf_data = tf.data.Dataset.from_tensor_slices((rutas, etiquetas)) 
        
        # Configurar el almacenamiento en caché, mezclado aleatorio 
        tf_data = tf_data.shuffle(buffer_size=self.buffer_size if self.buffer_size else 100)
        
        # Aplicar la función a todo el flujo de datos de forma paralela (Optimización por hilos) y agrupación por lotes (Batching)
        tf_data = tf_data.map(lambda ruta, etiqueta: self.procesar_img_data(ruta, 
                                                                            etiqueta, 
                                                                            one_hot_category= one_hot_category, 
                                                                            ruta_txt=ruta_txt), 
                              num_parallel_calls=tf.data.AUTOTUNE).batch(self.batch_size)


        pd_dataset = pd.DataFrame({'Rutas': rutas,
                                    'Etiquetas': etiquetas})
            
        return {'tf_dataset':tf_data, 'pd_dataset':pd_dataset}
    

    
      
    def img_path_x_label_col(self, 
                                indx_file_name_col, 
                                indx_label_col, 
                                indx_train = None, 
                                indx_val = None,
                                one_hot_category = True, 
                                test=None):

        assert self.train_data_folder or self.test_data_folder, "Debe de existir un directorio para entrenamiento o prueba"
        
        
        if self.train_data_folder and self.val_data_folder and test is None:
            
            if indx_train is not None and len(indx_train) > 0:
                rutas_train = [os.path.join(self.train_data_folder, f) for f in self._tab_data_processor.data.iloc[int(indx_train), int(indx_file_name_col)]]
                
                etiquetas_train = self._tab_data_processor.data.iloc[int(indx_train), int(indx_label_col)]
                
                self.train_data = self.crear_pd_tf_df(rutas_train, etiquetas_train, one_hot_category = one_hot_category)
                
            else:
                raise ValueError('Falta indicar indices para data de entrenamineto')
                
            
            if indx_val is not None and len(indx_val) > 0:
                rutas_val = [os.path.join(self.val_data_folder, f) for f in self._tab_data_processor.data.iloc[int(indx_val), int(indx_file_name_col)]]
                etiquetas_val = self._tab_data_processor.data.iloc[int(indx_val), int(indx_label_col)]
                
                self.val_data = self.crear_pd_tf_df(rutas_val, etiquetas_val, one_hot_category = one_hot_category)
            else:
                raise ValueError('Falta indicar indices para data de validacion')
            
            
        
        elif self.train_data_folder and not self.val_data_folder and test is None:
            
            rutas = [os.path.join(self.train_data_folder, f) for f in self._tab_data_processor.data.iloc[:,int(indx_file_name_col)]]
            
            etiquetas = self._tab_data_processor.data.iloc[:,int(indx_label_col)]
            
            paths_train, paths_val, labels_train, labels_val = train_test_split(
                rutas, 
                etiquetas, 
                test_size=0.20,        # Porcentaje para prueba 
                stratify=etiquetas, # para mantener el balanceo de los datos en la columna 'Abandono'
                random_state=42 
                )
            
            self.train_data = self.crear_pd_tf_df(paths_train, labels_train, one_hot_category = one_hot_category)
            
            self.val_data = self.crear_pd_tf_df(paths_val, labels_val, one_hot_category = one_hot_category)
            
            
            
        if self.test_data_folder and test:
            
            rutas = [os.path.join(self.test_data_folder, f) for f in self._tab_data_processor.data.iloc[:,int(indx_file_name_col)]]
            
            etiquetas = self._tab_data_processor.data.iloc[:,int(indx_label_col)]
            
            self.test_data = self.crear_pd_tf_df(rutas, etiquetas, one_hot_category = one_hot_category)
            

        return self
    
    
    
    
    
    def img_folder_x_text_folder(self, 
                                 img_xten:list = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG'], 
                                 txt_xten:list = ['*.txt'],
                                 exept_xten:tuple= ('.Zone.Identifier',),
                                 one_hot_category = None, 
                                 test=None):
        
        assert self.train_data_folder or self.test_data_folder, "Debe de existir un directorio para entrenamiento o prueba"
        
        assert self._text_tokenizer != None, "Se debe de configurar un tokenizador para procesar el texto"
        
        
        
        if self.train_data_folder and self.val_data_folder and test is None:
            
            rutas_img_train, rutas_imag_val, rutas_etiquetas_train, rutas_etiquetas_val = [], [], [], []
            
            # Obtener y ordenar las rutas de todas las imágenes
            for patron_img in img_xten:

                rutas_img_train.extend(str(ruta) for ruta in Path(self.train_data_folder).glob(patron_img)
                                    if not Path(ruta).name.endswith(exept_xten))
                
                
                rutas_imag_val.extend(str(ruta) for ruta in Path(self.val_data_folder).glob(patron_img)
                                    if not Path(ruta).name.endswith(exept_xten))

            # Obtener y ordenar las rutas de todas las etiquetas de texto
            for patron_txt in txt_xten:
                rutas_etiquetas_train.extend(str(ruta) for ruta in Path(self.train_data_folder).glob(patron_txt)
                                            if not Path(ruta).name.endswith(exept_xten))
                rutas_etiquetas_val.extend(str(ruta) for ruta in Path(self.val_data_folder).glob(patron_txt)
                                            if not Path(ruta).name.endswith(exept_xten))

            rutas_img_train = sorted(set(rutas_img_train)) # Eliminar duplicados y ordena
            rutas_imag_val = sorted(set(rutas_imag_val))  # Eliminar duplicados y ordena
            
            
            rutas_etiquetas_train = sorted(set(rutas_etiquetas_train))  
            rutas_etiquetas_val = sorted(set(rutas_etiquetas_val))  

            # Validación de seguridad: Asegurar que cada imagen tenga su contraparte de texto
            assert len(rutas_img_train) == len(rutas_etiquetas_train), "¡La cantidad de imágenes y archivos .txt para entrenamiento no coincide!"
            assert len(rutas_imag_val) == len(rutas_etiquetas_val), "¡La cantidad de imágenes y archivos .txt para validacion no coincide!"
            
            
            
            self.train_data = self.crear_pd_tf_df(rutas_img_train, 
                                                  rutas_etiquetas_train,
                                                  one_hot_category = one_hot_category, 
                                                  ruta_txt=True)
        

            self.val_data = self.crear_pd_tf_df(rutas_imag_val, 
                                                rutas_etiquetas_val, 
                                                one_hot_category = one_hot_category,
                                                ruta_txt=True)
            
            
        elif self.train_data_folder and not self.val_data_folder and test is None:
            
            rutas_imagenes, rutas_etiquetas = [], []
            
            for patron_img in img_xten:
                rutas_imagenes.extend(str(ruta) for ruta in Path(self.train_data_folder).glob(patron_img)
                                        if not Path(ruta).name.endswith(exept_xten))
                rutas_imagenes = sorted(set(rutas_imagenes))  
            

            for patron_txt in txt_xten:
                rutas_etiquetas.extend(str(ruta) for ruta in Path(self.train_data_folder).glob(patron_txt)
                                        if not Path(ruta).name.endswith(exept_xten))
                
            rutas_imagenes = sorted(set(rutas_imagenes))
            rutas_etiquetas = sorted(set(rutas_etiquetas))  
                
            assert len(rutas_imagenes) == len(rutas_etiquetas), "¡La cantidad de imágenes y archivos .txt no coincide!"
            
            
            paths_train, paths_val, labels_train, labels_val = train_test_split(
                rutas_imagenes, 
                rutas_etiquetas, 
                test_size=0.20,        # Porcentaje para prueba                 
                random_state=42 
                )
            
            self.train_data = self.crear_pd_tf_df(paths_train, 
                                                  labels_train, 
                                                  one_hot_category = one_hot_category,
                                                  ruta_txt=True)
            
            self.val_data = self.crear_pd_tf_df(paths_val, 
                                                labels_val,
                                                one_hot_category = one_hot_category, 
                                                ruta_txt=True)
        
        
        
        if self.test_data_folder and test :
            
            rutas_imagenes, rutas_etiquetas = [], []
            
            
            for patron_img in img_xten:
                rutas_imagenes.extend(str(ruta) for ruta in Path(self.test_data_folder).glob(patron_img)
                                        if not Path(ruta).name.endswith(exept_xten))  # Ignorar archivos mencionados en exept_xten

                 

            for patron_txt in txt_xten:
                rutas_etiquetas.extend(str(ruta) for ruta in Path(self.test_data_folder).glob(patron_txt)
                                        if not Path(ruta).name.endswith(exept_xten))
                
            rutas_imagenes = sorted(set(rutas_imagenes))
            rutas_etiquetas = sorted(set(rutas_etiquetas)) 
            
            assert len(rutas_imagenes) == len(rutas_etiquetas), "¡La cantidad de imágenes y archivos .txt no coincide!"
            
            self.test_data = self.crear_pd_tf_df(rutas_imagenes, 
                                                 rutas_etiquetas,
                                                 one_hot_category = one_hot_category, 
                                                 ruta_txt=True)
            

        return self
    
    
    def show_sample_image_x_label(self, data_from:Literal['train', 'val', 'test'] = 'train', num_lote:int=1):
        
        if data_from == 'train':
            if isinstance(self.train_data, dict):
                tf_dataset = self.train_data['tf_dataset']
            else:
                tf_dataset = self.train_data
        elif data_from == 'val':
            if isinstance(self.val_data, dict):
                tf_dataset = self.val_data['tf_dataset']
            else:
                tf_dataset = self.val_data
            
        elif data_from == 'test':
            if isinstance(self.test_data, dict):
                tf_dataset = self.test_data['tf_dataset']
            else:
                tf_dataset = self.test_data
        else:
            raise ValueError('Se debe de indicar de donde saldra el lote a visualizar')
        
        assert isinstance(tf_dataset, tf.data.Dataset), 'El flujo de datos no es un tf.data.Dataset'
        assert isinstance(num_lote, int), 'El indicador del lote a seleccionar no es de tipo \'int\''
        
        print("Clases detectadas:", self.categories)

        plt.figure(figsize=(10, 10))       
        
        # Se extrae un lote para graficar las primeras 9 imágenes de este
        for lote_imagenes, lote_etiquetas in tf_dataset.take(num_lote):
            
            cantidad = int(lote_imagenes.shape[0])
            indices = random.sample(range(cantidad), k=min(9, cantidad))
            
            for i, indice in enumerate(indices):
                ax = plt.subplot(3, 3, i + 1)
                img_rdm_indx = random.randint(0, self.batch_size - 1)
                
                # Keras lee las imágenes como float32 o int32. 
                # Las convertimos a enteros de 8 bits (uint8) para que matplotlib las renderice correctamente
                plt.imshow(lote_imagenes[indice].numpy().astype("uint8"))
                
                # Se busca el nombre de texto de la clase usando el índice numérico
                id_clase = int(lote_etiquetas[indice].numpy())
                plt.title(self.categories[id_clase])
                plt.axis("off")
                
        plt.show()
        
        return self
    
    
    def set_df_img(self, 
                      path:str, 
                      patron:list = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG'], 
                      exept_patron:tuple= ('.Zone.Identifier',),
                      subcarpetas:list = None, 
                      just_paths:bool = False):

        path = Path(path)
        columnas = ["Ruta_Imagen","Alto", "Ancho", "Canal", "Modo"]
        dict_df = {}
        
        
        
        if not subcarpetas:
            subcarpetas = [sub.name for sub in path.iterdir() 
                        if sub.is_dir() and not sub.name.startswith('.')
                        ]
        print(f'Subcarpetas encontradas: {subcarpetas}')
            
        if just_paths:
            
            for carpeta in subcarpetas:
                rutas = []
                
                for xten in patron:
                    ruta = path.joinpath(carpeta)
                    rutas.extend(str(archivo) for archivo in ruta.glob(xten)
                                        if archivo.is_file() 
                                        and not archivo.name.endswith(exept_patron))
                    
                    dict_df[carpeta] = sorted(set(rutas))
                
            tab_procesor = TabularDataProcessor()
            tab_procesor.data = pd.DataFrame.from_dict(dict_df, orient='index').T
            
            self._img_paths = tab_procesor
            
            self._img_paths.set_info()
            
            return self
   
        for carpeta in subcarpetas:
            
            rutas = []
            
            tab_procesor = TabularDataProcessor()
            
            for xten in patron:
                ruta = path.joinpath(carpeta)
                rutas.extend(str(archivo) for archivo in ruta.glob(xten)
                                    if archivo.is_file() and not archivo.name.endswith(exept_patron))

                rutas = sorted(set(rutas))
                
            print(f'Cantidad de archivos en {carpeta}: {len(rutas)}')
            

            filas_acumuladas = []
            
            for ruta_imagen in rutas:
                
                with Image.open(ruta_imagen) as img:
                    ancho, alto = img.size  # Pillow devuelve ancho primero
                    canales = len(img.getbands())
                    modo = img.mode
                    
                    
                filas_acumuladas.append({
                    "Ruta_Imagen" : ruta_imagen,
                    "Alto": alto,
                    "Ancho": ancho,
                    "Canal": canales,
                    "Modo": modo
                })
                
                df = pd.DataFrame(filas_acumuladas, columns=columnas)
                
                tab_procesor.data = df
                
                dict_df[carpeta] = tab_procesor
                
        self.df_img = dict_df
         
        return self


    def move_files_in_img_paths(self, dest_folder:str, col_paths:str, col_labels:str):
        
        assert self._img_paths is not None, "No hay rutas de imágenes para mover"
        
        nuevas_rutas = []
        
        for ruta_img, etiqueta in zip(self._img_paths.data[col_paths], self._img_paths.data[col_labels]):
            dest_path = Path(dest_folder) / etiqueta
            
            if not dest_path.exists():
                dest_path.mkdir(parents=True, exist_ok=True)
                
            destino = dest_path / Path(ruta_img).name
                
            shutil.move(ruta_img, destino)
            
            nuevas_rutas.append(str(destino))
            
        self.img_paths.data[col_paths] = nuevas_rutas
    
        print(f"Archivos movidos a {dest_folder}")
        
        return self


    def preparar_experimento(self,semilla:int = 42):
        
        # Fijar la semilla para reproducibilidad
        tf.config.experimental.enable_op_determinism()
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(semilla)
        
        return self





    







class CNN_Model_Clasification(Vision_Model_Procesor):
    def __init__(self, 
                 batch_size,
                 altura_pxl,
                 ancho_pxl,
                 canales):
        
        super().__init__(
                        batch_size,
                        altura_pxl,
                        ancho_pxl,
                        canales)


    def model_flow_settings(self,
                            capas_conv,
                            transition:Literal['flatten', 'GAP'],
                            capas_densas:int, 
                            fun_act:str,
                            loss:str,
                            optimizer:str,
                            n_output:int,
                            random_flip = "horizontal",
                            random_rotation = 0.2,
                            n_filtros: int = 32,
                            n_filter_up_escale = 2,
                            kernel_size:tuple = (3,3),
                            pool_layer: Literal['max', 'avg'] = 'max',
                            pool_size:tuple = (2,2),
                            padding = 'same',
                            leakyrelu_negative_slope: float | None = None,
                            n_dense_keep = None,
                            dropout:float= None,
                            fun_out:str= None,
                            metrics:list = None):
        
        self._model = Sequential()
        transition = str(transition).lower()
        pool_layer = str(pool_layer).lower()
        
        # Verificacion para funcion de activacion     
        if fun_act == 'LeakyReLU' and leakyrelu_negative_slope:
            fun_act = layers.LeakyReLU(negative_slope=leakyrelu_negative_slope)
          
        
        # Añadir capas de preprocesamiento directamente sobre la entrada
        self._model.add(layers.Rescaling(1./255,  input_shape=(self.altura_pxl, self.ancho_pxl, self.canales)))
        if random_flip:
            self._model.add(layers.RandomFlip(random_flip))
            
        if random_rotation:
            self._model.add(tf.keras.layers.RandomRotation(random_rotation))
        
        
        
        # Capas de convolucion
        for _ in range(capas_conv):
            

            self._model.add(layers.Conv2D(n_filtros, kernel_size=kernel_size, padding=padding))
            self._model.add(layers.BatchNormalization())
            
            
            if fun_act == 'PReLU':
                self._model.add(PReLU()) 
                
            elif isinstance(fun_act, layers.LeakyReLU):
                self._model.add(layers.LeakyReLU(negative_slope=leakyrelu_negative_slope))
             
            else:
                self._model.add(layers.Activation(fun_act))
                
                
                
            if pool_layer == 'max':
                self._model.add(layers.MaxPooling2D(pool_size = pool_size))
                    
            elif pool_layer == 'avg':                    
                self._model.add(layers.AveragePooling2D(pool_size = pool_size))
            else:
                raise ValueError('El parametro \"pool_layer\" debe de ser \'max\' o \'avg\'')
                
                
        
            if n_filter_up_escale:
                n_filtros = n_filtros*n_filter_up_escale

        
        # Definicion de bloque de trancicion
        if transition == "gap":
            self._model.add(tf.keras.layers.GlobalAveragePooling2D())
                
        elif transition == 'flatten':
            self._model.add(layers.Flatten())
            
        else:
            raise ValueError('El parametro \"transition\" debe de ser \'gap\' o \'flatten\'')
            

        remaining_neurons = self._model.output_shape[1]
        
        
        # Capas densas
        for _ in range(capas_densas):
            
            
            if n_dense_keep and int((remaining_neurons * n_dense_keep)) > n_output*1.1:
                remaining_neurons = int((remaining_neurons * n_dense_keep))
        
            
            if fun_act == 'PReLU' :
                self._model.add(Dense(units=remaining_neurons))
                self._model.add(PReLU())
            else:
                self._model.add(Dense(units=remaining_neurons,activation=fun_act))
            
            if dropout and n_dense_keep:
                self._model.add(Dropout(dropout * n_dense_keep))
            elif dropout:
                self._model.add(Dropout(dropout))
                

        # Capa salida
        self._model.add(
            Dense(units=n_output,
                  activation= fun_out if fun_out else fun_act))
        
        
        
        self._model.compile(loss=loss, 
                            optimizer=optimizer, 
                            metrics = metrics
                            )
        
        return self




    # Cambios aun por determinar
    def start_training(self,
                       epochs:int,
                       callbacks= []):
        
        
        
        # Preparar datos para entrenamiento y validacion
        assert self.train_data and self.val_data, "Deben de existir datos para entrenamiento"
        
        if isinstance(self.train_data, dict):
            
            train_data = self.train_data['tf_dataset']
            assert isinstance(train_data, tf.data.Dataset), "El flujo de datos de entrenamiento no es un tf.data.Dataset"
            train_data = train_data.prefetch(buffer_size=tf.data.AUTOTUNE)
        
        else:
           assert isinstance(self.train_data, tf.data.Dataset), "El flujo de datos de entrenamiento no es un tf.data.Dataset"
           train_data = self.train_data.prefetch(buffer_size=tf.data.AUTOTUNE)
          
        
        if isinstance(self.val_data, dict):
            
            val_data = self.val_data['tf_dataset']
            assert isinstance(val_data, tf.data.Dataset), "El flujo de datos de validación no es un tf.data.Dataset"
            val_data = val_data.prefetch(buffer_size=tf.data.AUTOTUNE)
        
        else:
            
            assert isinstance(self.val_data, tf.data.Dataset), "El flujo de datos de validación no es un tf.data.Dataset"
            val_data = self.val_data.prefetch(buffer_size=tf.data.AUTOTUNE)

            
        print("🚀 Iniciando el entrenamiento...\n")
        
        tiempo_inicio = time.time()
        
        self._model.fit(
            train_data,
            validation_data = val_data,
            epochs=epochs,
            callbacks=callbacks
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
        
        self._losses = pd.DataFrame(self._model.history.history)
            
        return self


                
    def img_prediction(self, img_path, umbral = None):
        assert self.categories, "Las categorias no estan definidas"
        
        # Cargar y redimensionar la imagen suelta
        img = self._cargar_imagen(img_path)
        
        
        # Convertir a lote (Añade una dimensión: de (H,W,C) a (1,H,W,C))
        # Predecir (La capa Rescaling interna del modelo se encargará de dividir por 255)
        img_batch = tf.expand_dims(img, 0) 
        
        
        if umbral is not None and 0 < umbral < 1:
           self._predictions = (self.model.predict(img_batch) >= umbral).astype("int32")
           return self

        elif umbral is not None:
            raise ValueError('Valores invalidos en el umbral')
        
        
        self._predictions = int(np.argmax(self.model.predict(img_batch)))
        
        self._predictions = self.categories[self._predictions]
        return self
        






    def batch_prediction(self, batch_path, img_xten = '*.png', umbral = None):
        
        img_class = []
        
        rutas_imagenes = sorted(str(ruta) for ruta in Path(batch_path).glob(img_xten))
        
        for ruta in rutas_imagenes:
            self.img_prediction(ruta, umbral = umbral)
            img_class.append(self.predictions)
            
        
        assert len(img_class) == len(rutas_imagenes), "¡La cantidad entre filas no coincide!"
                
        self._predictions = pd.DataFrame({'Rutas': rutas_imagenes,
                                        'Clasificacion': img_class})
        return self


            
    # Método para incorporar dentro de CNN_Model_Clasification.
    def evaluar(
        self,
        conjunto="val",
        *,
        tipo="multiclase",
        codificacion="indices",
        categorias=None,
        umbral=0.9,
        desde_logits=False,
        mostrar=True,
    ):
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            confusion_matrix,
        )

        # 1. Seleccionar un único conjunto.
        conjuntos = {
            "train": self.train_data,
            "val": self.val_data,
            "test": self.test_data,
        }

        if conjunto not in conjuntos:
            raise ValueError("conjunto debe ser 'train', 'val' o 'test'.")

        datos = conjuntos[conjunto]
        if datos is None:
            raise ValueError(f"No hay datos cargados para '{conjunto}'.")

        dataset = (
            datos.get("tf_dataset")
            if isinstance(datos, dict)
            else datos
        )

        if not isinstance(dataset, tf.data.Dataset):
            raise TypeError("Se requiere un tf.data.Dataset válido.")

        cardinalidad = int(
            tf.data.experimental.cardinality(dataset).numpy()
        )
        if cardinalidad <= 0:
            raise ValueError(
                "El dataset debe tener cardinalidad conocida, "
                "finita y mayor que cero."
            )

        if tipo not in ("binaria", "multiclase"):
            raise ValueError("tipo debe ser 'binaria' o 'multiclase'.")

        if codificacion not in ("indices", "one_hot"):
            raise ValueError(
                "codificacion debe ser 'indices' o 'one_hot'."
            )

        if tipo == "binaria" and not 0 < umbral < 1:
            raise ValueError("El umbral debe estar entre 0 y 1.")

        # 2. Recoger etiquetas y salidas en el MISMO recorrido.
        # Evita desalinearlas aunque el dataset use shuffle.
        etiquetas_lotes = []
        salidas_lotes = []

        for lote in dataset:
            if not isinstance(lote, (tuple, list)) or len(lote) != 2:
                raise ValueError(
                    "Cada lote debe contener (imagenes, etiquetas)."
                )

            imagenes, etiquetas = lote
            salidas = self.model(imagenes, training=False)

            etiquetas_lotes.append(np.asarray(etiquetas))
            salidas_lotes.append(np.asarray(salidas))

        if not salidas_lotes:
            raise ValueError("No se recibieron observaciones.")

        y_original = np.concatenate(etiquetas_lotes, axis=0)
        salidas = np.concatenate(salidas_lotes, axis=0)

        if salidas.ndim != 2 or len(salidas) == 0:
            raise ValueError(
                "Se espera una salida de forma (ejemplos, salidas)."
            )

        if not np.isfinite(salidas).all():
            raise ValueError("Las salidas contienen NaN o infinitos.")

        # 3. Convertir salidas a probabilidades y clases.
        if tipo == "binaria":
            if salidas.shape[1] != 1:
                raise ValueError(
                    "El modo binario espera una única salida. "
                    "Para dos salidas softmax, usa 'multiclase'."
                )

            p_positiva = salidas[:, 0]
            if desde_logits:
                p_positiva = tf.math.sigmoid(p_positiva).numpy()

            probabilidades = np.column_stack(
                [1.0 - p_positiva, p_positiva]
            )
            y_pred = (p_positiva >= umbral).astype(np.int64)
            n_clases = 2

        else:
            n_clases = salidas.shape[1]
            if n_clases < 2:
                raise ValueError(
                    "Multiclase requiere al menos dos salidas."
                )

            probabilidades = (
                tf.nn.softmax(salidas, axis=-1).numpy()
                if desde_logits
                else salidas
            )
            y_pred = np.argmax(probabilidades, axis=1)

        if (
            not np.isfinite(probabilidades).all()
            or np.any(probabilidades < 0)
            or np.any(probabilidades > 1)
            or not np.allclose(
                probabilidades.sum(axis=1), 1.0, atol=1e-5
            )
        ):
            raise ValueError(
                "Las salidas no son probabilidades válidas. "
                "Revisa la activación o desde_logits."
            )

        # 4. Normalizar las etiquetas reales.
        if y_original.dtype.kind not in "biuf":
            raise TypeError(
                "Las etiquetas deben estar codificadas numéricamente; "
                "no pueden ser nombres ni rutas de archivos."
            )

        if not np.isfinite(y_original).all():
            raise ValueError("Las etiquetas contienen NaN o infinitos.")

        if codificacion == "one_hot":
            if (
                y_original.ndim != 2
                or y_original.shape[1] != n_clases
                or not np.all(
                    (y_original == 0) | (y_original == 1)
                )
                or not np.all(y_original.sum(axis=1) == 1)
            ):
                raise ValueError(
                    "Las etiquetas no tienen un formato one-hot válido."
                )

            y_true = np.argmax(y_original, axis=1)

        else:
            if not (
                y_original.ndim == 1
                or (
                    y_original.ndim == 2
                    and y_original.shape[1] == 1
                )
            ):
                raise ValueError(
                    "Las etiquetas enteras deben tener forma (N,) o (N, 1)."
                )

            valores = y_original.reshape(-1)
            if not np.all(valores == np.floor(valores)):
                raise ValueError("Los índices de clase deben ser enteros.")

            if np.any(valores < 0) or np.any(valores >= n_clases):
                raise ValueError("Hay etiquetas fuera del rango de clases.")

            y_true = valores.astype(np.int64)

        if len(y_true) != len(y_pred):
            raise ValueError("No coinciden etiquetas y predicciones.")

        # Los nombres deben respetar el orden de índices del entrenamiento.
        nombres = categorias if categorias is not None else self.categories
        if nombres is None:
            nombres = [str(i) for i in range(n_clases)]

        if not isinstance(nombres, (list, tuple, np.ndarray)):
            raise TypeError("categorias debe ser una lista ordenada.")

        nombres = [str(nombre) for nombre in nombres]
        if len(nombres) != n_clases or len(set(nombres)) != n_clases:
            raise ValueError(
                "Debe existir un nombre único por cada clase."
            )

        # 5. Evaluar con TODAS las clases, aunque alguna no aparezca.
        ids = np.arange(n_clases)
        reporte = classification_report(
            y_true,
            y_pred,
            labels=ids,
            output_dict=True,
            zero_division=0,
        )

        por_clase = pd.DataFrame(reporte).T.loc[
            [str(i) for i in ids],
            ["precision", "recall", "f1-score", "support"],
        ]
        por_clase.index = pd.Index(nombres, name="Clase")
        por_clase["support"] = por_clase["support"].astype(int)

        matriz = pd.DataFrame(
            confusion_matrix(y_true, y_pred, labels=ids),
            index=nombres,
            columns=nombres,
        )
        matriz.index.name = "Real"
        matriz.columns.name = "Predicha"

        conteos = pd.DataFrame(
            {
                "Reales": np.bincount(y_true, minlength=n_clases),
                "Predichas": np.bincount(y_pred, minlength=n_clases),
            },
            index=pd.Index(nombres, name="Clase"),
        )

        predicciones = pd.DataFrame(
            {
                "Clase_real": y_true,
                "Clase_predicha": y_pred,
                "Nombre_real": [nombres[i] for i in y_true],
                "Nombre_predicho": [nombres[i] for i in y_pred],
                "Probabilidad_clase_predicha": probabilidades[
                    np.arange(len(y_pred)), y_pred
                ],
                "Acierto": y_true == y_pred,
            }
        )

        resultado = {
            "conjunto": conjunto,
            "tipo": tipo,
            "umbral": umbral if tipo == "binaria" else None,
            "n_ejemplos": len(y_true),
            "metricas": {
                "accuracy": accuracy_score(y_true, y_pred),
                "precision_macro": reporte["macro avg"]["precision"],
                "recall_macro": reporte["macro avg"]["recall"],
                "f1_macro": reporte["macro avg"]["f1-score"],
                "f1_ponderado": reporte["weighted avg"]["f1-score"],
            },
            "reporte_por_clase": por_clase,
            "matriz_confusion": matriz,
            "conteos": conteos,
            "predicciones": predicciones,
            "probabilidades": pd.DataFrame(
                probabilidades, columns=nombres
            ),
        }

        if mostrar:
            print(f"Conjunto: {conjunto} | Ejemplos: {len(y_true)}")
            print(pd.Series(resultado["metricas"]).round(4))
            print("\nReporte por clase:")
            print(por_clase.round(4))

            fig, ejes = plt.subplots(1, 2, figsize=(13, 5))
            sns.heatmap(
                matriz, annot=True, fmt="d", cmap="Blues", ax=ejes[0]
            )
            ejes[0].set_title("Matriz de confusión")

            conteos.plot.bar(ax=ejes[1])
            ejes[1].set_title("Distribución real y predicha")
            ejes[1].set_ylabel("Cantidad")
            fig.tight_layout()
            plt.show()

        return resultado