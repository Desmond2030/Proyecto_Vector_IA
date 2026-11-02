import sys
import os

sys.path.append(os.path.abspath(os.path.join('..')))

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


from src.Tabular_Data_Loader import TabularDataLoader
from src.Graficos import Graficos


class TabularDataProcessor(TabularDataLoader):
    def __init__(self):
        super().__init__()
        
        #Data Info
        self._n_col = None
        self._n_row = None
        self._col_names = None
        
        #Busqueda de datos
        self._searched_data = None
        
        #Componentes
        self.graficos = Graficos()
        

    @property
    def n_col(self):
        return self._n_col
    

    @property
    def n_row(self):
        return self._n_row
    
  
    @property
    def col_names(self):
        return self._col_names
    
    @col_names.setter
    def col_names(self, new_names):
        
        if new_names is not None:
            # Verificacion de tipos
            if isinstance(new_names, dict):
                self._data.rename(
                    columns = new_names,
                    inplace=True
                )
            elif isinstance(new_names, (list, tuple)):
                self._data.columns = new_names
                
            else:
                raise ValueError(f'col_names debe de ser tipo dict, list o tuple y es {type(new_names).__name__}')
  
        self._col_names = list(self._data.columns)
    
    @property
    def searched_data(self):
        return self._searched_data
    

    
    
    def set_info(self):
        self._n_row, self._n_col = self._data.shape
        self._col_names = list(self.data.columns)
        return self
    
        
    def search_as_data(self):
        self._data = self._searched_data
        return self
    
    
    def item_in_col(self, 
                          data:pd.DataFrame = None, 
                          columnas:list= None, 
                          contenido:list= None):
        indx_raw = []
        data = data if data else self._data
        columnas = columnas if columnas else data.columns    
        
        for items in contenido:
            for columna in columnas:
                if columna in data.columns:
                    indx_raw = list(set(indx_raw + list(self.data[self.data[columna] == items].index)))
            
                
        self._searched_data = self.data.loc[indx_raw]
        
        return self
    
    
    
    def set_types(self, tipos:dict ) -> 'DataProcessor':
        self._data = self._data.astype(tipos)
        return self
    
    

    
    def replace_to_numeric_bool(self,
                     columnas:list,
                     map:dict = {},
                     replace = None,
                     *args,
                     **kargs) -> 'DataProcessor':
        
        if replace:
            for col in columnas:
                self._data[col] = self._data[col].replace(*args, **kargs)
        
        for columna in columnas:
            self._data[columna] = self._data[columna].map(map).astype(int)
        return self
    
    




    def set_categorical_columns(self, 
                                columna: str, 
                                categorias: list = None, 
                                cat_codes:bool = False, 
                                *args, 
                                **kargs) -> 'DataProcessor':
        
        
        
        categories = pd.api.types.CategoricalDtype(
            categories = categorias if categorias else list(self._data[columna].unique()), 
            *args,
            **kargs
        )
       
        self._data[columna] = self._data[columna].astype(categories)
        
        if cat_codes:
            self._data[columna] = self._data[columna].cat.codes
        
        return self
    

    
    def get_dummies(self, 
                    columna: str, 
                    prefijo:str, 
                    categorias: list = None,
                    drop_dummie:list = None,
                    *args, 
                    **kargs) -> 'DataProcessor':
        
        
        self.set_categorical_columns(columna, *args, **kargs)
        
        dummies = pd.get_dummies(self._data[columna], prefix=prefijo, dtype=np.int8)
        dummies_columns = []
        
        for categoria in list(self._data[columna].unique()):
            dummies_columns = [col for col in dummies.columns if col != f"{prefijo}_{categoria}"]
            
        dummies = dummies[dummies_columns].copy()
        print(dummies.columns)
        
        idx = self._data.columns.get_loc(columna)
        
        for i, col_name in enumerate(dummies.columns):
            self._data.insert(loc=idx + i, column=col_name, value=dummies[col_name])
        
        self._data.drop(columns=[columna], inplace=True) 
        
        if drop_dummie:
            self._data.drop(columns=drop_dummie, inplace=True)
        
        return self



    def verif(self, DF) -> dict:
        d = {}
        f, c = DF.shape
        for i in range(c):
            col = list(DF.iloc[:, i])
            for e in range(len(col)):
                tipo = type(col[e])
                t = type(col[e -1])
                if t == tipo:
                    d.update({DF.columns.values[i] : tipo}) 
                else:
                    d.update({DF.columns.values[i] : "datos incosistentes"}) 
                    break
        return d, f, c