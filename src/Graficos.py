import sys
import os

sys.path.append(os.path.abspath(os.path.join('..')))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from dython.nominal import associations
from scipy.stats import gaussian_kde
        
import time
import folium
from folium.plugins import HeatMap
from geopy.geocoders import Nominatim


class Graficos:
    def __init__(self):
        pass
    
    
    
    
    def plot_boxplot_histogram(self, df, column):
        x = df[column].dropna()
        fig, (ax_box, ax_hist) = plt.subplots(2, 1, sharex=True, figsize=(10, 6),
                                            gridspec_kw={"height_ratios": [1, 4]})
        
        plt.subplot(2, 1, 1).set_title(f"Distribución de la variable '{column}'", fontsize=14, fontweight='bold')
        
        ax_box.boxplot(x, widths=0.6, orientation="horizontal")
        ax_box.set_yticks([])

        ax_hist.hist(x, density=True, alpha=0.6, edgecolor="white")
        xs = np.linspace(x.min(), x.max(), 300)
        ax_hist.plot(xs, gaussian_kde(x)(xs), color="darkblue", lw=2, label="Densidad Estimada por núcleo (KDE)")
        ax_hist.legend()
        plt.show()
        
        
        return self
        
        
        
        
    def plot_bar(self, df, columna, color = ['blue', 'red']):
        barras = tuple(str(unico) for unico in list(df[columna].unique()))
        print(barras)
        alto = []
        for barra in tuple(unico for unico in list(df[columna].unique())):
            alto.append(df.loc[:, columna].value_counts().get(barra, 0))
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize = (10,5), dpi = 100) #Densidad de pixeles es dpi
        bars = ax.bar(barras, alto, color=color, edgecolor='black', alpha=0.7)
        ax.bar_label(bars, padding=3, fontsize=11, fontweight='bold')
        plt.title(f"Distribución de la variable '{columna}'", fontsize=14, fontweight='bold')
        plt.xlabel(columna)
        plt.ylabel("Frecuencia")
        plt.ylim(0, max(alto) * 1.1) # Para aumentar el limite superior del eje y y que no se corte la barra más alta
        plt.show()
        
        return self
    
    def corr(self, data):
        resultados = associations(
            data, 
            figsize=(20, 20), 
            cmap='coolwarm',
            title='Matriz de Correlación',
            fmt=".2f",
            annot=True
            )
        return self
    
    
    def heatmap(self, data, title:str, label:str, xlabel:str, ylabel:str, ancho:int = 22, alto:int = 8, rotacion:int= 45, direccion_rotacion:str = 'right'):
        # Nota: Como Costa Rica tiene muchos cantones, dale buen ancho (ej. 20) para que no se amontone el texto
        plt.figure(figsize=(ancho, alto))

        # 3. Graficar el mapa de calor 📊🔥
        ax = sns.heatmap(
            data, 
            annot=True,             # Muestra la cantidad exacta de casos en cada celda
            fmt="g",                # Evita que los números grandes se muestren en notación científica (ej: 3.7e3)
            cmap="YlOrRd",          # Escala de colores (Amarillo -> Naranja -> Rojo)
            linewidths=0.8,         # Separa sutilmente las celdas con una línea
            cbar_kws={'label': label}
        )

        # 4. Ajustes estéticos y títulos
        
        ax.xaxis.tick_top()                         # Mueve los nombres que quedan abajo arriba
        ax.xaxis.set_label_position('top')          # Mueve el título arriba

        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.xlabel(xlabel, fontsize=12, fontweight='bold')
        plt.ylabel(ylabel, fontsize=12, fontweight='bold')
        plt.xticks(rotation=rotacion, ha= direccion_rotacion) # Rota los nombres de los cantones para que sean legibles

        plt.tight_layout()
        plt.show()
        
        
        
        
    def coord_x_prov_canton(self, datos, col_provincia, col_canton):

        # ==========================================
        # OBTENER LAS COORDENADAS DE LOS CANTONES
        # ==========================================

        # Buscador de coordenadas (User-agent es obligatorio)
        geolocalizador = Nominatim(user_agent="VectorIA_Proyecto_CUC")


        cantones_unicos = datos[[col_canton, col_provincia]].drop_duplicates()

        # Diccionario para almacenar las coordenadas guardadas
        coordenadas_dict = {}

        for idx, fila in cantones_unicos.iterrows():
            canton = fila[col_canton]
            provincia = fila[col_provincia]
            
            # Creamos una dirección clara para que el buscador no se confunda con ciudades de otros países
            direccion_busqueda = f"{canton}, {provincia}, Costa Rica"
            
            try:
                ubicacion = geolocalizador.geocode(direccion_busqueda)
                if ubicacion:
                    coordenadas_dict[canton] = (ubicacion.latitude, ubicacion.longitude)
                else:
                    # Reintento más genérico si falla el primero
                    ubicacion = geolocalizador.geocode(f"{canton}, Costa Rica")
                    if ubicacion:
                        coordenadas_dict[canton] = (ubicacion.latitude, ubicacion.longitude)
            except Exception as e:
                print(f"No se pudo obtener coordenadas para {canton}: {e}")
            
            # Respetar los términos de uso de Nominatim (máximo 1 consulta por segundo)
            time.sleep(1)

        # ==========================================
        # ASIGNAR LAS COORDENADAS AL DATAFRAME
        # ==========================================

        # Mapeamos las coordenadas encontradas a nuestro DataFrame original
        datos['latitud'] = datos['canton'].map(lambda x: coordenadas_dict.get(x, (None, None))[0])
        datos['longitud'] = datos['canton'].map(lambda x: coordenadas_dict.get(x, (None, None))[1])

        # Limpiamos el DataFrame eliminando los cantones que no se lograron geolocalizar
        df_mapa = datos.dropna(subset=['latitud', 'longitud'])
        return df_mapa

        
        
                
    def generar_mapa(self, datos, col_provincia, col_canton, col_escala:str, ):
        
        df_mapa = self.coord_x_prov_canton(datos, col_provincia, col_canton)
        
        # Se crea el mapa base centrado en Costa Rica
        mapa_cr = folium.Map(location=[9.7489, -83.7534], 
                            zoom_start=8, 
                            tiles="https://{s}.tile.openstreetmap.fr/osmfr/{z}/{x}/{y}.png",
                            attr="&copy; OpenStreetMap France | &copy; <a href='https://openstreetmap.org'>OpenStreetMap</a>"
                            )

        # Convertimos los datos al formato numérico que requiere Folium [[lat, lon, casos]]
        datos_calor = df_mapa[['latitud', 'longitud', col_escala]].values.tolist()

        # Añadimos la capa de calor
        HeatMap(
            datos_calor,
            radius=20,     # Ajusta el tamaño de la mancha de calor
            blur=15,       # Ajusta el difuminado
            max_zoom=10
        ).add_to(mapa_cr)

        # Guardar y mostrar
        mapa_cr.save("mapa_calor_completo.html")