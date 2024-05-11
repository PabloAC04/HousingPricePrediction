import pandas as pd
import matplotlib.pyplot as plt
import contextily as ctx
from PIL import Image

# Cargar datos desde un archivo CSV
df = pd.read_csv('../data/properati_argentina_2021_tp1.csv', usecols=['latitud', 'longitud', 'tipo_precio'])

# Definir un mapeo de colores para los tipos de precio
color_map = {
    'alto': 'red',    # Rojo para precios altos
    'medio': 'yellow', # Naranja para precios medios
    'bajo': 'green'  # Amarillo para precios bajos
}

# Aplicar el mapeo de colores al DataFrame
df['color'] = df['tipo_precio'].map(color_map)

# Crear una figura y un sistema de coordenadas
fig, ax = plt.subplots(figsize=(10, 10))

# Plotear puntos usando las coordenadas de longitud y latitud
ax.scatter(df['longitud'].astype(float), df['latitud'].astype(float), c=df['color'], s=2, alpha=0.5)  # Usar la columna 'color' para los colores

# Añadir el mapa base
ctx.add_basemap(ax, crs="EPSG:4326", source=ctx.providers.CartoDB.Positron)

# Ajustar los límites para centrar los puntos en el mapa
ax.set_xlim(df['longitud'].min() - 0.05, df['longitud'].max() + 0.05)
ax.set_ylim(df['latitud'].min() - 0.05, df['latitud'].max() + 0.05)

# Opcionales: personalizar más el gráfico
ax.axis('off')  # Ocultar los ejes

# Guardar la figura
plt.savefig('map_buenos_aires.png', dpi=300)

# Mostrar el gráfico si deseas verlo en un entorno de desarrollo interactivo
plt.show()
