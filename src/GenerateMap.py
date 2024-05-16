import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import contextily as ctx

# Load data from a CSV file
df = pd.read_csv('../data/properati_argentina_2021_tp1.csv', usecols=['latitud', 'longitud', 'tipo_precio'])

# Define a color mapping for price types
color_map = {
    'alto': 'red',    # Red for high prices
    'medio': 'blue', # Yellow for medium prices
    'bajo': 'green'  # Green for low prices
}

traduction = {
    'alto': 'High',
    'medio': 'Medium',
    'bajo': 'Low'
}

# Apply the color mapping to the DataFrame
df['color'] = df['tipo_precio'].map(color_map)

# Create a figure and coordinate system
fig, ax = plt.subplots(figsize=(10, 10))

# Plot points using the longitude and latitude coordinates
sc = ax.scatter(df['longitud'].astype(float), df['latitud'].astype(float), c=df['color'], s=2, alpha=0.5)

# Add the base map
ctx.add_basemap(ax, crs="EPSG:4326", source=ctx.providers.CartoDB.Positron)

# Adjust the limits to center the points on the map
ax.set_xlim(df['longitud'].min() - 0.05, df['longitud'].max() + 0.05)
ax.set_ylim(df['latitud'].min() - 0.05, df['latitud'].max() + 0.05)

# Create handles for the legend
handles = [mpatches.Patch(color=color, label=traduction[label]) for label, color in color_map.items()]
ax.legend(handles=handles, title="Price Type", title_fontsize='13', loc='upper right')

# Optional: further customize the plot
ax.axis('off')  # Hide the axes

# Save the figure
plt.savefig('map_buenos_aires.png', dpi=300)

# Display the plot if you want to see it in an interactive development environment
plt.show()
