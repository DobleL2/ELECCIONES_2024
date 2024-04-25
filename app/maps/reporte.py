import pandas as pd
#-----------------
from fpdf import FPDF 
from tabulate import tabulate
import pandas as pd

def add_dataframe_to_pdf(pdf, dataframe, x, y):
    # Convertir el DataFrame a una cadena de texto formateada como una tabla
    table = tabulate(dataframe, headers='keys', tablefmt='simple')
    
    # Dividir la tabla en filas para agregarlas al PDF
    rows = table.split('\n')
    
    # Agregar cada fila al PDF
    for index, row in dataframe.iterrows():
        pdf.set_xy(x, y)
        pdf.set_font('Helvetica', '', 10) 
        pdf.cell(0, 0, row['First name']+'\t'+str(row['Age'])+'\t'+row['City'] )
        y += 5  # Ajustar la coordenada y para la siguiente fila


class PDFWithBackground(FPDF):
    def __init__(self):
        super().__init__()
        self.background = None

    def set_background(self, image_path):
        self.background = image_path

    def add_page(self, orientation='',same=False):
        super().add_page(orientation)
        if self.background:
            self.image(self.background, 0, 0, self.w, self.h)

    def footer(self):
        # Posición a 1.5 cm desde el fondo
        self.set_y(-15)
        # Configurar la fuente para el pie de página
        self.set_font('Helvetica', 'I', 8)
        # Número de página
        self.cell(0, 10, 'Página ' + str(self.page_no()), 0, 0, 'C')
    
    
Preguntas = {
    'A': 'A: Apoyo Complementario Fuerzas Armadas',
    'B': 'B: Extradición de Ecuatorianos',
    'C': 'C: Judicaturas Especializadas',
    'D': 'D: Arbitraje Internacional',
    'E': 'E: Trabajo a Plazo Fijo y por Horas',
    'F': 'F: Control de Armas',
    'G': 'G: Incremento de Penas',
    'H': 'H: Cumplimiento de Pena Total',
    'I': 'I: Tipificación de delitos por porte de armas',
    'J': 'J: Uso inmediato de armas usadas en delitos',
    'K': 'K: Confiscación de Activos Ilícitos'
}   

provincia = 'PICHINCHA'
pregunta = Preguntas['A']
        
pdf = PDFWithBackground()
pdf.set_background('portada_provincias.png')
pdf.add_page()







PREGUNTAS = pd.read_excel('../resultados_azure/CATALOGO/PREGUNTAS.xls')
CANTON = pd.read_excel('../resultados_azure/CATALOGO/CANTON.xls')
CIRCUNSCRIPCION = pd.read_excel('../resultados_azure/CATALOGO/CIRCUNSCRIPCION.xls')
DIGNIDAD = pd.read_excel('../resultados_azure/CATALOGO/DIGNIDAD.xls')
JUNTA = pd.read_excel('../resultados_azure/CATALOGO/JUNTA.xls')
PARROQUIA = pd.read_excel('../resultados_azure/CATALOGO/PARROQUIA.xls')
OPCIONES = pd.read_excel('../resultados_azure/CATALOGO/OPCIONES.xls')
PROVINCIA = pd.read_excel('../resultados_azure/CATALOGO/PROVINCIA.xls')
ZONA = pd.read_excel('../resultados_azure/CATALOGO/ZONA.xls')

bases_azure = [
    "../resultados_azure/A_REFERENDUM_PREGUNTA_2024.04.25_01.50.01.txt",
    "../resultados_azure/B_REFERENDUM_PREGUNTA1_2024.04.25_01.50.02.txt",
    "../resultados_azure/C_REFERENDUM_PREGUNTA2_2024.04.25_01.50.03.txt",
    "../resultados_azure/D_REFERENDUM_PREGUNTA3_2024.04.25_01.50.04.txt",
    "../resultados_azure/E_REFERENDUM_PREGUNTA4_2024.04.25_01.50.05.txt",
    "../resultados_azure/F_CONSULTAPOPULAR_PREGUNTA1_2024.04.25_01.55.06.txt",
    "../resultados_azure/G_CONSULTAPOPULAR_PREGUNTA2_2024.04.25_01.55.08.txt",
    "../resultados_azure/H_CONSULTAPOPULAR_PREGUNTA3_2024.04.25_01.55.09.txt",
    "../resultados_azure/I_CONSULTAPOPULAR_PREGUNTA4_2024.04.25_01.55.09.txt"
]

final = pd.DataFrame()
for i in bases_azure:
    lectura =  pd.read_table(i,sep='|',names= 
                             ['COD_DIGNIDAD','COD_PROVINCIA',
                              'COD_CIRCUNSCRIPCION','COD_CANTON','COD_PARROQUIA',
                              'COD_ZONA','NUM_JUNTA','SEX_JUNTA','COD_JUNTA',
                              'COD_PREGUNTA','COD_OPCION','NUM_SUF_ACTA','BLANCOS','NULOS','FIN_RESULTADO'])
    final = pd.concat([final,lectura])

exterior = ['EUROPA ASIA Y OCEANIA', 'EE.UU CANADA', 'AMERICA LATINA EL CARIBE Y AFRICA']
baseP = final

baseP = pd.merge(DIGNIDAD,baseP,on='COD_DIGNIDAD')
baseP = pd.merge(PROVINCIA,baseP,on='COD_PROVINCIA')
baseP = pd.merge(CANTON[['COD_CANTON','NOM_CANTON']],baseP,on='COD_CANTON')
baseP = pd.merge(PARROQUIA[['COD_PARROQUIA','NOM_PARROQUIA']],baseP,on='COD_PARROQUIA')
#baseP = pd.merge(Est_parroquia,baseP,on='COD_PARROQUIA')
baseP = pd.merge(PREGUNTAS[['COD_PREGUNTA','NOM_PREGUNTA','LIS_PREGUNTA']],baseP,on='COD_PREGUNTA')
baseP = pd.merge(OPCIONES[['COD_OPCION','NOM_OPCION']],baseP,on='COD_OPCION')

baseP = baseP[-baseP['NOM_PROVINCIA'].isin(exterior)]

resultadosP = baseP.groupby(by = ['NOM_PREGUNTA','NOM_PROVINCIA','NOM_OPCION']).sum(numeric_only=True)[['FIN_RESULTADO']].reset_index()
resultadosP = resultadosP[-resultadosP['NOM_PROVINCIA'].isin(exterior)]


preguntas = resultadosP['NOM_PREGUNTA'].unique()
provincias = resultadosP['NOM_PROVINCIA'].unique()


def proporcion(NO,SI):
    if NO>=SI:
        return NO/SI, 'NO'
    return SI/NO, 'SI'
baseP.groupby(by = ['NOM_PREGUNTA','NOM_PROVINCIA','NOM_OPCION']).sum(numeric_only=True)[['FIN_RESULTADO']].reset_index()
resultado_prop = pd.pivot_table(baseP,'FIN_RESULTADO',aggfunc='sum',index=['NOM_PREGUNTA','NOM_PROVINCIA'],columns='NOM_OPCION').reset_index()
resultado_prop['prop'] = resultado_prop.apply(lambda row: proporcion(row['NO'],row['SI']),axis=1)
resultado_prop = pd.concat([resultado_prop,resultado_prop['prop'].apply(lambda x: pd.Series(str(x).strip("()'").split(", '")))],axis=1)
resultado_prop['Normalizacion'] = resultado_prop[0].astype(float)/5
resultado_prop['TOTAL'] = resultado_prop['NO'] + resultado_prop['SI']
resultado_prop['NO_prop'] = resultado_prop['NO']/resultado_prop['TOTAL']
resultado_prop['SI_prop'] = resultado_prop['SI']/resultado_prop['TOTAL']
resultado_prop['NO %'] = resultado_prop['NO_prop'].apply(lambda x: f"{x:.2%}")
resultado_prop['SI %'] = resultado_prop['SI_prop'].apply(lambda x: f"{x:.2%}")

for pregunta in preguntas:
    df1 = resultado_prop[resultado_prop['NOM_PREGUNTA']==pregunta][['NOM_PROVINCIA','SI %','NO %']]
    df2 = resultado_prop[resultado_prop['NOM_PREGUNTA']==pregunta][['NOM_PROVINCIA',0]]
    
    prop_SI = resultado_prop[(resultado_prop['NOM_PREGUNTA']==pregunta) & (resultado_prop[1]=='SI')][['NOM_PROVINCIA','SI_prop']].set_index('NOM_PROVINCIA').to_dict()['SI_prop']
    prop_NO = resultado_prop[(resultado_prop['NOM_PREGUNTA']==pregunta) & (resultado_prop[1]=='NO')][['NOM_PROVINCIA','NO_prop']].set_index('NOM_PROVINCIA').to_dict()['NO_prop']
    
    dic_SI = resultado_prop[(resultado_prop['NOM_PREGUNTA']==pregunta) & (resultado_prop[1]=='SI')][['NOM_PROVINCIA','Normalizacion',0]]
    dic_SI_norm = dic_SI.copy() 
    if dic_SI.shape[0] == 0:
        dic_SI = {}
        dic_SI_norm = {}
    else:
        dic_SI[0] = dic_SI[0].astype(float)
        dic_SI = dic_SI.set_index('NOM_PROVINCIA').to_dict()[0]    
        dic_SI_norm['Normalizacion'] = dic_SI_norm['Normalizacion'].astype(float)
        dic_SI_norm = dic_SI_norm.set_index('NOM_PROVINCIA').to_dict()['Normalizacion']
        
    dic_NO = resultado_prop[(resultado_prop['NOM_PREGUNTA']==pregunta) & (resultado_prop[1]=='NO')][['NOM_PROVINCIA','Normalizacion',0]]
    dic_NO_norm = dic_NO.copy() 
    if dic_NO.shape[0] == 0:
        dic_NO = {}
        dic_NO_norm = {}
    else:

        dic_NO[0] = dic_NO[0].astype(float)
        dic_NO = dic_NO.set_index('NOM_PROVINCIA').to_dict()[0]     
        dic_NO_norm['Normalizacion'] = dic_NO_norm['Normalizacion'].astype(float)
        dic_NO_norm = dic_NO_norm.set_index('NOM_PROVINCIA').to_dict()['Normalizacion']
    

# GRAFICO 1

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.colorbar import ColorbarBase
import matplotlib.ticker as ticker

# Load GeoJSON file
gdf = gpd.read_file("Ecuador.geojson")
gdf['name'] = gdf['name'].str.upper()
gdf['name'] = gdf['name'].str.replace('SANTO DOMINGO DE LOS TSACHILAS', 'STO DGO TSACHILAS')

color_si = '#0083E9'
color_no = '#E99100'

# Plot the shapes with a heatmap-like effect based on the score
fig, ax = plt.subplots(figsize=(10, 10))
gdf.plot(ax=ax, color='white', edgecolor='black')

for name, score in prop_SI.items():
    shape = gdf[gdf['name'] == name]
    shape.plot(ax=ax, color=color_si, alpha=score, linewidth=0, label=name)
    
for name, score in prop_NO.items():
    shape = gdf[gdf['name'] == name]
    shape.plot(ax=ax, color=color_no, alpha=score, linewidth=0, label=name)

# Create a colormap with a gradient using the specified color
cmap = LinearSegmentedColormap.from_list('custom_cmap', [(0, '#ffffff'), (1, color_si)])

# Crea un objeto Normalizar para mapear los valores al rango 0-1
norm = Normalize(vmin=0, vmax=1)

valores_a_convertir_si = prop_SI.values()
valores_a_convertir_no = prop_NO.values()

# Obtiene los colores correspondientes a los valores dados
colores_correspondientes_si = [cmap(norm(valor)) for valor in valores_a_convertir_si]



# Convierte los colores de formato RGB a formato hexadecimal
colores_hex_si = [plt.cm.colors.to_hex(color) for color in colores_correspondientes_si]


# Add colorbar
cbar_ax1 = fig.add_axes([0, 0.52, 0.03, 0.3])
norm = Normalize(vmin=0, vmax=1)
cbar = ColorbarBase(ax=cbar_ax1, cmap=cmap, norm=norm)
cbar.ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))  # Set the labels to percentages from 1 to 100%

# Create a colormap with a gradient using the specified color
cmap = LinearSegmentedColormap.from_list('custom_cmap', [(0, '#ffffff'), (1, color_no)])
colores_correspondientes_no = [cmap(norm(valor)) for valor in valores_a_convertir_no]
colores_hex_no = [plt.cm.colors.to_hex(color) for color in colores_correspondientes_no]
# Add colorbar
cbar_ax2 = fig.add_axes([0, 0.18, 0.03, 0.3])
norm = Normalize(vmin=0, vmax=1)
cbar = ColorbarBase(ax=cbar_ax2, cmap=cmap, norm=norm)
cbar.ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))  # Set the labels to percentages from 1 to 100%

plt.legend()
ax.axis('off')
# Save the plot as a PNG image
plt.savefig('mapas/heatmap_plot.png', dpi=300, transparent=True)  # Adjust dpi as needed

data_SI = {
    'NOM_PROVINCIA' : prop_SI.keys(),
    'PROP' : prop_SI.values()
}
data_SI = pd.DataFrame(data_SI)
data_SI['Color Hex'] = colores_hex_si
data_SI['RESPUESTA'] = 'SI'

data_NO = {
    'NOM_PROVINCIA' : prop_NO.keys(),
    'PROP' : prop_NO.values()
}
data_NO = pd.DataFrame(data_NO)
data_NO['Color Hex'] = colores_hex_no
data_NO['RESPUESTA'] = 'NO'

data_1 = pd.concat([data_SI,data_NO])
data_1


#PLOT 2
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.colorbar import ColorbarBase
import matplotlib.ticker as ticker

# Load GeoJSON file
gdf = gpd.read_file("Ecuador.geojson")
gdf['name'] = gdf['name'].str.upper()
gdf['name'] = gdf['name'].str.replace('SANTO DOMINGO DE LOS TSACHILAS', 'STO DGO TSACHILAS')

color_si = '#0083E9'
color_no = '#E99100'

# Plot the shapes with a heatmap-like effect based on the score
fig, ax = plt.subplots(figsize=(10, 10))
gdf.plot(ax=ax, color='white', edgecolor='black')

for name, score in dic_SI_norm.items():
    shape = gdf[gdf['name'] == name]
    shape.plot(ax=ax, color=color_si, alpha=score, linewidth=0, label=name)
    
for name, score in dic_NO_norm.items():
    shape = gdf[gdf['name'] == name]
    shape.plot(ax=ax, color=color_no, alpha=score, linewidth=0, label=name)

# Create a colormap with a gradient using the specified color
cmap = LinearSegmentedColormap.from_list('custom_cmap', [(0, '#ffffff'), (1, color_si)])

# Crea un objeto Normalizar para mapear los valores al rango 0-1
norm = Normalize(vmin=0, vmax=5)

valores_a_convertir_si = dic_SI.values()
valores_a_convertir_no = dic_NO.values()

# Obtiene los colores correspondientes a los valores dados
colores_correspondientes_si = [cmap(norm(valor)) for valor in valores_a_convertir_si]



# Convierte los colores de formato RGB a formato hexadecimal
colores_hex_si = [plt.cm.colors.to_hex(color) for color in colores_correspondientes_si]


# Add colorbar
cbar_ax1 = fig.add_axes([0, 0.52, 0.03, 0.3])

cbar = ColorbarBase(ax=cbar_ax1, cmap=cmap, norm=norm)
#cbar.ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))  # Set the labels to percentages from 1 to 100%

# Create a colormap with a gradient using the specified color
cmap = LinearSegmentedColormap.from_list('custom_cmap', [(0, '#ffffff'), (1, color_no)])
colores_correspondientes_no = [cmap(norm(valor)) for valor in valores_a_convertir_no]
colores_hex_no = [plt.cm.colors.to_hex(color) for color in colores_correspondientes_no]
# Add colorbar
cbar_ax2 = fig.add_axes([0, 0.18, 0.03, 0.3])

cbar = ColorbarBase(ax=cbar_ax2, cmap=cmap, norm=norm)
#cbar.ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))  # Set the labels to percentages from 1 to 100%

plt.legend()
ax.axis('off')
# Save the plot as a PNG image
plt.savefig('mapas/heatmap_plot.png', dpi=300, transparent=True)  # Adjust dpi as needed

data_SI = {
    'NOM_PROVINCIA' : dic_SI.keys(),
    'PROP' : dic_SI.values()
}
data_SI = pd.DataFrame(data_SI)
data_SI['Color Hex'] = colores_hex_si
data_SI['RESPUESTA'] = 'SI'

data_NO = {
    'NOM_PROVINCIA' : dic_NO.keys(),
    'PROP' : dic_NO.values()
}
data_NO = pd.DataFrame(data_NO)
data_NO['Color Hex'] = colores_hex_no
data_NO['RESPUESTA'] = 'NO'

data_1 = pd.concat([data_SI,data_NO])



pdf.set_background('background.jpeg')
pdf.add_page()

pdf.set_xy(15,15)
pdf.add_font('Anton-Regular', '', '../fonts/Anton-Regular.ttf', uni=True)  # Register the custom font
pdf.set_font('Helvetica','B',size=25)   # Helvetica, Times, Courier
pdf.cell(0,0,'Resultados Provinciales',0,1,'L')

pdf.set_xy(15,15)
pdf.set_font('Helvetica', '', 20)  # Use the registered font
pdf.cell(0,0,'Referéndum',0,1,'R')



image = 'heatmap_plot'
imagen = f'mapas/{image}.png'
pdf.image(f'{imagen}',x=25,y=25,w=115,h=115)

image = 'heatmap_plot'
imagen = f'mapas/{image}.png'
pdf.image(f'{imagen}',x=85,y=145,w=115,h=115)

pdf.set_xy(15,26)
pdf.set_font('Helvetica', 'B', 18)  # Use the registered font
pdf.cell(0,0,'Pregunta:',0,1,'L')

pdf.set_xy(47,26)
pdf.set_font('Helvetica', '', 16)  # Use the registered font
pdf.cell(0,0,pregunta,0,1,'L')



df = df1.astype(str)
x = 135
y = 35
pdf.set_xy(x,y)
pdf.set_font("Helvetica", size=5)
table_data = [df.columns.tolist()] + df.values.tolist()

with pdf.table(width=50,col_widths=(25,10,10), align='L') as table:
    for data_row in table_data:
        row = table.row()
        for datum in data_row:
            row.cell(datum)
            
df = df1.astype(str)
x = 25
y = 155
pdf.set_xy(x,y)
pdf.set_font("Helvetica", size=5)
table_data = [df.columns.tolist()] + df.values.tolist()

with pdf.table(width=50,col_widths=(25,10,10), align='L') as table:
    for data_row in table_data:
        row = table.row()
        for datum in data_row:
            row.cell(datum)



pdf.output('Mapas_provincias.pdf')

