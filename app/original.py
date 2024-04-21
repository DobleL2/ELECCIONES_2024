from collections import namedtuple
import math
import pandas as pd
import streamlit as st
import src.connection as connection
import src.tablas as tablas
import src.etls as etls 
import src.graficos as graficos
from st_on_hover_tabs import on_hover_tabs

# Configuración de página y estilos
st.set_page_config(layout="wide", page_title='Resultados Conteo', page_icon=':white_circle:')
st.markdown('<style>' + open('./style.css').read() + '</style>', unsafe_allow_html=True)

# Funciones para cargar las tablas
@st.cache(allow_output_mutation=True)
def cargar_transmision():
    return tablas.transmision()

@st.cache(allow_output_mutation=True)
def cargar_provincias():
    return tablas.provincias()

@st.cache(allow_output_mutation=True)
def cargar_juntas():
    return tablas.juntas()

# Sidebar con pestañas
with st.sidebar:
    tabs = on_hover_tabs(tabName=['Dashboard', 'Money', 'Economy'], 
                         iconName=['dashboard', 'trending_up', 'monitoring'],
                         styles = {'navtab': {'background-color':'#111',
                                              'color': '#818181',
                                              'font-size': '18px',
                                              'transition': '.3s',
                                              'white-space': 'nowrap',
                                              'text-transform': 'uppercase'},
                                   'tabOptionsStyle': {':hover :hover': {'color': 'white',
                                                                  'cursor': 'pointer'}},
                                   'iconStyle':{'position':'fixed',
                                                'left':'7.5px',
                                                'text-align': 'left'},
                                   'tabStyle' : {'list-style-type': 'none',
                                                 'margin-bottom': '30px',
                                                 'padding-left': '30px'}},
                         key="1")

# Contenido de la aplicación según la pestaña seleccionada
if tabs =='Dashboard':
    st.title("Navigation Bar")
    st.write('Name of option is {}'.format(tabs))

elif tabs == 'Money':
    st.title("Paper")
    st.write('Name of option is {}'.format(tabs))

elif tabs == 'Economy':
    st.title("Tom")
    st.write('Name of option is {}'.format(tabs))

# Resultados Control Electoral
st.title("Resultados Control Electoral")

Preguntas = {
    'A': 'Apoyo Complementario Fuerzas Armadas',
    'B': 'Extradición de Ecuatorianos',
    'C': 'Judicaturas Especializadas',
    'D': 'Arbitraje Internacional',
    'E': 'Trabajo a Plazo Fijo y por Horas',
    'F': 'Control de Armas',
    'G': 'Incremento de Penas',
    'H': 'Cumplimiento de Pena Total',
    'I': 'Tipificación de delitos por porte de armas',
    'J': 'Uso inmediato de armas usadas en delitos',
    'K': 'Confiscación de Activos Ilícitos'
}   

numero_letra = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D',
    4: 'E',
    5: 'F',
    6: 'G',
    7: 'H',
    8: 'I',
    9: 'J',
    10: 'K'
}

# Carga de datos
df_transmision = cargar_transmision()
st.metric('Actas ingresadas: ', df_transmision.shape[0])

# Resumen de votos por pregunta
df_transmision = etls.convertir_formato(df_transmision)
pregunta_seleccionada = st.selectbox('Pregunta: ', df_transmision['COD_PREGUNTA'].unique())
resumen = df_transmision.groupby(by='COD_PREGUNTA').sum(numeric_only=True)

# Mostrar resumen por pregunta
for i in range(11):
    if numero_letra[i] == pregunta_seleccionada:
        st.subheader(Preguntas[numero_letra[i]])
        graficos.resumen_general_pregunta(resumen, i)

# Mostrar datos de transmisión
st.write(df_transmision)

# Mostrar datos de provincias
df_provincias = cargar_provincias()
st.write(df_provincias)

# Mostrar datos de juntas
df_juntas = cargar_juntas()
st.write(df_juntas)
