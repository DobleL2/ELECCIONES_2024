from collections import namedtuple
import math
import pandas as pd
import streamlit as st
import src.connection as connection
import src.tablas as tablas
import src.etls as etls 
import src.graficos as graficos
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from st_on_hover_tabs import on_hover_tabs
import streamlit as st
st.set_page_config(layout="wide")

st.header("Custom tab component for on-hover navigation bar")
st.markdown('<style>' + open('./style.css').read() + '</style>', unsafe_allow_html=True)


with st.sidebar:
    tabs = on_hover_tabs(tabName=['Dashboard', 'Money', 'Economy'], 
                         iconName=['dashboard', 'money', 'economy'], default_choice=0)

if tabs =='Dashboard':
    st.title("Navigation Bar")
    st.write('Name of option is {}'.format(tabs))

elif tabs == 'Money':
    st.title("Paper")
    st.write('Name of option is {}'.format(tabs))

elif tabs == 'Economy':
    st.title("Tom")
    st.write('Name of option is {}'.format(tabs))

with st.sidebar:
        tabs = on_hover_tabs(tabName=['Dashboard', 'Money', 'Economy'], 
                             iconName=['dashboard', 'money', 'economy'],
                             styles = {'navtab': {'background-color':'#111',
                                                  'color': '#818181',
                                                  'font-size': '18px',
                                                  'transition': '.3s',
                                                  'white-space': 'nowrap',
                                                  'text-transform': 'uppercase'},
                                       'tabOptionsStyle': {':hover :hover': {'color': 'red',
                                                                      'cursor': 'pointer'}},
                                       'iconStyle':{'position':'fixed',
                                                    'left':'7.5px',
                                                    'text-align': 'left'},
                                       'tabStyle' : {'list-style-type': 'none',
                                                     'margin-bottom': '30px',
                                                     'padding-left': '30px'}},
                             key="1")

"""
# Resultados Control Electoral
"""

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

df = tablas.transmision()
st.metric('Actas ingresadas: ',df.shape[0])

st.write(df)

df = etls.convertir_formato(df)

# Resumen de votos por pregunta
resumen = df.groupby(by='COD_PREGUNTA').sum(numeric_only=True)

for i in range(11):
    st.subheader(Preguntas[numero_letra[i]])
    graficos.resumen_general_pregunta(resumen,i)

# Mostrar datos de transmisión
st.write(df)

df = tablas.provincias()

st.write(df)

df = tablas.juntas()

st.write(df)