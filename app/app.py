from collections import namedtuple
import math
import pandas as pd
import streamlit as st
import src.connection as connection
import src.tablas as tablas
import src.etls as etls 
import src.graficos as graficos
import src.tiempos as tiempos
from st_on_hover_tabs import on_hover_tabs
import datetime
import pytz

# Funciones para cargar las tablas
@st.cache_data
def cargar_transmision():
    return tablas.transmision()

@st.cache_data
def cargar_provincias():
    return tablas.provincias()

@st.cache_data
def cargar_juntas():
    return tablas.juntas()

@st.cache_data
def cargar_muestra1():
    return tablas.muestra1()

@st.cache_data
def cargar_muestra():
    return tablas.muestra()

@st.cache_data
def cargar_fecha():
    return tiempos.obtener_hora()

# Configuración de página y estilos
st.set_page_config(layout="wide", page_title='Resultados Conteo', page_icon=':white_circle:',
                       initial_sidebar_state="collapsed",  # Opcional: colapsa la barra lateral al inicio
                    menu_items=None ) # Esto quita el menú de opciones (los tres puntos))
st.markdown('<style>' + open('./style.css').read() + '</style>', unsafe_allow_html=True)



# Mostrar datos de provincias
df_provincias = cargar_provincias()
df_provincias_dict = df_provincias[['COD_PROVINCIA', 'NOM_PROVINCIA']].to_dict(orient='records')

def reemplazar_prov(n):
    return df_provincias_dict[n]['NOM_PROVINCIA']

# Mostrar datos de juntas
df_juntas = cargar_juntas()


# Incluir CSS personalizado para ocultar el elemento stDecoration
st.markdown("""
    <style>
        #stDecoration {
            display: none !important;
        }
    </style>
""", unsafe_allow_html=True)
# Incluir CSS personalizado para ocultar el div stToolbar
st.markdown("""
    <style>
        [data-testid="stToolbar"] {
            display: none !important;
        }
    </style>
""", unsafe_allow_html=True)

# Convertir la hora actual a la zona horaria de Ecuador
hora_ecuador = cargar_fecha()

# Sidebar con pestañas
with st.sidebar:
    tabs = on_hover_tabs(tabName=['Dashboard', 'Provincias', 'Muestra'], 
                         iconName=['dashboard', 'monitoring', 'trending_up'],
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
    selected_tab = st.query_params.get("tab", None)
    if selected_tab is None:
        selected_tab = tabs

tit1,tit2 = st.columns([3,1])
# Resultados Control Electoral
tit1.title("Resultados Control Electoral")




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

numero_letra = {
    1: 'A',
    2: 'B',
    3: 'C',
    4: 'D',
    5: 'E',
    6: 'F',
    7: 'G',
    8: 'H',
    9: 'I',
    10: 'J',
    11: 'K'
}

letra_numero = {
    'A':1, 
    'B':2, 
    'C':3, 
    'D':4, 
    'E':5, 
    'F':6, 
    'G':7, 
    'H':8, 
    'I':9, 
    'J':10, 
    'K':11, }

# Carga de datos
if tit2.button("Actualizar Datos"):
    cargar_fecha.clear()
    cargar_transmision.clear()


num_prov = tablas.muestra_provincias()

st.divider()
col1,col2,col3 = st.columns(3)
muestra = cargar_muestra()
muestra1 = cargar_muestra1()
df_transmision = cargar_transmision()
col3.write(f"""
**Ultima actualización:** 
""")
col3.write(hora_ecuador)
col1.metric('Total de actas:',df_transmision.shape[0])
df_transmision = df_transmision[df_transmision['JUNTA_TRANSMITIDA'].isin(muestra)]
lista_juntas_ingresadas = list(df_transmision['JUNTA_TRANSMITIDA'])
juntas_ingresadas = df_juntas[df_juntas['COD_JUNTA'].isin(lista_juntas_ingresadas)]
juntas_ingresadas['NOM_PROVINCIA'] = juntas_ingresadas['COD_PROVINCIA'].astype(int).apply(reemplazar_prov)
cantidad_provincias = juntas_ingresadas.groupby(by='NOM_PROVINCIA').count()[['COD_PROVINCIA']].reset_index()
cantidad_provincias = cantidad_provincias.merge(num_prov,on='NOM_PROVINCIA')
cantidad_provincias['%'] = (cantidad_provincias['COD_PROVINCIA']/cantidad_provincias['CANTIDAD_PROV']).apply(lambda x: f"{x:.2%}")
cantidad_provincias['Progress'] = (cantidad_provincias['COD_PROVINCIA']/cantidad_provincias['CANTIDAD_PROV']).apply(lambda x: int(round(x*100,0)))

col1.metric('Dentro de la muestra:',df_transmision.shape[0])
avance = (df_transmision.shape[0]/len(muestra))*100
col2.progress(value=int(round(avance,0)),
              text=f'###### Avance de actas obtenidas en la muestra: ${round(avance,2)}\%$')




# Resumen de votos por pregunta
df_transmision = etls.convertir_formato(df_transmision)
#pregunta_seleccionada = st.selectbox('Pregunta: ', df_transmision['COD_PREGUNTA'].unique())
resumen = df_transmision.groupby(by='COD_PREGUNTA').sum(numeric_only=True)


st.divider()
# Contenido de la aplicación según la pestaña seleccionada
if selected_tab =='Dashboard':
    st.header("Dashboard de resultados generales a nivel nacional")
    for pregunta in range(11):
        # Mostrar resumen por pregunta
        st.subheader(Preguntas[numero_letra[pregunta+1]])
        ref1,ref2= st.columns(2)
        
        ref1.altair_chart(graficos.resumen_general_pregunta(resumen, pregunta))
        
        sub_col1,sub_col2 = ref2.columns(2)
        sub_col1.altair_chart(graficos.pie_chart(resumen,pregunta))
        A = resumen.iloc[pregunta]
        total = A['SI'] + A['NO']
        sub_col2.markdown('<span style="color:#ff7f0e; font-size: 20px; font-weight: bold;">  </span><span style="font-size: 20px;"></span>', unsafe_allow_html=True)
        sub_col2.markdown(f'<span style="color:#ff7f0e; font-size: 20px; font-weight: bold;">SI: </span><span style="font-size: 20px;">{str(round((A["SI"]/total)*100,2))} %</span>', unsafe_allow_html=True)
        sub_col2.markdown(f'<span style="color:#1f77b4; font-size: 20px; font-weight: bold;">NO: </span><span style="font-size: 20px;">{str(round((A["NO"]/total)*100,2))} %</span>', unsafe_allow_html=True)

        st.divider()

elif selected_tab == 'Provincias':
    st.header("Análisis de progreso por provincia y resultados")
    for i in range(29):
        try:
            B = cantidad_provincias.iloc[i]
            st.progress(int(B['Progress']),text=f"**{B['NOM_PROVINCIA']}:** {B['%']}")
        except:
            pass
    st.divider()
    st.header('Seleccione la provincia y la pregunta para ver los resultados')
    col1,col2 = st.columns(2)
    provincia = col1.selectbox(label='##### Provincia: ',options=list(cantidad_provincias['NOM_PROVINCIA'].unique())) 
    pregu = col2.selectbox(label='##### Pregunta: ',options=list(Preguntas.keys()) )
    if provincia != None and pregu!= None:
        muestra_provincia = tablas.muestra_lista_provincia(provincia)
        prov_transmision = df_transmision[df_transmision['JUNTA_TRANSMITIDA'].isin(muestra_provincia)]
        resumen_prov = prov_transmision[prov_transmision['COD_PREGUNTA']==letra_numero[pregu]].sum(numeric_only=True)
        resumen_prov = pd.DataFrame(resumen_prov).transpose()


        st.subheader(Preguntas[pregu])
        ref1,ref2= st.columns(2)

        ref1.altair_chart(graficos.resumen_general_pregunta(resumen_prov, 0))
        
        sub_col1,sub_col2 = ref2.columns(2)
        sub_col1.altair_chart(graficos.pie_chart(resumen_prov,0))
        A = resumen_prov.iloc[0]
        total = A['SI'] + A['NO']
        sub_col2.markdown('<span style="color:#ff7f0e; font-size: 20px; font-weight: bold;">  </span><span style="font-size: 20px;"></span>', unsafe_allow_html=True)
        sub_col2.markdown(f'<span style="color:#ff7f0e; font-size: 20px; font-weight: bold;">SI: </span><span style="font-size: 20px;">{str(round((A["SI"]/total)*100,2))} %</span>', unsafe_allow_html=True)
        sub_col2.markdown(f'<span style="color:#1f77b4; font-size: 20px; font-weight: bold;">NO: </span><span style="font-size: 20px;">{str(round((A["NO"]/total)*100,2))} %</span>', unsafe_allow_html=True)

        st.divider()
elif selected_tab == 'Muestra':
    st.header("Proyección de resultados a partir de muestra matemática")
    


