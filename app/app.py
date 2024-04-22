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
import numpy as np
import scipy.stats as stats
import altair as alt





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

# Cargar y mostrar el logo
logo_image = 'images/logo.png'  # Cambia 'ruta_del_logo.png' por la ruta de tu archivo de imagen del logo

tit1,tit2,tit3 = st.columns([1,3,1])
# Resultados Control Electoral
tit1.image(logo_image, width=150)  # Ajusta el ancho según sea necesario
tit2.title("Resultados Conteo Rápido")
tit2.subheader("Muestra Matemática")



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
if tit3.button("Actualizar Datos"):
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
        if pregunta == 0:
            st.title('Referendum')
        if pregunta == 5:
            st.title('Consulta Popular')
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
            st.progress(int(B['Progress']),text=f"**{B['NOM_PROVINCIA']}:** {B['%']} "+"   . . . .   "+    f"({B['COD_PROVINCIA']}/{B['CANTIDAD_PROV']})")
        except:
            pass
    st.divider()
    st.header('Seleccione la provincia y la pregunta para ver los resultados')
    #col1,col2 = st.columns(2)
    provincia = st.selectbox(label='##### Provincia: ',options=list(cantidad_provincias['NOM_PROVINCIA'].unique())) 
    #pregu = col2.selectbox(label='##### Pregunta: ',options=list(Preguntas.keys()) )
    if provincia != None:
        for pregu in list(Preguntas.keys()):
            if pregu == 'A':
                st.title('Referendum')
            if pregu == 'F':
                st.title('Consulta Popular')
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
    st.subheader('Seleccione la provincia y la pregunta para ver los resultados')
    col1,col2 = st.columns(2)

    provincia = col1.selectbox(label='##### Provincia: ',options=['NACIONAL']+list(cantidad_provincias['NOM_PROVINCIA'].unique()))
    pregu = col2.selectbox(label='##### Pregunta: ',options=list(Preguntas.keys()) )
    ordenada = df_transmision
    ordenada = ordenada[ordenada['COD_PREGUNTA']==letra_numero[pregu]]
    if provincia != 'NACIONAL':
        actas_prov = tablas.muestra_lista_provincia(provincia)
        ordenada = ordenada[ordenada['JUNTA_TRANSMITIDA'].isin(actas_prov)]
    ordenada['FECHA_HORA'] = pd.to_datetime(ordenada['FECHA_HORA'])
    ordenada.sort_values(by='FECHA_HORA',inplace=True)
    # Obtener automáticamente el mínimo de la columna de fecha


    def redondear_hora_al_inmediato_inferior(dt):
        return pd.Timestamp(dt).floor('5min')
    ordenada['TIEMPO'] = ordenada['FECHA_HORA'].apply(redondear_hora_al_inmediato_inferior)
    #ordenada.groupby(by='TIEMPO')

    ordenada = ordenada[['BLANCOS','NULOS','SI','NO','TIEMPO']].groupby(by='TIEMPO').sum().reset_index()

    for i in ['BLANCOS','NULOS','SI','NO']:
        ordenada[i] = ordenada[i].cumsum()
    ordenada['TOTAL_T'] = ordenada['BLANCOS']+ordenada['NULOS']+ordenada['SI']+ordenada['NO']
    ordenada['TOTAL_SN'] =ordenada['SI']+ordenada['NO']
    ordenada['p_SI_T'] = ordenada['SI']/ordenada['TOTAL_T']
    ordenada['p_SI_SN'] = ordenada['SI']/ordenada['TOTAL_SN']
    ordenada['p_NO_T'] = ordenada['NO']/ordenada['TOTAL_T']
    ordenada['p_NO_SN'] = ordenada['NO']/ordenada['TOTAL_SN']
    def calcular_intervalo_confianza_proporcion(proporcion, tamaño_muestra, nivel_confianza=0.99):
        """
        Calcula el intervalo de confianza para una proporción utilizando la fórmula binomial.

        Args:
        - proporcion (float): La proporción de interés.
        - tamaño_muestra (int): El tamaño de la muestra.
        - nivel_confianza (float, opcional): El nivel de confianza deseado. Por defecto es 0.95.

        Returns:
        - tuple: Un tuple con el intervalo de confianza inferior y superior.
        """
        # Calcular el error estándar de la proporción
        error_estandar = np.sqrt(proporcion * (1 - proporcion) / tamaño_muestra)

        # Calcular el valor z crítico para el nivel de confianza dado
        valor_z = stats.norm.ppf((1 + nivel_confianza) / 2)

        # Calcular los límites del intervalo de confianza
        limite_inferior = proporcion - valor_z * error_estandar
        limite_superior = proporcion + valor_z * error_estandar

        return limite_inferior, limite_superior, error_estandar

    def calcular_intervalo_fila(fila):
        prop = fila['p_SI_T']
        tam_muestra = fila['TOTAL_T']
        limite_inferior_si_t, limite_superior_si_t, error_estandar_si_t = calcular_intervalo_confianza_proporcion(prop, tam_muestra)
        prop = fila['p_NO_T']
        tam_muestra = fila['TOTAL_T']
        limite_inferior_no_t, limite_superior_no_t, error_estandar_no_t = calcular_intervalo_confianza_proporcion(prop, tam_muestra)
        prop = fila['p_SI_SN']
        tam_muestra = fila['TOTAL_SN']
        limite_inferior_si_sn, limite_superior_si_sn, error_estandar_si_sn = calcular_intervalo_confianza_proporcion(prop, tam_muestra)
        prop = fila['p_NO_SN']
        tam_muestra = fila['TOTAL_SN']
        limite_inferior_no_sn, limite_superior_no_sn, error_estandar_no_sn = calcular_intervalo_confianza_proporcion(prop, tam_muestra)
        return pd.Series({'LIM_INF_SI_T': limite_inferior_si_t, 'LIM_SUP_SI_T': limite_superior_si_t, 'ERROR_ESTANDAR_SI_T': error_estandar_si_t,'LIM_INF_NO_T': limite_inferior_no_t, 'LIM_SUP_NO_T': limite_superior_no_t, 'ERROR_ESTANDAR_NO_T': error_estandar_no_t,'LIM_INF_SI_SN': limite_inferior_si_sn, 'LIM_SUP_SI_SN': limite_superior_si_sn, 'ERROR_ESTANDAR_SI_SN': error_estandar_si_sn,'LIM_INF_NO_SN': limite_inferior_no_sn, 'LIM_SUP_NO_SN': limite_superior_no_sn, 'ERROR_ESTANDAR_NO_SN': error_estandar_no_sn})
    # Aplicar la función a cada fila del DataFrame
    intervalos_confianza = ordenada.apply(calcular_intervalo_fila, axis=1)

    # Unir los resultados al DataFrame original
    df_con_intervalos = pd.concat([ordenada, intervalos_confianza], axis=1)
    ordenada = df_con_intervalos
    

    st.title(Preguntas[pregu])
    st.header('Analisis del SI')
    col1,col2 =st.columns(2)
    col1.header('Intervalos de confianza para la estabilización de resultados generales')
    # Creamos los datos
    source = pd.DataFrame({
        "yield_error": ordenada['ERROR_ESTANDAR_SI_T'],
        "yield_center": ordenada['p_SI_T'],
        "variety": ordenada['TIEMPO'],
    })

    # Creamos la visualización
    bar = alt.Chart(source).mark_errorbar().encode(
        x=alt.X("yield_center:Q").scale(zero=False).title("yield"),
        xError=("yield_error:Q"),
        y=alt.Y("variety:N", title="Variety"),  # Cambiamos el título del eje y
    )

    point = alt.Chart(source).mark_point(
        filled=True,
        color="black"
    ).encode(
        alt.X("yield_center:Q"),
        alt.Y("variety:N"),
    )

    # Renombramos los valores del eje y
    bar = bar.transform_calculate(
        variety_label="datum.variety"  # Puedes ajustar esta transformación según tus necesidades
    ).encode(
        y=alt.Y("variety:N", axis=alt.Axis(title="Variety", labels=False), sort=None),
        text=alt.Text("variety_label:N")
    )
    def imprimir_en_porcentaje(valor):
        porcentaje = valor * 100
        return "{:.2f}%".format(porcentaje)
    # Mostramos el gráfico en Streamlit
    col1.altair_chart(point + bar)
    sub1,sub2,sub3 = col1.columns(3)
    sub1.metric('LIM INF',imprimir_en_porcentaje(list(ordenada['LIM_INF_SI_T'])[-1]))
    sub2.metric('VALOR',imprimir_en_porcentaje(list(ordenada['p_SI_T'])[-1]))
    sub3.metric('LIM INF',imprimir_en_porcentaje(list(ordenada['LIM_SUP_SI_T'])[-1]))
    col2.header('Intervalos de confianza para la estabilización de resultados VOTOS VALIDOS')
    # Creamos los datos
    source = pd.DataFrame({
        "yield_error": ordenada['ERROR_ESTANDAR_SI_SN'],
        "yield_center": ordenada['p_SI_SN'],
        "variety": ordenada['TIEMPO'],
    })

    # Creamos la visualización
    bar = alt.Chart(source).mark_errorbar().encode(
        x=alt.X("yield_center:Q").scale(zero=False).title("yield"),
        xError=("yield_error:Q"),
        y=alt.Y("variety:N", title="Variety"),  # Cambiamos el título del eje y
    )

    point = alt.Chart(source).mark_point(
        filled=True,
        color="black"
    ).encode(
        alt.X("yield_center:Q"),
        alt.Y("variety:N"),
    )

    # Renombramos los valores del eje y
    bar = bar.transform_calculate(
        variety_label="datum.variety"  # Puedes ajustar esta transformación según tus necesidades
    ).encode(
        y=alt.Y("variety:N", axis=alt.Axis(title="Variety", labels=False), sort=None),
        text=alt.Text("variety_label:N")
    )

    # Mostramos el gráfico en Streamlit
    col2.altair_chart(point + bar)
    sub1,sub2,sub3 = col2.columns(3)
    sub1.metric('LIM INF',imprimir_en_porcentaje(list(ordenada['LIM_INF_SI_SN'])[-1]))
    sub2.metric('VALOR',imprimir_en_porcentaje(list(ordenada['p_SI_SN'])[-1]))
    sub3.metric('LIM INF',imprimir_en_porcentaje(list(ordenada['LIM_SUP_SI_SN'])[-1]))
    st.divider()
    st.header('Analisis del NO')
    col1,col2 =st.columns(2)
    col1.header('Intervalos de confianza para la estabilización de resultados generales')
    # Creamos los datos
    source = pd.DataFrame({
        "yield_error": ordenada['ERROR_ESTANDAR_NO_T'],
        "yield_center": ordenada['p_NO_T'],
        "variety": ordenada['TIEMPO'],
    })

    # Creamos la visualización
    bar = alt.Chart(source).mark_errorbar().encode(
        x=alt.X("yield_center:Q").scale(zero=False).title("yield"),
        xError=("yield_error:Q"),
        y=alt.Y("variety:N", title="Variety"),  # Cambiamos el título del eje y
    )

    point = alt.Chart(source).mark_point(
        filled=True,
        color="black"
    ).encode(
        alt.X("yield_center:Q"),
        alt.Y("variety:N"),
    )

    # Renombramos los valores del eje y
    bar = bar.transform_calculate(
        variety_label="datum.variety"  # Puedes ajustar esta transformación según tus neceNOdades
    ).encode(
        y=alt.Y("variety:N", axis=alt.Axis(title="Variety", labels=False), sort=None),
        text=alt.Text("variety_label:N")
    )
    def imprimir_en_porcentaje(valor):
        porcentaje = valor * 100
        return "{:.2f}%".format(porcentaje)
    # Mostramos el gráfico en Streamlit
    col1.altair_chart(point + bar)
    sub1,sub2,sub3 = col1.columns(3)
    sub1.metric('LIM INF',imprimir_en_porcentaje(list(ordenada['LIM_INF_NO_T'])[-1]))
    sub2.metric('VALOR',imprimir_en_porcentaje(list(ordenada['p_NO_T'])[-1]))
    sub3.metric('LIM INF',imprimir_en_porcentaje(list(ordenada['LIM_SUP_NO_T'])[-1]))
    col2.header('Intervalos de confianza para la estabilización de resultados VOTOS VALIDOS')
    # Creamos los datos
    source = pd.DataFrame({
        "yield_error": ordenada['ERROR_ESTANDAR_NO_SN'],
        "yield_center": ordenada['p_NO_SN'],
        "variety": ordenada['TIEMPO'],
    })

    # Creamos la visualización
    bar = alt.Chart(source).mark_errorbar().encode(
        x=alt.X("yield_center:Q").scale(zero=False).title("yield"),
        xError=("yield_error:Q"),
        y=alt.Y("variety:N", title="Variety"),  # Cambiamos el título del eje y
    )

    point = alt.Chart(source).mark_point(
        filled=True,
        color="black"
    ).encode(
        alt.X("yield_center:Q"),
        alt.Y("variety:N"),
    )

    # Renombramos los valores del eje y
    bar = bar.transform_calculate(
        variety_label="datum.variety"  # Puedes ajustar esta transformación según tus neceNOdades
    ).encode(
        y=alt.Y("variety:N", axis=alt.Axis(title="Variety", labels=False), sort=None),
        text=alt.Text("variety_label:N")
    )

    # Mostramos el gráfico en Streamlit
    col2.altair_chart(point + bar)
    sub1,sub2,sub3 = col2.columns(3)
    sub1.metric('LIM INF',imprimir_en_porcentaje(list(ordenada['LIM_INF_NO_SN'])[-1]))
    sub2.metric('VALOR',imprimir_en_porcentaje(list(ordenada['p_NO_SN'])[-1]))
    sub3.metric('LIM INF',imprimir_en_porcentaje(list(ordenada['LIM_SUP_NO_SN'])[-1]))