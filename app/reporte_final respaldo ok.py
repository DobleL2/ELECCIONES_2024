
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

# Mostrar datos de provincias
df_provincias = tablas.provincias()
df_provincias_dict = df_provincias[['COD_PROVINCIA', 'NOM_PROVINCIA']].to_dict(orient='records')

def reemplazar_prov(n):
    return df_provincias_dict[n]['NOM_PROVINCIA']

# Mostrar datos de juntas
df_juntas = tablas.juntas()



num_prov = tablas.muestra_provincias()


muestra = tablas.muestra()
muestra1 = tablas.muestra1()
df_transmision = tablas.transmision()
df_transmision = df_transmision[df_transmision['JUNTA_TRANSMITIDA'].isin(muestra)]
lista_juntas_ingresadas = list(df_transmision['JUNTA_TRANSMITIDA'])

juntas_ingresadas = df_juntas[df_juntas['COD_JUNTA'].isin(lista_juntas_ingresadas)]
juntas_ingresadas['NOM_PROVINCIA'] = juntas_ingresadas['COD_PROVINCIA'].astype(int).apply(reemplazar_prov)
cantidad_provincias = juntas_ingresadas.groupby(by='NOM_PROVINCIA').count()[['COD_PROVINCIA']].reset_index()
cantidad_provincias = cantidad_provincias.merge(num_prov,on='NOM_PROVINCIA')
cantidad_provincias['%'] = (cantidad_provincias['COD_PROVINCIA']/cantidad_provincias['CANTIDAD_PROV']).apply(lambda x: f"{x:.2%}")
cantidad_provincias['Progress'] = (cantidad_provincias['COD_PROVINCIA']/cantidad_provincias['CANTIDAD_PROV']).apply(lambda x: int(round(x*100,0)))

# Resumen de votos por pregunta
df_transmision = etls.convertir_formato(df_transmision)
#pregunta_seleccionada = st.selectbox('Pregunta: ', df_transmision['COD_PREGUNTA'].unique())
resumen = df_transmision.groupby(by='COD_PREGUNTA').sum(numeric_only=True)


PROVINCIAS = ['NACIONAL']+list(cantidad_provincias['NOM_PROVINCIA'].unique())

PREGUNTAS = list(Preguntas.keys())


def metricas(provincia,pregu):
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
        

    
    ultima = ordenada.iloc[-1]
    
    return ordenada,ultima['LIM_INF_SI_SN'],ultima['LIM_SUP_SI_SN'],ultima['p_SI_SN'],ultima['LIM_INF_NO_SN'],ultima['LIM_SUP_NO_SN'],ultima['p_NO_SN']

def imprimir_en_porcentaje(valor):
    porcentaje = valor * 100
    return "{:.2f}%".format(porcentaje)
def grafico(opcion,pregunta,provincia,ordenada):
    source = pd.DataFrame({
        "yield_error": ordenada[f'ERROR_ESTANDAR_{opcion}_SN'],
        "yield_center": ordenada[f'p_{opcion}_SN'],
        "variety": ordenada['TIEMPO'],
    })

    bar = alt.Chart(source).mark_errorbar().encode(
        y=alt.Y("yield_center:Q").scale(zero=False).title("yield"),  # Cambiar a eje y
        yError=("yield_error:Q"),  # Error ahora es para el eje y
        x=alt.X("variety:N", axis=alt.Axis(labelAngle=-85, labelLimit=5)),  # Cambiar a eje x
    )

    point = alt.Chart(source).mark_point(
        filled=True,
        color="black"
    ).encode(
        alt.Y("yield_center:Q"),  # Cambiar a eje y
        alt.X("variety:N"),  # Cambiar a eje x
    )

    combined_chart = (bar + point).properties(
        width=700,
        height=200,
        title='Proporción con intervalo de error'
    ).configure_axisX(labelAngle=-45) 

    combined_chart.save(f'images/tendencia/{provincia}_{str(pregunta)}_{opcion}.png')



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
    
        
pdf = PDFWithBackground()
pdf.set_background('images/portada.png')
pdf.add_page()

 
def solo_hora(n):
    return n



for provincia in PROVINCIAS:
    for pregunta in PREGUNTAS:
        ordenada_new,lim_inf_si,lim_sup_si,value_si,lim_inf_no,lim_sup_no,value_no = metricas(provincia,pregunta)
        lim_inf_si = imprimir_en_porcentaje(float(lim_inf_si))
        lim_inf_no = imprimir_en_porcentaje(float(lim_inf_no))
        lim_sup_si = imprimir_en_porcentaje(float(lim_sup_si))
        lim_sup_no = imprimir_en_porcentaje(float(lim_sup_no))
        value_si = imprimir_en_porcentaje(float(value_si))
        value_no = imprimir_en_porcentaje(float(value_no))
        opciones = ['SI','NO']
        for opt in opciones:
            grafico(opt,pregunta,provincia,ordenada_new)
        image_si = provincia+'_'+str(pregunta)+'_'+'SI'
        image_no = provincia+'_'+str(pregunta)+'_'+'NO'
        df1 = ordenada_new[['TIEMPO','LIM_INF_SI_SN','p_SI_SN','LIM_SUP_SI_SN']][-16:]
        df1['LIM_INF_SI_SN'] = df1['LIM_INF_SI_SN'].astype(float).apply(imprimir_en_porcentaje)
        df1['p_SI_SN'] = df1['p_SI_SN'].astype(float).apply(imprimir_en_porcentaje)
        df1['LIM_SUP_SI_SN'] = df1['LIM_SUP_SI_SN'].astype(float).apply(imprimir_en_porcentaje)
        df1 = df1.astype(str)
        df1['TIEMPO'] = df1['TIEMPO'].astype(str).apply(solo_hora)
        df2 = ordenada_new[['TIEMPO','LIM_INF_NO_SN','p_NO_SN','LIM_SUP_NO_SN']][-16:]
        df2['LIM_INF_NO_SN'] = df2['LIM_INF_NO_SN'].astype(float).apply(imprimir_en_porcentaje)
        df2['p_NO_SN'] = df2['p_NO_SN'].astype(float).apply(imprimir_en_porcentaje)
        df2['LIM_SUP_NO_SN'] = df2['LIM_SUP_NO_SN'].astype(float).apply(imprimir_en_porcentaje)
        df2['TIEMPO'] = df2['TIEMPO'].astype(str).apply(solo_hora)
        df2 = df2.astype(str)
        df1.columns = ['HORA','LIM_INF','VALOR','LIM_SUP']
        df2.columns = ['HORA','LIM_INF','VALOR','LIM_SUP']
        
        pdf.set_background('images/background.jpeg')
        pdf.add_page()
  
        pdf.set_xy(15,15)
        pdf.add_font('Anton-Regular', '', 'fonts/Anton-Regular.ttf', uni=True)  # Register the custom font
        pdf.set_font('Helvetica','B',size=30)   # Helvetica, Times, Courier
        pdf.cell(0,0,'Resultados Conteo Rápido',0,1,'L')

        pdf.set_xy(15,24)
        pdf.set_font('Helvetica', '', 15)  # Use the registered font
        pdf.cell(0,0,'Muestra Matemática',0,1,'L')

        pdf.set_xy(15,35)
        pdf.set_font('Helvetica', 'B', 18)  # Use the registered font
        pdf.cell(0,0,'Jurisdicción:',0,1,'L')

        pdf.set_xy(56,35)
        pdf.set_font('Helvetica', '', 16)  # Use the registered font
        pdf.cell(0,0,provincia,0,1,'L')

        pdf.set_xy(15,43)
        pdf.set_font('Helvetica', 'B', 18)  # Use the registered font
        pdf.cell(0,0,'Pregunta:',0,1,'L')
        
        pdf.set_xy(47,43)
        pdf.set_font('Helvetica', '', 16)  # Use the registered font
        pdf.cell(0,0,Preguntas[pregunta],0,1,'L')

        imagen = f'images/tendencia/{image_si}.png'
        pdf.image(f'{imagen}',x=15,y=55,w=145,h=65)

        pdf.set_xy(160,65)
        pdf.set_font('Helvetica', 'B', 11)  # Use the registered font
        pdf.cell(0,0,'Límite',0,1,'L')
        pdf.set_xy(160,70)
        pdf.cell(0,0,'superior',0,1,'L')

        pdf.set_xy(160,83)
        pdf.set_font('Helvetica', 'B', 11)  # Use the registered font
        pdf.cell(0,0,'Valor',0,1,'L')

        pdf.set_xy(160,96)
        pdf.set_font('Helvetica', 'B', 11)  # Use the registered font
        pdf.cell(0,0,'Límite',0,1,'L')
        pdf.set_xy(160,101)
        pdf.cell(0,0,'inferior',0,1,'L')

        #Valores
        pdf.set_xy(180,67)
        pdf.set_font('Helvetica',size= 11)  # Use the registered font
        pdf.cell(0,0,f'{lim_sup_si}',0,1,'L')

        pdf.set_xy(180,83)
        pdf.set_font('Helvetica',size= 11)  # Use the registered font
        pdf.cell(0,0,f'{value_si}',0,1,'L')

        pdf.set_xy(180,98)
        pdf.set_font('Helvetica',size= 11)  # Use the registered font
        pdf.cell(0,0,f'{lim_inf_si}',0,1,'L')


        imagen = f'images/tendencia/{image_no}.png'
        pdf.image(f'{imagen}',x=15,y=125,w=145,h=65)

        pdf.set_xy(160,125+10)
        pdf.set_font('Helvetica', 'B', 11)  # Use the registered font
        pdf.cell(0,0,'Límite',0,1,'L')
        pdf.set_xy(160,130+10)
        pdf.cell(0,0,'superior',0,1,'L')

        pdf.set_xy(160,143+10)
        pdf.set_font('Helvetica', 'B', 11)  # Use the registered font
        pdf.cell(0,0,'Valor',0,1,'L')

        pdf.set_xy(160,156+10)
        pdf.set_font('Helvetica', 'B', 11)  # Use the registered font
        pdf.cell(0,0,'Límite',0,1,'L')
        pdf.set_xy(160,161+10)
        pdf.cell(0,0,'inferior',0,1,'L')

        #Valores
        pdf.set_xy(180,67+70)
        pdf.set_font('Helvetica',size= 11)  # Use the registered font
        pdf.cell(0,0,f'{lim_sup_no}',0,1,'L')

        pdf.set_xy(180,83+70)
        pdf.set_font('Helvetica',size= 11)  # Use the registered font
        pdf.cell(0,0,f'{value_no}',0,1,'L')

        pdf.set_xy(180,98+70)
        pdf.set_font('Helvetica',size= 11)  # Use the registered font
        pdf.cell(0,0,f'{lim_inf_no}',0,1,'L')



        pdf.set_xy(45,200)
        pdf.set_font('Helvetica', 'B', 10)  # Use the registered font
        pdf.cell(0,0,'Evolución tendencia SI',0,1,'L')

        pdf.set_xy(130,200)
        pdf.set_font('Helvetica', 'B', 10)  # Use the registered font
        pdf.cell(0,0,'Evolución tendencia NO',0,1,'L')

        df = df1
        x = 25
        y = 205
        pdf.set_xy(x,y)
        pdf.set_font("Helvetica", size=6)
        table_data = [df.columns.tolist()] + df.values.tolist()

        with pdf.table(width=70,col_widths=(25,15, 15, 15), align='L') as table:
            for data_row in table_data:
                row = table.row()
                for datum in data_row:
                    row.cell(datum)
        
        df = df2
        x = 110
        y = 205
        pdf.set_xy(x,y)
        pdf.set_font("Helvetica", size=6)
        table_data = [df.columns.tolist()] + df.values.tolist()

        with pdf.table(width=70,col_widths=(25,15, 15, 15), align='L') as table:
            for data_row in table_data:
                row = table.row()
                for datum in data_row:
                    row.cell(datum)

pdf.output('reporte_final_a_nivel_nacional_preliminar.pdf')