import streamlit as st
import altair as alt
import pandas as pd


def resumen_general_pregunta(resumen,pregunta):
    A = resumen.iloc[pregunta]

    source = pd.DataFrame({
        'Opciones': ['BLANCOS','NULOS','SI','NO'],
        'Valores': [A['BLANCOS'],A['NULOS'],A['SI'],A['NO']],
    })

    # Crear gráfico de barras
    bar_chart = alt.Chart(source).mark_bar().encode(
        x='Opciones',
        y='Valores',
        color=alt.Color('Opciones', scale=alt.Scale(scheme='set1')),  # Modificar colores de las barras
    )

    # Agregar texto con los valores en las barras
    text = bar_chart.mark_text(
        align='center',
        baseline='middle',
        dy=-15,  # Ajustar posición vertical del texto
        color='black',  # Color del texto
        size=14  # Tamaño del texto
    ).encode(
        text='Valores'  # Mostrar los valores en las barras
    )

    # Combinar gráfico de barras y texto
    bar_chart_with_text = (bar_chart + text).properties(
        width=alt.Step(80),  # Modificar el ancho del gráfico
        height=alt.Step(200)  # Modificar la altura del gráfico
    )

    # Mostrar gráfico de barras con valores
    return st.altair_chart(bar_chart_with_text)