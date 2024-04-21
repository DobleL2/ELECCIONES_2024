import pandas as pd
import altair as alt

def resumen_general_pregunta(resumen, pregunta):
    A = resumen.iloc[pregunta]

    source = pd.DataFrame({
        'Opciones': ['BLANCOS', 'NULOS', 'SI', 'NO'],
        'Valores': [A['BLANCOS'], A['NULOS'], A['SI'], A['NO']],
    })

    # Crear gráfico de barras
    bar_chart = alt.Chart(source).mark_bar().encode(
        x=alt.X('Opciones', axis=alt.Axis(title=None)),  # Quitar el título del eje X
        y=alt.Y('Valores', axis=alt.Axis(title=None)),   # Quitar el título del eje Y
        color=alt.Color('Opciones', scale=alt.Scale(scheme='set1'), legend=None),  # Modificar colores de las barras
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
        autosize=alt.AutoSizeParams(
            type='fit',  # Ajustar el gráfico para que quepa en el contenedor
            resize=True  # Permitir que el gráfico se ajuste automáticamente al cambiar el tamaño de la ventana del navegador
        ),
        width = 300,
        height=300 # Altura fija
        
    )

    return bar_chart_with_text


def pie_chart(resumen, pregunta):
    A = resumen.iloc[pregunta]

    # Filtrar las categorías 'SI' y 'NO'
    filtered_data = {
        'Opciones': ['SI', 'NO'],
        'Valores': [A['SI'], A['NO']]
    }

    source = pd.DataFrame(filtered_data)

    # Crear el gráfico circular (pie chart)
    pie_chart = alt.Chart(source).mark_arc().encode(
        angle='Valores',  # Ángulo de cada sector basado en los valores
        color=alt.Color('Opciones', legend=None),  # Color de cada sector basado en la categoría
        tooltip=['Opciones', 'Valores']  # Tooltip que muestra la categoría y el valor
    ).properties(
        width=300,  # Ancho del gráfico
        height=300,  # Alto del gráfico
        title='Pie Chart'  # Título del gráfico
    )

    return pie_chart