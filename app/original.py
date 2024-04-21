    st.header("Análisis de progreso por provincia y resultados")
    for i in range(24):
        try:
            B = cantidad_provincias.iloc[i]
            if int(B['Progress'])>15:
                color_barra = '#ff7f0e'#colores[i]
                st.markdown(
                    f"""
                    <style>
                        #progress-{i} > div > div {{
                            background-color: {color_barra} !important;
                        }}
                    </style>
                    """,
                    unsafe_allow_html=True
                )
            st.progress(int(B['Progress']),text=f"**{B['NOM_PROVINCIA']}:** {B['%']}")
        except:
            pass