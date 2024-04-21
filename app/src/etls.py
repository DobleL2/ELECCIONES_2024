import pandas as pd

def convertir_formato(actas):
    base_vertical = pd.DataFrame()
    for i in range(1,12):
        aux = actas[['COD_TRANSMISION',
                    'CEDULA_TRANSMITIO',
                    'JUNTA_TRANSMITIDA',
                    'TOTAL_SUFRAGANTES',
                    f'P{str(i)}_BLANCOS',
                    f'P{str(i)}_NULOS',
                    f'P{str(i)}_SI',
                    f'P{str(i)}_NO',
                    f'P{str(i)}_VALIDA',
                    'FECHA_HORA']].copy()
        aux['COD_PREGUNTA'] = i
        aux.columns = ['COD_TRANSMISION',
                    'CEDULA_TRANSMITIO',
                    'JUNTA_TRANSMITIDA',
                    'TOTAL_SUFRAGANTES',
                    'BLANCOS',
                    'NULOS',
                    'SI',
                    'NO',
                    'VALIDA',
                    'FECHA_HORA',
                    'COD_PREGUNTA']
        aux = aux[['COD_PREGUNTA','COD_TRANSMISION',
                    'CEDULA_TRANSMITIO',
                    'JUNTA_TRANSMITIDA',
                    'TOTAL_SUFRAGANTES',
                    'BLANCOS',
                    'NULOS',
                    'SI',
                    'NO',
                    'VALIDA',
                    'FECHA_HORA']]
        base_vertical = pd.concat([base_vertical,aux])
    print("--------------- SE HA EXPORTADO LA BASE EN EL FORMATO CORRESPONDIENTE ----------------")
    #base_vertical = base_vertical[base_vertical['VALIDA']==1].drop('VALIDA')
    base_vertical = base_vertical.sort_values(by=['JUNTA_TRANSMITIDA','COD_PREGUNTA']).reset_index(drop=True)
    #base_vertical.to_csv('Resultados/Test1.csv',index=False,sep=';')
    return base_vertical
