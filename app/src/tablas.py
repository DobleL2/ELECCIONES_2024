import src.connection as connection
import pandas as pd

def transmision():
    query = """
        SELECT * 
        FROM dbo.TRANSMISION_CR_3
    """
    return connection.query_function(query=query)

def provincias():
    query = """
        SELECT *
        FROM dbo.PROVINCIA
    """
    return connection.query_function(query=query)

def juntas():
    query = """
        SELECT *
        FROM dbo.JUNTA
    """
    return connection.query_function(query=query)

def muestra1():
    return list(pd.read_csv('data/Muestra_1.csv',sep=';')['COD_JUNTA'])

def muestra():
    return list(pd.read_csv('data/Muestra.csv',sep=';')['COD_JUNTA'])

def muestra_provincias():
    muest = pd.read_csv('data/Muestra.csv',sep=';')
    muest = muest.groupby(by='NOM_PROVINCIA').count()[['COD_PROVINCIA']].reset_index()
    muest.columns = ['NOM_PROVINCIA','CANTIDAD_PROV']
    return muest

def muestra_lista_provincia(name):
    aux = pd.read_csv('data/Muestra.csv',sep=';')
    aux = aux[aux['NOM_PROVINCIA']==name]['COD_JUNTA']
    return list(aux)