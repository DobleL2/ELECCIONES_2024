import src.connection as connection
import pandas as pd

def transmision():
    query = """
        SELECT * 
        FROM dbo.TRANSMISION_CR_3_SIMULACION
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