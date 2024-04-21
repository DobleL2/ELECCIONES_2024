import src.connection as connection

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