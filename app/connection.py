import pyodbc
import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
from sqlalchemy.orm import sessionmaker

load_dotenv('.env')

server = os.getenv('SERVER')
username ='test2' #os.getenv('USER')
password = os.getenv('PASSWORD')
driver= os.getenv('DRIVER')
database = os.getenv('DATABASE')

def connection():
    conn = pyodbc.connect(f'DRIVER={driver};SERVER={server};PORT=1433;DATABASE={database};UID={username};PWD={password}')
    print("Connection successful!")
    return conn

def query_function(query):
    conn = connection()
    cursor = conn.cursor()
    cursor.execute(query)  # Replace with your actual SQL query
    data = pd.read_sql_query(sql=query, con=conn)
    conn.close()
    return data