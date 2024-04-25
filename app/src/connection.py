import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv('.env')

server = os.getenv('SERVER')
username = os.getenv('USER_NAME')
password = os.getenv('PASSWORD')
driver = os.getenv('DRIVER')
database = os.getenv('DATABASE')


def generate_engine():
   # Construct the connection string
   connection_string = f'mssql+pyodbc://{username}:{password}@{server}/{database}?driver={driver}'
   # Create SQLAlchemy engine
   return create_engine(connection_string)

def query_function(query):
   engine = generate_engine()
   # Execute the query and read into DataFrame
   data = pd.read_sql(query, engine)
   return data