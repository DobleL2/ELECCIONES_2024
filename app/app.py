from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
import connection as connection

"""
# Welcome to Streamlit!

Edit `/streamlit_app.py` to customize this app to your heart's desire :heart:

If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
forums](https://discuss.streamlit.io).

In the meantime, below is an example of what you can do with just a few lines of code:
"""

with st.echo(code_location='below'):
   total_points = st.slider("Number of points in spiral", 1, 5000, 2000)
   num_turns = st.slider("Number of turns in spiral", 1, 100, 9)

   Point = namedtuple('Point', 'x y')
   data = []

   points_per_turn = total_points / num_turns

   for curr_point_num in range(total_points):
      curr_turn, i = divmod(curr_point_num, points_per_turn)
      angle = (curr_turn + 1) * 2 * math.pi * i / points_per_turn
      radius = curr_point_num / total_points
      x = radius * math.cos(angle)
      y = radius * math.sin(angle)
      data.append(Point(x, y))

   st.altair_chart(alt.Chart(pd.DataFrame(data), height=500, width=500)
      .mark_circle(color='#0068c9', opacity=0.5)
      .encode(x='x:Q', y='y:Q'))
   
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
    conn = connection()  # Assuming you have the connection function defined elsewhere
    data = pd.read_sql(query, conn)  # Execute the query and read into DataFrame
    conn.close()
    return data

# Execute query
query = "SELECT TOP 100 * FROM dbo.JUNTA"
juntas = query_function(query)

st.write(juntas)
