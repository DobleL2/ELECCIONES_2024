# Usar python:3.10 como imagen base
FROM python:3.10

# Establecer el directorio de trabajo en /app
WORKDIR /app

# Copiar los archivos de la aplicación
COPY ./app /app

# Crear un entorno virtual e instalar dependencias
RUN python -m venv venv
ENV PATH="/app/venv/bin:$PATH"

# Instalar dependencias
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install streamlit

# Exponer el puerto 8501
EXPOSE 8501

# Establecer el comando de inicio
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]