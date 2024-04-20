# Usar python:3.10 como imagen base
FROM python:3.10

# Establecer el directorio de trabajo en /app
WORKDIR /app

# Instalar dependencias necesarias para el sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Clonar el repositorio (asegúrate de que esta es la ruta correcta y el repositorio es accesible)
RUN git clone https://github.com/DobleL2/ELECCIONES_2024.git .

# Crear un entorno virtual e instalar dependencias
RUN python -m venv venv
ENV PATH="/app/venv/bin:$PATH"

# Actualizar pip e instalar dependencias
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install streamlit

# Exponer el puerto 8501
EXPOSE 8501

# Comprobar la ubicación de streamlit y asignar permisos de ejecución
RUN which streamlit
RUN chmod +x $(which streamlit)

# Establecer el comando de salud
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Configurar el punto de entrada para iniciar Streamlit
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
