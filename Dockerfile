# app/Dockerfile

FROM python:3.10

WORKDIR /app

# Instala dependencias necesarias para el sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Clona el repositorio de GitHub
RUN git clone https://github.com/DobleL2/ELECCIONES_2024.git .

# Crea un entorno virtual dentro del contenedor y activa el entorno
RUN python -m venv venv
ENV PATH="/app/venv/bin:$PATH"

# Instala dependencias dentro del entorno virtual
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install streamlit

# Comprueba la instalación de Streamlit y da permisos de ejecución
RUN which streamlit
RUN chmod +x $(which streamlit)

EXPOSE 8501

# Realiza una comprobación de salud
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Configura el punto de entrada para iniciar Streamlit
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
