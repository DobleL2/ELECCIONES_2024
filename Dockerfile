# app/Dockerfile

FROM python:3.10

WORKDIR /ELECCIONES_2024

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/DobleL2/ELECCIONES_2024.git .

RUN pip3 install -r requirements.txt
RUN pip install streamlit

RUN which streamlit
RUN chmod +x $(which streamlit)

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
