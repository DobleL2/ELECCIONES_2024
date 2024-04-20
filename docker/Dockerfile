# Use official Python image as base
FROM python:3.10

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install ODBC library and its dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends unixodbc unixodbc-dev && \
    rm -rf /var/lib/apt/lists/*
    
# Set the working directory in the container
WORKDIR /app

# Create and activate a virtual environment
RUN python -m venv venv
ENV PATH="/home/appuser/app/venv/bin:$PATH"

# Copy the dependencies file to the working directory
COPY ./app/requirements.txt /app/

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app

# Expose the port streamlit runs on
EXPOSE 8501

# Command to run the streamlit app
CMD ["streamlit", "run", "app.py"]
