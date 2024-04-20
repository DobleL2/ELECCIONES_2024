# Use python:3.10 as the base image
FROM python:3.10

# Set the working directory to /app
WORKDIR /app

# Copy the application files
COPY ./app /app

# Install system dependencies for pyodbc and SQL Server ODBC Driver
RUN apt-get update && apt-get install -y gnupg2 curl unixodbc-dev

# Import the GPG keys of the Microsoft SQL Server repository
RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -

# Register the Microsoft SQL Server Ubuntu repository
RUN curl https://packages.microsoft.com/config/ubuntu/$(. /etc/os-release; echo $VERSION_ID)/prod.list > /etc/apt/sources.list.d/mssql-release.list

# Install SQL Server ODBC Driver
RUN apt-get update && ACCEPT_EULA=Y apt-get install -y msodbcsql17

# Set up ODBC configuration
RUN echo "[ODBC Driver 17 for SQL Server]" >> /etc/odbcinst.ini
RUN echo "Driver = /opt/microsoft/msodbcsql17/lib64/libmsodbcsql-17.8.so.1.1" >> /etc/odbcinst.ini

RUN apt-get update && apt-get install -y gnupg2 curl unixodbc-dev

# Create a virtual environment and install dependencies
RUN python -m venv venv
ENV PATH="/app/venv/bin:$PATH"

# Upgrade pip and install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install streamlit

# Expose the port 8501
EXPOSE 8501

# Set the start command
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]