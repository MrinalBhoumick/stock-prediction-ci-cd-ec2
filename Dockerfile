# Use an official Python runtime as a parent image
FROM python:3.10-slim AS base

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y wget && \
    wget https://dl.k6.io/v0.41.0/linux/amd64/k6-v0.41.0-linux-amd64.tar.gz && \
    tar -xzf k6-v0.41.0-linux-amd64.tar.gz && \
    mv k6-v0.41.0-linux-amd64/k6 /usr/local/bin/ && \
    rm -rf k6-v0.41.0-linux-amd64*

# Install virtualenv globally
RUN pip install virtualenv

# Create and activate a virtual environment
RUN virtualenv /venv
ENV PATH="/venv/bin:$PATH"

# Copy the current directory contents into the container at /app
COPY . /app

# Install necessary dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose the port number on which Streamlit runs (default is 8501)
EXPOSE 8501 8000

# Set environment variables for Streamlit
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Run Streamlit using the specified command
CMD ["streamlit", "run", "app.py"]
