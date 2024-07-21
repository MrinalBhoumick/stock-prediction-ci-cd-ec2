# Use an official Python runtime as a parent image
FROM python:3.10-slim AS base

# Set the working directory in the container
WORKDIR /app

# Install virtualenv globally
RUN pip install virtualenv

# Create and activate a virtual environment
RUN virtualenv /venv
ENV PATH="/venv/bin:$PATH"

# Install k6
RUN apt-get update && \
    apt-get install -y gnupg curl && \
    curl -s https://dl.k6.io/key.asc | apt-key add - && \
    echo "deb https://dl.k6.io/deb/ stable main" | tee /etc/apt/sources.list.d/k6.list && \
    apt-get update && \
    apt-get install -y k6

# Copy the current directory contents into the container at /app
COPY . /app

# Install necessary dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose the port number on which Streamlit runs (default is 8501)
EXPOSE 8501

# Set environment variables for Streamlit
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Run Streamlit using the specified command
CMD ["streamlit", "run", "app.py"]
