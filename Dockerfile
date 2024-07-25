# syntax=docker/dockerfile:1.2
FROM python:3.8.5

# Set the working directory to /app
WORKDIR /app

# Copy all files
COPY . .

# update pip
RUN pip install --upgrade pip

# Install requierements
RUN pip install -r requirements.txt

# Expose port 8080, which is the port by default
#EXPOSE 8080

# Command to run the application when the container starts
CMD ["uvicorn", "challenge.api:app", "--host", "0.0.0.0", "--port", "8080"]
