# Use an official Python runtime as a parent image
FROM arm64v8/python:3.11-slim-buster

# Set the working directory in the container to /app
WORKDIR /app

# Add the current directory contents into the container at /app
ADD . /app

# Set environment variables
ENV PYTHONPATH='./'

# Install any needed packages specified in requirements.txt
RUN apt-get update && apt-get install -y gcc python3-dev
RUN pip3 install -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Run train_t5.py when the container launches
CMD ["python3", "train_t5.py"]

