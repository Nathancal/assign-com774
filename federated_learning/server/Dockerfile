FROM python:3.9.0

WORKDIR /federated_learning/server

COPY . /federated_learning/server

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Define environment variable
ENV NAME World

# Run server.py when the container launches
CMD ["python", "./server.py"]