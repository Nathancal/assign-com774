FROM python:3.9.0

WORKDIR /federated_learning/server

COPY . /federated_learning/server

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Define environment variable
ENV NAME World
ENV AZURE_CLIENT_ID=1bee10b2-17dd-4a50-b8aa-488d27bdd5a1
ENV AZURE_CLIENT_SECRET=MZK8Q~M5oNATdagyRKMUs-V-2dNggq3aAlRRdb8W
ENV AZURE_TENANT_ID=6f0b9487-4fa8-42a8-aeb4-bf2e2c22d4e8


# Run server.py when the container launches
CMD ["python", "./server.py"]