FROM python:3.9.0

WORKDIR /federated_learning/server

COPY . /federated_learning/server

RUN /usr/local/bin/python -m pip install --upgrade pip setuptools

RUN /usr/local/bin/python -m pip install pyyaml

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

RUN /usr/local/bin/python -m pip install --no-build-isolation ruamel.yaml

# Make port 8080 available to the world outside this container
EXPOSE 6667
EXPOSE 6660

# Define environment variable
ENV NAME World
ENV TOTAL_SUBJECTS=4

# Run server.py when the container launches
CMD ["python", "run_client.py"]
