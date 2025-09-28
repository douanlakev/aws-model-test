# Pull Python Image from Docker Hub
FROM python:3.10

# Set the working directory
WORKDIR /opt/program

# Install python3-venv
RUN apt-get update && \
    apt-get install -y python3-venv && \
    rm -rf /var/lib/apt/lists/*

# Create a virtual environment
RUN python3 -m venv /opt/venv

# Ensure the virtual environment is used
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Copy the requirements file and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Set some environment variables
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE

# Copy the inference script and other necessary files
COPY src/inference.py ./

# Set the entry point for the container
ENTRYPOINT ["python", "inference.py"]
