# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install build dependencies and necessary system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    python3-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Copy the pyproject.toml and poetry.lock first, to leverage Docker's caching
COPY pyproject.toml poetry.lock* /app/

# Install dependencies
RUN poetry install --no-dev

# Copy the rest of the application code
COPY . /app

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=/app/app.py
ENV FLASK_ENV=production

# Install gunicorn
RUN poetry run pip install gunicorn

# Run gunicorn when the container launches
CMD ["poetry", "run", "gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
