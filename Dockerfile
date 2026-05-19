FROM python:3.10-slim

WORKDIR /app

# Install dependencies first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Default command keeps the container alive so we can run make commands inside it
CMD ["tail", "-f", "/dev/null"]