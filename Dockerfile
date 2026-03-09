# Pull base python image
FROM python:3.11-slim

# Install system dependencies needed for Ollama install script (skip recs and rm apt lists to reduce image size)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the project code into the container
WORKDIR /label_verification
COPY . .

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install the nltk stopwords data
RUN python -m nltk.downloader stopwords

# Run the label verification pipeline
CMD ["python3", "-m", "label_verification"] 

