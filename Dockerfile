FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Expose Streamlit port
EXPOSE 10000

# Health check
HEALTHCHECK CMD curl --fail http://localhost:10000/_stcore/health || exit 1

# Run Streamlit on Render's expected port
CMD ["streamlit", "run", "app.py", \
     "--server.port=10000", \
     "--server.address=0.0.0.0", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false", \
     "--server.headless=true"]
