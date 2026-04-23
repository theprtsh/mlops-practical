FROM python:3.12-slim

WORKDIR /app

# Install dependencies
# Using pip to keep it simple within the container for the lab
#
# Copy everything first
COPY . .
RUN pip install .

# Copy application files
COPY . .

# Ensure data and models are generated
# In a real pipeline this might be pre-baked or fetched, 
# but for simplicity in a lab demo, we ensure directories exist.
RUN python data.py && python train.py

# Expose port
EXPOSE 8000

# Run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
