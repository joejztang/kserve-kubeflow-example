FROM python:3.11-slim

#TODO: continue working on this
COPY . .
RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python", "transformer.py"]