FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements-ml.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-ml.txt -r requirements-dev.txt

COPY . /app

EXPOSE 5003
CMD ["python", "scripts/run_function_tool.py", "--tool", "videos_wan", "--host", "0.0.0.0", "--port", "5003"]
