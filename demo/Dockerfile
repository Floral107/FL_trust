# RobShap demo — self-contained Streamlit dashboard.
# Build:  docker build -t robshap-demo .
# Run:    docker run -p 8501:8501 robshap-demo
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY robshap_demo/ robshap_demo/
COPY app.py make_sample_data.py ./
COPY data/ data/
RUN python make_sample_data.py

EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
