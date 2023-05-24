FROM python:3.8
RUN apt-get update && apt-get install -y \
    cmake \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
COPY . ./
EXPOSE 8080
CMD ["streamlit", "run", "main.py", "--server.port=8080", "--server.address=0.0.0.0"]