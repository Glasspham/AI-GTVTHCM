# Dockerfile cho Python API
FROM python:3.10-slim

# Cài đặt công cụ hệ thống và driver MSSQL
RUN apt-get update && apt-get install -y \
    curl gnupg2 apt-transport-https unixodbc-dev g++ \
    && curl https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > /usr/share/keyrings/microsoft-prod.gpg \
    && curl https://packages.microsoft.com/config/debian/12/prod.list > /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y msodbcsql17 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements và cài đặt thư viện
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ code vào container
COPY . .

# 🚀 TỰ ĐỘNG THAY ĐỔI CẤU HÌNH KHI BUILD
# 1. db_connection.py: Driver MSSQL cho Linux
RUN sed -i 's/DRIVER={.*}/DRIVER={{ODBC Driver 17 for SQL Server}}/g' db_connection.py && \
    sed -i 's/server = .*/server = "sqlserver"/g' db_connection.py && \
    sed -i 's/password = .*/password = "Password123!"/g' db_connection.py && \
    sed -i 's/Trusted_Connection=yes;/TrustServerCertificate=yes;/g' db_connection.py

# 2. load_data.py: Cho phép chạy khi DB chưa có data hoặc thiếu cột Score
RUN sed -i '/ui.Score,/d' load_data.py && \
    sed -i 's/if "Score" in df.columns:/if True: # Force Score/g' load_data.py && \
    sed -i 's/df\["Score"\] = df\["Score"\].astype(float)/df["Score"] = 1.0/g' load_data.py && \
    sed -i 's/return None/return df/g' load_data.py

# 3. main.py: Giữ API không bị sập nếu DB trống lúc khởi động
RUN sed -i 's/^manager.reload_if_needed()/try: manager.reload_if_needed()\nexcept Exception as e: print(f"⚠️ Startup: {e}")/g' main.py

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
