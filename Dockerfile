# ----------------------------
# 1️⃣ Base Image
# ----------------------------
FROM python:3.11-slim

# ----------------------------
# 2️⃣ Set Working Directory
# ----------------------------
WORKDIR /app

# ----------------------------
# 3️⃣ Install Build Tools (for CatBoost/XGBoost)
# ----------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------
# 4️⃣ Copy Requirements & Install Python Dependencies
# ----------------------------
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ----------------------------
# 5️⃣ Copy Application Files
# ----------------------------
COPY src ./src
COPY templates ./templates
COPY artifacts ./artifacts
COPY application.py .

# ----------------------------
# 6️⃣ Expose Port
# ----------------------------
EXPOSE 5000

# ----------------------------
# 7️⃣ Run with Gunicorn
# ----------------------------
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "application:app"]
