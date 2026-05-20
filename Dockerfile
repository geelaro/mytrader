# Requires Docker Desktop (Windows/Mac) — uses host.docker.internal for FutuOpenD.
# On Linux, replace host.docker.internal with the host IP or use --add-host.
FROM python:3.10-slim

WORKDIR /app

# System deps for matplotlib & openpyxl
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY Pipfile Pipfile.lock ./
RUN pip install --no-cache-dir pipenv && \
    pipenv install --system --deploy --ignore-pipfile

# App
COPY . .

# Dashboard port
EXPOSE 8501

CMD ["python", "live_trader.py", "--broker", "futu", "--futu-host", "host.docker.internal", "--futu-port", "11111", "--daemon", "--interval", "5", "--notify", "--all-day"]
