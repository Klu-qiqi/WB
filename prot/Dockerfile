FROM python:3.10-slim

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=180

WORKDIR /app

# This Dockerfile is intended to be built from the repository root context:
# docker build -f prot/Dockerfile -t auto-dispatch-system .
COPY prot/requirements.txt ./requirements.txt
RUN PIP_REQUIRE_HASHES=0 python -m pip install \
    --prefer-binary \
    --retries 10 \
    --timeout 180 \
    -r requirements.txt

COPY prot/ ./
COPY Dd2WPGKz/ ./Dd2WPGKz/

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
