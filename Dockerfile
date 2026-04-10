# Базовый образ с Docker Hub (без ghcr.io). Тяжёлые ML-колёса надёжнее на bookworm, не slim.
FROM python:3.13-bookworm

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir uv

RUN adduser --disabled-password --gecos "" agent
USER agent
WORKDIR /home/agent

COPY pyproject.toml uv.lock README.md ./
COPY src src

RUN \
    --mount=type=cache,target=/home/agent/.cache/uv,uid=1000 \
    uv sync --locked

ENTRYPOINT ["uv", "run", "src/server.py"]
CMD ["--host", "0.0.0.0"]
EXPOSE 9009
EXPOSE 9010
