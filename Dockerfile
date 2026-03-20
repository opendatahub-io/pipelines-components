FROM registry.redhat.io/ubi9/python-311@sha256:d7620b96616955d78425518143affdc9463fb1e71d00aa2b7dc2785c54621a0b

WORKDIR /app

USER root
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh

COPY pyproject.toml __init__.py ./
COPY components/ components/
COPY pipelines/ pipelines/
COPY scripts/ scripts/

RUN chown -R 1001:1001 /app
USER 1001

RUN uv sync --no-cache --extra test

RUN uv run python -m scripts.generate_managed_pipelines.generate_managed_pipelines

CMD ["python"]