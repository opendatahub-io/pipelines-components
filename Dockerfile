FROM registry.redhat.io/ubi9/python-311@sha256:d7620b96616955d78425518143affdc9463fb1e71d00aa2b7dc2785c54621a0b

WORKDIR /app

USER root
RUN curl -LsSf https://astral.sh/uv/0.6.0/install.sh -o /tmp/uv-install.sh && \
    echo "5464b06094b6363861b99cd60a010892e52fc2a7503b7594419828e74afecfe6  /tmp/uv-install.sh" | sha256sum -c - && \
    sh /tmp/uv-install.sh && rm /tmp/uv-install.sh

COPY pyproject.toml __init__.py ./
COPY components/ components/
COPY pipelines/ pipelines/
COPY scripts/ scripts/
COPY utils/ utils/

RUN chown -R 1001:1001 /app
USER 1001

RUN uv sync --no-cache --extra test

RUN uv run python -m scripts.generate_managed_pipelines.generate_managed_pipelines

CMD [".venv/bin/python", "-m", "scripts.init_managed_pipelines.init_managed_pipelines"]