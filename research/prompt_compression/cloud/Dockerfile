FROM python:3.12.11-slim-bookworm
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/models/huggingface
WORKDIR /opt/promptcomp
COPY cloud/requirements.lock /tmp/requirements.lock
RUN python -m pip install --require-hashes -r /tmp/requirements.lock
COPY pyproject.toml ./
COPY README.md ATTENTION.md VALIDATION.md ./
COPY promptcomp ./promptcomp
COPY configs ./configs
COPY cloud ./cloud
COPY scripts ./scripts
COPY tests ./tests
RUN python -m pip install --no-deps --no-build-isolation .
ENTRYPOINT ["promptcomp"]

