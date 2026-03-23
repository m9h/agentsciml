ARG JAX_TAG=26.02-py3
FROM nvcr.io/nvidia/jax:${JAX_TAG}

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /workspace

# Clone and install agentsciml
RUN git clone https://github.com/m9h/agentsciml.git /workspace/agentsciml && \
    cd /workspace/agentsciml && uv sync --dev --python 3.12

# Clone and install dmipy
RUN git clone https://github.com/m9h/dmipy.git /workspace/dmipy && \
    cd /workspace/dmipy && uv sync --python 3.12

# Default: run the evolutionary loop
WORKDIR /workspace/agentsciml
ENTRYPOINT ["uv", "run", "agentsciml", "run"]
CMD ["--project", "/workspace/dmipy", "--adapter", "dmipy", "--budget", "5.0", "--generations", "20"]
