# HF Spaces (Docker SDK): CPU-only, models baked in, non-root, app_port 7860.
# Ingest does NOT run here — it's an offline step the developer runs from
# their own machine against Qdrant Cloud (ARCHITECTURE §7/§8). This image
# only serves the API, so it never needs data/raw or data/processed.
FROM python:3.11-slim

# HF Spaces convention: run as a non-root user with a writable $HOME (the
# rest of the image filesystem is ephemeral/read-only in practice).
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# CPU-only torch, pinned to the version verified locally (2.12.1). Installed
# BEFORE requirements.txt so pip sees it already satisfied and never resolves
# the default CUDA wheel (dead weight on a CPU-only free-tier Space). If this
# exact patch version isn't published on the CPU index, bump to the nearest
# available patch and re-verify locally first.
RUN pip install --no-cache-dir --user torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user src/ src/

# Bake bge-m3 into the image so the first request doesn't pay a cold model
# download on the ephemeral filesystem. The reranker (bge-reranker-v2-m3) is
# deliberately NOT baked: RERANK_ENABLED defaults off (M4 finding) and is
# only ever pulled lazily if a deployment turns the flag on.
RUN python -c "from src.config import EMBEDDING_MODEL; from FlagEmbedding import BGEM3FlagModel; BGEM3FlagModel(EMBEDDING_MODEL, use_fp16=False)"

EXPOSE 7860
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "7860"]
