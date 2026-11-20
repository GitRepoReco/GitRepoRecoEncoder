# Service de vectorisation (multilingual-e5-base) : encode les dépôts nouveaux/modifiés
# et les insère directement dans la table d'embeddings. Connexions PostgreSQL via psycopg2.
#
# Build GPU (par défaut, torch de PyPI embarque CUDA) :
#   docker build -t repo-embedder .
# Build CPU uniquement (image plus légère) :
#   docker build -t repo-embedder --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu .
#
# Run GPU : nécessite NVIDIA Container Toolkit sur l'hôte et `--gpus all`.
FROM python:3.12-slim

ARG TORCH_INDEX_URL=""

ARG APP_VERSION=dev

ENV APP_VERSION=${APP_VERSION}

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf-cache

WORKDIR /app

COPY requirements.txt

# PyTorch d'abord, puis le reste (sentence-transformers ne le réinstalle pas).
RUN if [ -n "$TORCH_INDEX_URL" ]; then \
        pip install --no-cache-dir torch --index-url "$TORCH_INDEX_URL"; \
    else \
        pip install --no-cache-dir torch; \
    fi \
    && pip install --no-cache-dir -r requirements.txt

# Le modèle est téléchargé à la construction : pas de téléchargement (ni de limite Hugging Face) au démarrage.
ARG MODEL_NAME=intfloat/multilingual-e5-base
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${MODEL_NAME}')"

COPY app ./app 

# --- Variables d'environnement NON secrètes (surchargeables au `docker run` / dans le compose) ---
ENV POSTGRES_HOST=postgres \
    POSTGRES_PORT=5432 \
    POSTGRES_DB=postgres  \
    EMBEDER_DB_USER=embeder_user \
    MODEL_NAME=${MODEL_NAME} \
    BATCH_SIZE=64 \
    CHUNK_ROWS=2000 \
    MAX_SEQ_LENGTH=512 \
    README_MAX_CHARS=2000 \
    HALF=0 \
    SYNC_INTERVAL=604800\
    LIMIT=0\

# Les mots de passe (POSTGRES_PASSWORD, EMBEDER_DB_PASSWORD) ne sont PAS dans l'image :
# ils seraient lisibles par quiconque a l'image (docker history). Ils sont passés à l'exécution.

CMD ["python", "embed_repos.py", "sync"]
