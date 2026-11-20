"""
Vectorisation des dépôts GitHub avec multilingual-e5-base (768 dim). Connexions PostgreSQL via psycopg2.

Trois modes :
  python embed_repos.py encode   -> encode TOUS les dépôts et écrit des chunks .npz dans ./embeddings_out/

IMPORTANT : le modèle, MAX_SEQ_LENGTH, README_MAX_CHARS, HALF et le nettoyage doivent rester
identiques pour tous les vecteurs de la table, sinon les anciens et les nouveaux ne sont plus comparables.
"""
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values


# Configuration
def _int(name, default):
    return int(os.getenv(name, str(default)))


PG_HOST = os.getenv("POSTGRES_HOST", "localhost")
PG_PORT = _int("POSTGRES_PORT", 5432)
PG_DB = os.getenv("POSTGRES_DB", "github_data")
EMBEDER_DB_USER = os.getenv("EMBEDER_DB_USER", "embeder_user")
EMBEDER_DB_PASSWORD = os.getenv("EMBEDER_DB_PASSWORD", "embeder_password_change_me")

MODEL_NAME = os.getenv("MODEL_NAME", "intfloat/multilingual-e5-base")
BATCH_SIZE = _int("BATCH_SIZE", 64)
CHUNK_ROWS = _int("CHUNK_ROWS", 2000)
MAX_SEQ_LENGTH = _int("MAX_SEQ_LENGTH", 512)
README_MAX_CHARS = _int("README_MAX_CHARS", 2000)

# on ne ramène de la base que le début du README
README_FETCH_CHARS = _int("README_FETCH_CHARS", 8000)
DEVICE = os.getenv("DEVICE") or None
HALF = os.getenv("HALF", "0") == "1"
LIMIT = _int("LIMIT", 0)
FULL_CHECK = os.getenv("FULL_CHECK", "0") == "1"
SYNC_INTERVAL = _int("SYNC_INTERVAL", 0)
SHOW_PROGRESS = os.getenv("SHOW_PROGRESS", "1") == "1"
OUT_DIR = Path(os.getenv("OUT_DIR", "embeddings_out"))


# Connexions 
def _connect():
    return psycopg2.connect(host=PG_HOST, port=PG_PORT, dbname=PG_DB,
                            user=EMBEDER_DB_USER, password=EMBEDER_DB_PASSWORD, connect_timeout=10)


# Dernier README de chaque dépôt (LEFT JOIN : on garde les dépôts sans README).
_SELECT = f"""
SELECT r.id, r.name, r.description, r.topics, r.languages,
       LEFT(m.content, {README_FETCH_CHARS}) AS readme
FROM github.repositories r
LEFT JOIN LATERAL (
    SELECT content FROM github.readmes
    WHERE repository_id = r.id
    ORDER BY collected_at DESC
    LIMIT 1
) m ON TRUE
"""
QUERY_ALL = _SELECT + "ORDER BY r.id" + (f" LIMIT {LIMIT}" if LIMIT else "")

# Passe légère : uniquement des dates, pas de contenu de README.
QUERY_CHANGES = """
SELECT r.id, GREATEST(r.updated_at, m.collected_at) AS changed_at
FROM github.repositories r
LEFT JOIN LATERAL (
    SELECT collected_at FROM github.readmes
    WHERE repository_id = r.id
    ORDER BY collected_at DESC
    LIMIT 1
) m ON TRUE
"""

# Nettoyage et construction du texte (INCHANGÉ par rapport aux tests validés)
def clean_readme(text):
    if not text:
        return ""

    """
    Nettoyage :
    - commentaires HTML
    - blocs de code
    - images / badges markdown
    - badges cliquables
    - [texte](url) -> texte
    - balises HTML
    - URLs nues
    - marqueurs markdown en début de ligne
    """

    text = re.sub(r"<!--.*?-->", " ", text, flags=re.S)           
    text = re.sub(r"```.*?```", " ", text, flags=re.S)            
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)             
    text = re.sub(r"\[!\[[^\]]*\]\([^)]*\)\]\([^)]*\)", " ", text)  
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)          
    text = re.sub(r"<[^>]+>", " ", text)                         
    text = re.sub(r"https?://\S+", " ", text)                     
    text = re.sub(r"^[#>\-\*\|=_~`\s]+", "", text, flags=re.M)    
    text = re.sub(r"[|=_~`*]{2,}", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:README_MAX_CHARS]


def languages_to_text(languages, top=3):
    """languages est un JSONB : dict {nom: octets} ou liste de {name, size}."""
    if not languages:
        return ""
    if isinstance(languages, str):
        try:
            languages = json.loads(languages)
        except ValueError:
            return ""
    names = []
    if isinstance(languages, dict):
        names = [k for k, _ in sorted(languages.items(), key=lambda kv: -(kv[1] or 0))]
    elif isinstance(languages, list):
        items = []
        for it in languages:
            if isinstance(it, dict):
                items.append((it.get("name") or it.get("language"), it.get("size") or it.get("bytes") or 0))
            elif isinstance(it, str):
                items.append((it, 0))
        names = [n for n, _ in sorted(items, key=lambda x: -x[1]) if n]
    return ", ".join(names[:top])


def build_text(name, description, topics, languages, readme):
    parts = []
    if name:
        parts.append(name.strip())
    if description and description.strip():
        parts.append(description.strip())
    if topics:
        parts.append("Topics: " + ", ".join(topics))
    lang = languages_to_text(languages)
    if lang:
        parts.append("Langages: " + lang)
    body = clean_readme(readme)
    if body:
        parts.append(body)
    text = ". ".join(p.rstrip(".") for p in parts[:-1]) if body else ". ".join(p.rstrip(".") for p in parts)
    if body:
        text = (text + ". " if text else "") + body
    return "passage: " + text


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# Modèle
_MODEL = None


def get_model():
    """Chargé à la demande (et une seule fois) : une passe 'sync' sans rien à faire ne charge pas le modèle."""
    global _MODEL
    if _MODEL is None:
        import torch
        from sentence_transformers import SentenceTransformer

        device = DEVICE or ("cuda" if torch.cuda.is_available() else "cpu")
        model = SentenceTransformer(MODEL_NAME, device=device)
        model.max_seq_length = MAX_SEQ_LENGTH
        use_half = HALF and device.startswith("cuda")
        if use_half:
            model.half()
        print(f"Modèle chargé | périphérique={device} | batch={BATCH_SIZE} | fp16={use_half} "
              f"| max_seq_length={MAX_SEQ_LENGTH} | README_MAX_CHARS={README_MAX_CHARS}", flush=True)
        if device == "cpu":
            print("ATTENTION : exécution sur CPU, l'encodage sera lent.", flush=True)
        _MODEL = model
    return _MODEL


def embed(texts):
    emb = get_model().encode(texts, batch_size=BATCH_SIZE, normalize_embeddings=True,
                             show_progress_bar=SHOW_PROGRESS).astype(np.float32)
    # Renormalisation en float32 (utile si le modèle tourne en fp16).
    emb /= np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
    return emb

# Mode 2 : encode les dépots en chunk dans des .npz
def done_ids():
    ids = set()
    for f in OUT_DIR.glob("chunk_*.npz"):
        ids.update(np.load(f, allow_pickle=True)["ids"].tolist())
    return ids


def encode():
    OUT_DIR.mkdir(exist_ok=True)
    already = done_ids()
    print(f"{len(already)} dépôts déjà encodés, reprise.")
    chunk_no = len(list(OUT_DIR.glob("chunk_*.npz")))
    buf_ids, buf_texts = [], []

    def flush():
        nonlocal chunk_no, buf_ids, buf_texts
        if not buf_ids:
            return
        emb = embed(buf_texts)
        hashes = [text_hash(t) for t in buf_texts]
        np.savez_compressed(OUT_DIR / f"chunk_{chunk_no:05d}.npz",
                            ids=np.array(buf_ids, dtype=object),
                            emb=emb, hashes=np.array(hashes, dtype=object))
        chunk_no += 1
        print(f"chunk {chunk_no} sauvegardé ({len(buf_ids)} dépôts)", flush=True)
        buf_ids, buf_texts = [], []

    conn = _connect()
    try:
        # Curseur nommé = curseur côté serveur : les 145 000 lignes ne sont pas chargées d'un coup.
        with conn.cursor(name="stream_repos") as cur:
            cur.itersize = 1000
            cur.execute(QUERY_ALL)
            for rid, name, desc, topics, langs, readme in cur:
                if rid in already:
                    continue
                buf_ids.append(rid)
                buf_texts.append(build_text(name, desc, topics, langs, readme))
                if len(buf_ids) >= CHUNK_ROWS:
                    flush()
        flush()
    finally:
        conn.close()
    print("Encodage terminé.")

# ----------------------------------------------------------------------
def main():
    mode = sys.argv[1] if len(sys.argv) == 2 else None
    if mode == "encode":
        encode()
    else:
        sys.exit("Usage : python embed_repos.py [encode]")


if __name__ == "__main__":
    main()
