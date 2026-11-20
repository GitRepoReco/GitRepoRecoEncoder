"""
Vectorisation des dépôts GitHub avec multilingual-e5-base (768 dim). Connexions PostgreSQL via psycopg2.

Trois modes :
  python embed_repos.py sync     -> mode Docker : encode les dépôts NOUVEAUX ou MODIFIÉS et les insère
                                    directement dans la table d'embeddings (aucun fichier .npz).
                                    Sert aussi au remplissage initial si le conteneur a un GPU.
  python embed_repos.py encode   -> encode TOUS les dépôts et écrit des chunks .npz dans ./embeddings_out/
  python embed_repos.py load     -> insère ces .npz dans la table d'embeddings

Deux comptes Postgres, sur la même base POSTGRES_DB :
  - POSTGRES_USER / POSTGRES_PASSWORD         : LECTURE des tables github.repositories et github.readmes
  - EMBEDER_DB_USER / EMBEDER_DB_PASSWORD     : LECTURE/ÉCRITURE de la table d'embeddings (EMBEDDINGS_TABLE)

Variables d'environnement (valeur par défaut entre parenthèses) :
  POSTGRES_HOST (localhost), POSTGRES_PORT (5432), POSTGRES_DB (github_data)
  POSTGRES_USER (postgres), POSTGRES_PASSWORD (postgres)
  EMBEDER_DB_USER (embeder_user), EMBEDER_DB_PASSWORD (embeder_password_change_me)
  EMBEDDINGS_TABLE (github.repo_embeddings)  "schema.table"
  MODEL_NAME (intfloat/multilingual-e5-base)
  BATCH_SIZE (64), CHUNK_ROWS (2000), MAX_SEQ_LENGTH (512), README_MAX_CHARS (2000)
  DEVICE (auto : cuda si disponible sinon cpu), HALF (0 : fp32 ; 1 : fp16, seulement si le GPU y gagne)
  LIMIT (0 = pas de limite ; utile pour un test), FULL_CHECK (0 ; 1 = revérifie tous les dépôts via le hash)
  SYNC_INTERVAL (0 = une seule passe ; sinon nombre de secondes entre deux passes)

IMPORTANT : le modèle, MAX_SEQ_LENGTH, README_MAX_CHARS, HALF et le nettoyage doivent rester
identiques pour tous les vecteurs de la table, sinon les anciens et les nouveaux ne sont plus comparables.

Dépendances : psycopg2-binary, pgvector, sentence-transformers, numpy
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


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
def _int(name, default):
    return int(os.getenv(name, str(default)))


PG_HOST = os.getenv("POSTGRES_HOST", "localhost")
PG_PORT = _int("POSTGRES_PORT", 5432)
PG_DB = os.getenv("POSTGRES_DB", "github_data")
READ_USER = os.getenv("POSTGRES_USER", "postgres")
READ_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
EMBEDER_DB_USER = os.getenv("EMBEDER_DB_USER", "embeder_user")
EMBEDER_DB_PASSWORD = os.getenv("EMBEDER_DB_PASSWORD", "embeder_password_change_me")
EMBEDDINGS_TABLE = os.getenv("EMBEDDINGS_TABLE", "github.repo_embeddings")

MODEL_NAME = os.getenv("MODEL_NAME", "intfloat/multilingual-e5-base")
DIM = 768
BATCH_SIZE = _int("BATCH_SIZE", 64)
CHUNK_ROWS = _int("CHUNK_ROWS", 2000)
MAX_SEQ_LENGTH = _int("MAX_SEQ_LENGTH", 512)
README_MAX_CHARS = _int("README_MAX_CHARS", 2000)
README_FETCH_CHARS = _int("README_FETCH_CHARS", 8000)  # on ne ramène de la base que le début du README
DEVICE = os.getenv("DEVICE") or None
HALF = os.getenv("HALF", "0") == "1"
LIMIT = _int("LIMIT", 0)
FULL_CHECK = os.getenv("FULL_CHECK", "0") == "1"
SYNC_INTERVAL = _int("SYNC_INTERVAL", 0)
SHOW_PROGRESS = os.getenv("SHOW_PROGRESS", "1") == "1"
OUT_DIR = Path(os.getenv("OUT_DIR", "embeddings_out"))


# ----------------------------------------------------------------------
# Connexions (psycopg2) : paramètres séparés, pas de DSN à encoder
# ----------------------------------------------------------------------
def _connect(user, password):
    print("HOST : " + PG_HOST + ", PORT : " + str(PG_PORT) + ", DB : " + PG_DB + ", USER :" + user + ", PWD : " + password)
    return psycopg2.connect(host=PG_HOST, port=PG_PORT, dbname=PG_DB,
                            user=user, password=password, connect_timeout=10)


def read_connection():
    """Compte en lecture sur github.repositories / github.readmes."""
    return _connect(READ_USER, READ_PASSWORD)


def write_connection():
    """Compte propriétaire de la table d'embeddings (lecture + écriture)."""
    return _connect(EMBEDER_DB_USER, EMBEDER_DB_PASSWORD)


def table_ident():
    """EMBEDDINGS_TABLE = 'schema.table' -> identifiant SQL correctement échappé."""
    return sql.Identifier(*EMBEDDINGS_TABLE.split("."))


def upsert_sql():
    # Le "%s" unique après VALUES est rempli par psycopg2.extras.execute_values.
    return sql.SQL(
        "INSERT INTO {} (repository_id, embedding, text_hash, model_name) "
        "VALUES %s "
        "ON CONFLICT (repository_id) DO UPDATE SET "
        "embedding = EXCLUDED.embedding, text_hash = EXCLUDED.text_hash, "
        "model_name = EXCLUDED.model_name, embedded_at = now()"
    ).format(table_ident())


def upsert_rows(cur, rows):
    """rows : liste de (repository_id, embedding numpy, text_hash, model_name)."""
    execute_values(cur, upsert_sql(), rows, page_size=500)


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
QUERY_BY_IDS = _SELECT + "WHERE r.id = ANY(%s)"

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


# ----------------------------------------------------------------------
# Nettoyage et construction du texte (INCHANGÉ par rapport aux tests validés)
# ----------------------------------------------------------------------
def clean_readme(text):
    if not text:
        return ""
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.S)           # commentaires HTML
    text = re.sub(r"```.*?```", " ", text, flags=re.S)            # blocs de code
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)             # images / badges markdown
    text = re.sub(r"\[!\[[^\]]*\]\([^)]*\)\]\([^)]*\)", " ", text)  # badges cliquables
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)          # [texte](url) -> texte
    text = re.sub(r"<[^>]+>", " ", text)                          # balises HTML
    text = re.sub(r"https?://\S+", " ", text)                     # URLs nues
    text = re.sub(r"^[#>\-\*\|=_~`\s]+", "", text, flags=re.M)    # marqueurs markdown en début de ligne
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


# ----------------------------------------------------------------------
# Modèle
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# Mode 1 : sync (direct en base, sans .npz)
# ----------------------------------------------------------------------
def sync_once():
    from pgvector.psycopg2 import register_vector

    t0 = time.time()
    wconn = write_connection()
    rconn = read_connection()
    rconn.autocommit = True  # lectures seules : pas de transaction longue ouverte
    try:
        register_vector(wconn)

        # Ce qui est déjà vectorisé : {id: (hash, modèle, date)}
        with wconn.cursor() as cur:
            cur.execute(sql.SQL("SELECT repository_id, text_hash, model_name, embedded_at FROM {}")
                        .format(table_ident()))
            existing = {rid: (h, m, at) for rid, h, m, at in cur.fetchall()}
        wconn.commit()

        # Passe légère : quels dépôts sont nouveaux ou ont changé depuis leur vectorisation ?
        with rconn.cursor() as cur:
            cur.execute(QUERY_CHANGES)
            changes = cur.fetchall()
        todo = []
        for rid, changed_at in changes:
            old = existing.get(rid)
            if (old is None
                    or FULL_CHECK
                    or old[1] != MODEL_NAME
                    or (changed_at is not None and old[2] is not None and changed_at > old[2])):
                todo.append(rid)
        if LIMIT:
            todo = todo[:LIMIT]
        print(f"{len(changes)} dépôts en base | {len(existing)} déjà vectorisés | {len(todo)} à traiter.", flush=True)

        inserted = unchanged = 0
        for i in range(0, len(todo), CHUNK_ROWS):
            batch = todo[i:i + CHUNK_ROWS]
            with rconn.cursor() as cur:
                cur.execute(QUERY_BY_IDS, (batch,))
                rows = cur.fetchall()

            ids, texts, hashes, same = [], [], [], []
            for rid, name, desc, topics, langs, readme in rows:
                text = build_text(name, desc, topics, langs, readme)
                h = text_hash(text)
                old = existing.get(rid)
                if old and old[0] == h and old[1] == MODEL_NAME:
                    # texte identique : inutile de ré-encoder
                    same.append(rid)
                    continue
                ids.append(rid)
                texts.append(text)
                hashes.append(h)

            with wconn.cursor() as cur:
                if same:
                    # On "touche" embedded_at pour ne pas les retraiter à chaque passe.
                    cur.execute(
                        sql.SQL("UPDATE {} SET embedded_at = now() WHERE repository_id = ANY(%s)")
                        .format(table_ident()),
                        (same,),
                    )
                if ids:
                    emb = embed(texts)
                    upsert_rows(cur, [(rid, e, h, MODEL_NAME) for rid, e, h in zip(ids, emb, hashes)])
            # un commit par lot : une interruption ne perd presque rien
            wconn.commit()
            inserted += len(ids)
            unchanged += len(same)
            print(f"  lot {i // CHUNK_ROWS + 1} : {len(ids)} vectorisés, {len(same)} inchangés "
                  f"(total {inserted + unchanged}/{len(todo)})", flush=True)
    except Exception:
        wconn.rollback()
        raise
    finally:
        rconn.close()
        wconn.close()

    print(f"Passe terminée en {time.time() - t0:.0f} s : {inserted} insérés/mis à jour, {unchanged} inchangés.", flush=True)


# ----------------------------------------------------------------------
# Mode 2 : encode (tous les dépôts -> fichiers .npz)
# ----------------------------------------------------------------------
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

    conn = read_connection()
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
# Mode 3 : load (.npz -> table d'embeddings)
# ----------------------------------------------------------------------
def load():
    from pgvector.psycopg2 import register_vector

    files = sorted(OUT_DIR.glob("chunk_*.npz"))
    if not files:
        sys.exit(f"Aucun chunk_*.npz dans {OUT_DIR} : lancez d'abord 'encode'.")

    inserted = skipped = 0
    wconn = write_connection()
    rconn = read_connection()
    rconn.autocommit = True
    try:
        register_vector(wconn)
        for f in files:
            d = np.load(f, allow_pickle=True)
            ids, emb, hashes = d["ids"].tolist(), d["emb"], d["hashes"].tolist()
            if emb.ndim != 2 or emb.shape[1] != DIM:
                sys.exit(f"{f.name} : forme {emb.shape}, attendu (_, {DIM}). Arrêt.")
            if not np.isfinite(emb).all():
                sys.exit(f"{f.name} : contient des NaN/inf. Arrêt.")

            # On n'insère que les dépôts qui existent encore (clé étrangère).
            with rconn.cursor() as cur:
                cur.execute("SELECT id FROM github.repositories WHERE id = ANY(%s)", (ids,))
                known = {r[0] for r in cur.fetchall()}
            rows = [(i, e, h, MODEL_NAME) for i, e, h in zip(ids, emb, hashes) if i in known]
            skipped += len(ids) - len(rows)

            with wconn.cursor() as cur:
                upsert_rows(cur, rows)
            wconn.commit()
            inserted += len(rows)
            print(f"{f.name} : {len(rows)} insérés (total {inserted})", flush=True)
    except Exception:
        wconn.rollback()
        raise
    finally:
        rconn.close()
        wconn.close()
    print(f"Chargement terminé. Insérés/mis à jour : {inserted} | ignorés (dépôt absent) : {skipped}")
    print("Pensez à créer l'index HNSW APRÈS ce chargement (voir init_embeddings.sql).")


# ----------------------------------------------------------------------
def main():
    mode = sys.argv[1] if len(sys.argv) == 2 else None
    if mode == "encode":
        encode()
    elif mode == "load":
        load()
    elif mode == "sync":
        if SYNC_INTERVAL <= 0:
            sync_once()
        else:
            # mode service : une passe toutes les SYNC_INTERVAL secondes
            while True:
                try:
                    sync_once()
                # on ne tue pas le service pour une erreur ponctuelle
                except Exception as exc:  
                    print(f"Erreur pendant la passe : {exc}", flush=True)
                time.sleep(SYNC_INTERVAL)
    else:
        sys.exit("Usage : python embed_repos.py [sync|encode|load]")


if __name__ == "__main__":
    main()
