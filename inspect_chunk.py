"""
Inspecte un chunk d'embeddings et vérifie qu'il est cohérent.

Usage :
  python inspect_chunk.py                      # prend le premier chunk de embeddings_out/
  python inspect_chunk.py embeddings_out/chunk_00000.npz

Utilise la connexion en lecture de embed_repos.py (variables POSTGRES_*), via psycopg2.
"""
import sys
from pathlib import Path

import numpy as np

import embed_repos_dock as er 

path = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted(er.OUT_DIR.glob("chunk_*.npz"))[0]
d = np.load(path, allow_pickle=True)
ids, emb, hashes = d["ids"].tolist(), d["emb"], d["hashes"].tolist()

# 1. Contrôles techniques
norms = np.linalg.norm(emb, axis=1)
print(f"Fichier        : {path}")
print(f"Nb de dépôts   : {len(ids)}")
print(f"Forme embeddings: {emb.shape}  (attendu : ({len(ids)}, {er.DIM}))")
print(f"Type           : {emb.dtype}")
print(f"NaN / inf      : {int((~np.isfinite(emb)).sum())}  (attendu : 0)")
print(f"Norme L2       : min={norms.min():.4f} max={norms.max():.4f}  (attendu : ~1.0)")
print(f"Textes uniques : {len(set(hashes))}/{len(hashes)}  (des doublons = dépôts au texte identique)")
print(f"Vecteurs nuls  : {int((norms < 1e-6).sum())}")

# 2. Reconstruction de ce qui a été encodé 
rng = np.random.default_rng(0)
sample = rng.choice(len(ids), size=min(3, len(ids)), replace=False)

conn = er.read_connection()
try:
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT r.id, r.full_name, r.name, r.description, r.topics, r.languages,
                   LEFT(m.content, {er.README_FETCH_CHARS}) AS readme
            FROM github.repositories r
            LEFT JOIN LATERAL (
                SELECT content FROM github.readmes
                WHERE repository_id = r.id ORDER BY collected_at DESC LIMIT 1
            ) m ON TRUE
            WHERE r.id = ANY(%s)
            """,
            (ids,),
        )
        rows = cur.fetchall()
finally:
    conn.close()
info = {r[0]: r for r in rows}

print("\n" + "=" * 70)
print("TEXTES ENCODÉS (3 dépôts au hasard)")
print("=" * 70)
for i in sample:
    r = info.get(ids[i])
    if not r:
        continue
    text = er.build_text(r[2], r[3], r[4], r[5], r[6])
    print(f"\n[{r[1]}]  ({len(text)} caractères)")
    print(text[:600] + (" [...]" if len(text) > 600 else ""))
    print(f"vecteur : {emb[i][:5].round(3)} ...")

# Vérification de la similarité entre dépôts pour vérifier la cohérence des données
print("\n" + "=" * 70)
print("VOISINS LES PLUS PROCHES (similarité cosinus, dans ce chunk uniquement)")
print("=" * 70)
sims = emb @ emb.T
for i in sample:
    order = np.argsort(-sims[i])[1:6]
    r = info.get(ids[i])
    print(f"\n{r[1] if r else ids[i]}")
    for j in order:
        rj = info.get(ids[j])
        label = rj[1] if rj else ids[j]
        desc = (rj[3] or "")[:70] if rj else ""
        print(f"   {sims[i, j]:.3f}  {label}  {desc}")
