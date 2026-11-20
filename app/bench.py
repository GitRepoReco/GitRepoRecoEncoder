import time, torch
from sentence_transformers import SentenceTransformer

MODEL_NAME = os.getenv("MODEL_NAME", "intfloat/multilingual-e5-base")

DEVICE = os.getenv("DEVICE") or None
device = DEVICE or ("cuda" if torch.cuda.is_available() else "cpu")

m = SentenceTransformer(MODEL_NAME, device=device)
m.max_seq_length = 512
m.float()
props = torch.cuda.get_device_properties(0)
print(props.name, f"{props.total_memory/1e9:.1f} Go de VRAM")

texts = ["passage: " + "exemple de texte de readme sur un projet " * 60] * 256
m.encode(texts[:32], batch_size=32)  # échauffement

for bs in (8, 16, 32, 64, 128, 256, 512):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t = time.time()
    m.encode(texts, batch_size=bs)
    torch.cuda.synchronize()
    print(f"batch {bs:>3} : {256/(time.time()-t):6.1f} textes/s | VRAM pic {torch.cuda.max_memory_allocated()/1e9:.1f} Go")