from pathlib import Path
import json
import time

import torch
import faiss
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from tqdm.auto import tqdm  # for progress bar

# =======================================
# CONFIG
# =======================================

# True  = only process FIRST PDF (fast test)
# False = process ALL PDFs (full run)
TEST_MODE = False

EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Paths: script is at: project_root/scripts/build_knowledge_base.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_ROOT / "data" / "docs"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
VSTORE_DIR = ARTIFACTS_DIR / "vectorstores"

INDEX_PATH = VSTORE_DIR / "ipm_faiss.index"
METADATA_PATH = VSTORE_DIR / "ipm_metadata.json"


# =======================================
# HELPERS
# =======================================

def extract_page_chunks(pdf_path: Path, max_page_chunk_chars: int = 4000):
    """
    Extract text page-by-page from a PDF and use each page (or subpage) as a chunk.
    Much more memory friendly than slicing one giant string.
    """
    reader = PdfReader(str(pdf_path))
    chunks = []
    total_chars = 0

    for page_idx, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.strip()
        if not text:
            continue

        total_chars += len(text)

        # If a page is too long, split it into smaller pieces
        start = 0
        while start < len(text):
            end = min(len(text), start + max_page_chunk_chars)
            chunk = text[start:end]
            if chunk.strip():
                chunks.append((page_idx, chunk))
            start = end

    return chunks, total_chars


def infer_crop_from_path(path: Path) -> str:
    # data/docs/rice/xxx.pdf -> "rice"
    return path.parent.name.lower()


# =======================================
# MAIN
# =======================================

def main():
    start_time = time.time()

    print("=" * 80)
    print("Building Smart Agro-Cure IPM Knowledge Base (RAG)")
    print("=" * 80)
    print(f"Project root      : {PROJECT_ROOT}")
    print(f"Docs directory    : {DOCS_DIR}")
    print(f"Artifacts dir     : {ARTIFACTS_DIR}")
    print(f"Vectorstore dir   : {VSTORE_DIR}")
    print(f"Embedding model   : {EMB_MODEL_NAME}")
    print(f"Using device      : {DEVICE}")
    print(f"TEST_MODE         : {TEST_MODE}")
    print("-" * 80)

    torch.set_num_threads(4)  # keep CPU from going crazy

    # -------------------------------
    # 1) Load PDF paths
    # -------------------------------
    pdf_paths = sorted(DOCS_DIR.rglob("*.pdf"))
    if not pdf_paths:
        print(f"[ERROR] No PDFs found under {DOCS_DIR}")
        return

    print(f"Found {len(pdf_paths)} PDF(s):")
    for p in pdf_paths:
        print(f"  - {p.relative_to(PROJECT_ROOT)}")

    if TEST_MODE:
        print("\n[INFO] TEST_MODE is ON -> will process ONLY the first PDF.\n")

    # -------------------------------
    # 2) Load embedding model
    # -------------------------------
    t0 = time.time()
    print("\n[STEP] Loading sentence-transformer model...")
    emb_model = SentenceTransformer(EMB_MODEL_NAME, device=DEVICE)
    print(f"[OK] Model loaded in {time.time() - t0:.2f} seconds.")

    # -------------------------------
    # 3) Extract + collect all chunks
    # -------------------------------
    all_texts = []
    all_metadata = []
    total_chars = 0

    print("\n[STEP] Extracting page-based chunks from PDFs...")
    for pdf_idx, pdf_path in enumerate(pdf_paths, start=1):
        crop = infer_crop_from_path(pdf_path)
        doc_name = pdf_path.name

        print("-" * 80)
        print(f"[PDF {pdf_idx}/{len(pdf_paths)}] {pdf_path.relative_to(PROJECT_ROOT)}")
        print(f"  -> crop = {crop}")

        t_pdf = time.time()
        page_chunks, chars = extract_page_chunks(pdf_path)
        total_chars += chars
        print(f"  -> extracted {chars} characters into {len(page_chunks)} chunks "
              f"in {time.time() - t_pdf:.2f} s")

        for local_chunk_id, (page_idx, chunk) in enumerate(page_chunks):
            all_texts.append(chunk)
            all_metadata.append(
                {
                    "id": len(all_metadata),
                    "crop": crop,
                    "doc_name": doc_name,
                    "page_idx": page_idx,
                    "chunk_id": local_chunk_id,
                }
            )

        if TEST_MODE:
            print("\n[TEST_MODE] Stopping after this PDF.")
            break

    print("-" * 80)
    print(f"[INFO] Total characters processed : {total_chars}")
    print(f"[INFO] Total chunks created       : {len(all_texts)}")

    if not all_texts:
        print("[ERROR] No text chunks were created. Aborting.")
        return

    # -------------------------------
    # 4) Compute embeddings
    # -------------------------------
    print("\n[STEP] Computing embeddings for all chunks...")
    batch_size = 32
    num_chunks = len(all_texts)
    print(f"[INFO] Using batch_size={batch_size}, num_chunks={num_chunks}")

    t_embed_start = time.time()
    all_embeddings = []

    for start in tqdm(range(0, num_chunks, batch_size),
                      desc="Embedding", unit="batch"):
        end = min(num_chunks, start + batch_size)
        batch_texts = all_texts[start:end]
        batch_emb = emb_model.encode(
            batch_texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        all_embeddings.append(batch_emb)

    import numpy as np
    embeddings = np.concatenate(all_embeddings, axis=0)

    embed_time = time.time() - t_embed_start
    print(f"[OK] Embeddings computed in {embed_time:.2f} seconds.")
    print(f"[INFO] Average time per chunk: {embed_time / num_chunks:.4f} s")

    # Rough ETA (you already see it live in tqdm, this is summary)
    est_per_batch = embed_time / (num_chunks / batch_size)
    print(f"[INFO] Approx batch speed: {est_per_batch:.2f} s / batch "
          f"~ {batch_size / est_per_batch:.2f} chunks/s")

    # -------------------------------
    # 5) Build FAISS index
    # -------------------------------
    print("\n[STEP] Building FAISS index...")
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings.astype("float32"))
    print(f"[OK] Index built with {index.ntotal} vectors of dimension {d}.")

    # -------------------------------
    # 6) Save index + metadata
    # -------------------------------
    print("\n[STEP] Saving index and metadata to disk...")
    VSTORE_DIR.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(INDEX_PATH))
    print(f"[OK] Saved FAISS index to: {INDEX_PATH}")

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "texts": all_texts,
                "metadata": all_metadata,
                "embedding_model": EMB_MODEL_NAME,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[OK] Saved metadata to: {METADATA_PATH}")

    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print(" Finished building IPM knowledge base.")
    print(f"   Total time: {total_time:.2f} seconds "
          f"({total_time/60:.2f} minutes)")
    print("=" * 80)


if __name__ == "__main__":
    main()
