"""
Indexes tax knowledge documents into a ChromaDB vector database.
Before running this script, you need to download the IRS forms and flows using the running scripts/irs-forms.py script.

Run this once before starting the chatbot:
    python indexer.py

Documents indexed:
  - data/irs_forms/  : IRS tax form PDFs (text extracted)
  - scripts/flows/   : End-to-end tax workflow examples
"""

import sys
import time
import os
from multiprocessing import Pool, cpu_count
from pathlib import Path

import chromadb
import fitz
import torch
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.config import CHROMA_DIR, DATA_DIR, FLOWS_DIR, IRS_FORMS_DIR
from app.constants import COLLECTION_NAME, EMBED_MODEL

BATCH_SIZE = 1000

CHUNK_SIZE = 1500    
CHUNK_OVERLAP = 200   


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class LocalEmbeddingFunction:
    """Wraps SentenceTransformer with explicit device selection for ChromaDB."""

    def __init__(self, model_name: str, device: str | None = None):
        self._model_name = model_name
        self.device = device or get_device()
        self.model = SentenceTransformer(model_name, device=self.device)
        default_batch_size = 8 if self.device == "mps" else 32
        self.encode_batch_size = int(os.getenv("INDEXER_EMBED_BATCH_SIZE", str(default_batch_size)))

    def __call__(self, input: list[str]) -> list[list[float]]:
        try:
            return self.model.encode(
                input,
                show_progress_bar=False,
                batch_size=self.encode_batch_size,
            ).tolist()
        except RuntimeError as e:
            # MPS can OOM on large or long-text batches; fallback to CPU and retry once.
            if self.device == "mps" and "MPS backend out of memory" in str(e):
                print("  WARNING: MPS out of memory during embedding. Falling back to CPU for this run.")
                if hasattr(torch, "mps"):
                    torch.mps.empty_cache()
                self.device = "cpu"
                self.model = self.model.to("cpu")
                self.encode_batch_size = int(os.getenv("INDEXER_EMBED_BATCH_SIZE_CPU", "16"))
                return self.model.encode(
                    input,
                    show_progress_bar=False,
                    batch_size=self.encode_batch_size,
                ).tolist()
            raise

    def embed_query(self, input: list[str]) -> list[list[float]]:
        return self.__call__(input)

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        return self.__call__(input)

    def name(self) -> str:
        return self._model_name


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract all text from a PDF using PyMuPDF."""
    try:
        doc = fitz.open(str(pdf_path))
        pages = [page.get_text() for page in doc]
        doc.close()
        return "\n\n".join(pages).strip()
    except Exception as e:
        print(f"  WARNING: could not extract text from {pdf_path.name}: {e}")
        return ""


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks, breaking on paragraph boundaries."""
    if len(text) <= chunk_size:
        return [text]

    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        candidate = f"{current}\n\n{para}".strip() if current else para
        if len(candidate) > chunk_size and current:
            chunks.append(current)
            # keep tail of current chunk as overlap seed
            current = current[-overlap:] + "\n\n" + para if overlap else para
        else:
            current = candidate

    if current.strip():
        chunks.append(current.strip())

    return chunks


def collect_pdf_documents(directory: Path) -> list[dict]:
    """Scan a directory for PDFs, extract text, chunk, and return doc dicts."""
    docs = []
    if not directory.exists():
        print(f"  Skipping {directory} (not found)")
        return docs

    pdf_files = sorted(directory.glob("*.pdf"))
    print(f"  {directory.relative_to(DATA_DIR)}: {len(pdf_files)} PDFs")

    workers = int(os.getenv("INDEXER_PDF_WORKERS", "1"))
    workers = max(1, min(workers, len(pdf_files)))
    print(f"  Extracting text with {workers} worker(s) …")
    if workers == 1:
        texts = [extract_pdf_text(p) for p in pdf_files]
    else:
        with Pool(workers) as pool:
            texts = pool.map(extract_pdf_text, pdf_files)

    for pdf_path, text in zip(pdf_files, texts):
        if not text:
            continue

        name = pdf_path.stem
        rel_path = str(pdf_path.relative_to(DATA_DIR))
        chunks = chunk_text(text)

        for ci, chunk in enumerate(chunks):
            chunk_id = f"form::{rel_path}::chunk{ci}"
            docs.append({
                "id": chunk_id,
                "text": chunk,
                "metadata": {
                    "source": "irs_form",
                    "form": name,
                    "file": rel_path,
                    "chunk": ci,
                    "total_chunks": len(chunks),
                },
            })

    return docs


def collect_documents() -> list[dict]:
    """Walk data/irs_forms and scripts/flows, return a list of {id, text, metadata} dicts."""
    docs = []

    docs.extend(collect_pdf_documents(IRS_FORMS_DIR))

    for flow_dir in sorted(FLOWS_DIR.rglob("*")):
        if not flow_dir.is_dir():
            continue
        for txt_path in sorted(flow_dir.glob("*.txt")):
            text = txt_path.read_text(encoding="utf-8", errors="ignore").strip()
            if not text:
                continue
            rel_path = Path("flows") / txt_path.relative_to(FLOWS_DIR)
            docs.append({
                "id": f"flow::{rel_path}",
                "text": text,
                "metadata": {
                    "source": "flow_example",
                    "flow": flow_dir.name,
                    "file": str(rel_path),
                },
            })

    return docs


def build_index(reset: bool = False) -> None:
    device = get_device()
    print(f"Data directory  : {DATA_DIR}")
    print(f"ChromaDB path   : {CHROMA_DIR}")
    print(f"Embedding model : {EMBED_MODEL}")
    print(f"Device          : {device}\n")

    # --- Stage 1: PDF extraction + chunking (no torch model yet) ---
    t0 = time.time()
    print("Scanning source directories …")
    all_docs = collect_documents()
    t_extract = time.time() - t0
    print(f"Total chunks found: {len(all_docs)}")
    print(f"  ⏱ Extraction + chunking: {t_extract:.1f}s")

    # --- Stage 2: Load model + Chroma, then embed/index ---
    embed_fn = LocalEmbeddingFunction(EMBED_MODEL, device=device)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    if reset and COLLECTION_NAME in [c.name for c in client.list_collections()]:
        print(f"Deleting existing collection '{COLLECTION_NAME}' …")
        client.delete_collection(COLLECTION_NAME)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    existing_ids = set(collection.get(include=[])["ids"])
    print(f"Existing documents in collection: {len(existing_ids)}")

    new_docs = [d for d in all_docs if d["id"] not in existing_ids]
    print(f"\nNew documents to index: {len(new_docs)}")

    if not new_docs:
        print("Nothing to index. Run with --reset to force re-indexing.")
        return

    t1 = time.time()
    total_batches = (len(new_docs) + BATCH_SIZE - 1) // BATCH_SIZE
    for i in range(0, len(new_docs), BATCH_SIZE):
        batch = new_docs[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        print(f"  Batch {batch_num}/{total_batches} — {len(batch)} chunks …", end=" ", flush=True)
        collection.add(
            ids=[d["id"] for d in batch],
            documents=[d["text"] for d in batch],
            metadatas=[d["metadata"] for d in batch],
        )
        print("done")

    t_embed = time.time() - t1
    print(f"\n  ⏱ Embedding + indexing: {t_embed:.1f}s")
    print(f"\nIndexed {len(new_docs)} documents in {t_extract + t_embed:.1f}s")
    print(f"Collection total: {collection.count()} documents")


if __name__ == "__main__":
    reset_flag = "--reset" in sys.argv
    if reset_flag:
        print("--reset flag detected: will delete and rebuild the index.\n")
    script_start = time.time()
    print(f"Script started at {time.strftime('%Y-%m-%d %H:%M:%S')} with model: {EMBED_MODEL}")
    build_index(reset=reset_flag)
    print(f"\nTotal time for model {EMBED_MODEL}: {time.time() - script_start:.1f}s")
