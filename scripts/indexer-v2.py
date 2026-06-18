"""
Indexes tax knowledge documents into a ChromaDB vector database.
Before running this script, you need to download the IRS forms and flows using the running scripts/irs-forms.py script.

Run this once before starting the chatbot:
    python scripts/indexer-v2.py

Documents indexed:
  - data/irs_forms/  : IRS tax form PDFs (text extracted)
  - scripts/flows/   : End-to-end tax workflow examples

Processes one source file at a time to keep memory use low on constrained hosts (e.g. Render 2GB).

Environment variables:
  INDEXER_ADD_BATCH_SIZE       Chroma add batch size (default: 100)
  INDEXER_EMBED_BATCH_SIZE     SentenceTransformer encode batch size (default: 8 on MPS, 16 on CPU/CUDA)
  INDEXER_EMBED_BATCH_SIZE_CPU CPU fallback batch size after MPS OOM (default: 8)
"""

import gc
import os
import sys
import time
from pathlib import Path

import chromadb
import fitz
import torch
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.config import CHROMA_DIR, DATA_DIR, FLOWS_DIR, IRS_FORMS_DIR
from app.constants import COLLECTION_NAME, EMBED_MODEL

ADD_BATCH_SIZE = int(os.getenv("INDEXER_ADD_BATCH_SIZE", "100"))

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
        default_batch_size = 8 if self.device == "mps" else 16
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
                self.encode_batch_size = int(os.getenv("INDEXER_EMBED_BATCH_SIZE_CPU", "8"))
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


def docs_from_pdf(pdf_path: Path) -> list[dict]:
    """Extract, chunk, and return doc dicts for a single PDF."""
    text = extract_pdf_text(pdf_path)
    if not text:
        return []

    name = pdf_path.stem
    rel_path = str(pdf_path.relative_to(DATA_DIR))
    chunks = chunk_text(text)
    docs: list[dict] = []

    for ci, chunk in enumerate(chunks):
        docs.append({
            "id": f"form::{rel_path}::chunk{ci}",
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


def doc_from_flow(txt_path: Path, flow_dir: Path) -> dict | None:
    """Read a flow example text file and return a single doc dict."""
    text = txt_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return None

    rel_path = Path("flows") / txt_path.relative_to(FLOWS_DIR)
    return {
        "id": f"flow::{rel_path}",
        "text": text,
        "metadata": {
            "source": "flow_example",
            "flow": flow_dir.name,
            "file": str(rel_path),
        },
    }


def add_documents(
    collection,
    docs: list[dict],
    existing_ids: set[str],
    batch_size: int = ADD_BATCH_SIZE,
) -> int:
    """Embed and add new docs in batches. Returns the number of documents added."""
    new_docs = [d for d in docs if d["id"] not in existing_ids]
    if not new_docs:
        return 0

    added = 0
    for i in range(0, len(new_docs), batch_size):
        batch = new_docs[i : i + batch_size]
        collection.add(
            ids=[d["id"] for d in batch],
            documents=[d["text"] for d in batch],
            metadatas=[d["metadata"] for d in batch],
        )
        existing_ids.update(d["id"] for d in batch)
        added += len(batch)

    return added


def index_pdf_directory(
    collection,
    directory: Path,
    existing_ids: set[str],
    batch_size: int = ADD_BATCH_SIZE,
) -> tuple[int, int, int]:
    """Index PDFs one file at a time. Returns (chunks_seen, chunks_added, pdfs_processed)."""
    if not directory.exists():
        print(f"  Skipping {directory} (not found)")
        return 0, 0, 0

    pdf_files = sorted(directory.glob("*.pdf"))
    print(f"  {directory.relative_to(DATA_DIR)}: {len(pdf_files)} PDFs")

    chunks_seen = 0
    chunks_added = 0
    pdfs_processed = 0

    for i, pdf_path in enumerate(pdf_files, start=1):
        docs = docs_from_pdf(pdf_path)
        if not docs:
            continue

        pdfs_processed += 1
        chunks_seen += len(docs)
        added = add_documents(collection, docs, existing_ids, batch_size)
        chunks_added += added

        if added:
            print(f"    [{i}/{len(pdf_files)}] {pdf_path.name}: +{added} chunk(s)")

        del docs
        if i % 100 == 0:
            gc.collect()

    return chunks_seen, chunks_added, pdfs_processed


def index_flow_documents(
    collection,
    existing_ids: set[str],
    batch_size: int = ADD_BATCH_SIZE,
) -> tuple[int, int]:
    """Index flow example text files one at a time. Returns (files_seen, files_added)."""
    if not FLOWS_DIR.exists():
        print(f"  Skipping {FLOWS_DIR} (not found)")
        return 0, 0

    files_seen = 0
    files_added = 0

    for flow_dir in sorted(FLOWS_DIR.rglob("*")):
        if not flow_dir.is_dir():
            continue
        for txt_path in sorted(flow_dir.glob("*.txt")):
            doc = doc_from_flow(txt_path, flow_dir)
            if doc is None:
                continue

            files_seen += 1
            added = add_documents(collection, [doc], existing_ids, batch_size)
            if added:
                files_added += added
                print(f"    flow {txt_path.relative_to(FLOWS_DIR)}: +{added}")

    return files_seen, files_added


def build_index(reset: bool = False) -> None:
    device = get_device()
    print(f"Data directory  : {DATA_DIR}")
    print(f"ChromaDB path   : {CHROMA_DIR}")
    print(f"Embedding model : {EMBED_MODEL}")
    print(f"Device          : {device}")
    print(f"Add batch size  : {ADD_BATCH_SIZE}\n")

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
    print(f"Existing documents in collection: {len(existing_ids)}\n")

    t0 = time.time()
    print("Indexing IRS form PDFs (streaming, one file at a time) …")
    pdf_chunks_seen, pdf_chunks_added, pdfs_processed = index_pdf_directory(
        collection, IRS_FORMS_DIR, existing_ids
    )

    print("\nIndexing flow examples …")
    flow_files_seen, flow_files_added = index_flow_documents(collection, existing_ids)

    t_total = time.time() - t0
    total_added = pdf_chunks_added + flow_files_added

    print(f"\nPDFs processed     : {pdfs_processed}")
    print(f"PDF chunks seen    : {pdf_chunks_seen}")
    print(f"Flow files seen    : {flow_files_seen}")
    print(f"New documents added: {total_added}")
    print(f"  ⏱ Indexing time  : {t_total:.1f}s")

    if total_added == 0:
        print("Nothing to index. Run with --reset to force re-indexing.")
    else:
        print(f"Collection total   : {collection.count()} documents")


if __name__ == "__main__":
    reset_flag = "--reset" in sys.argv
    if reset_flag:
        print("--reset flag detected: will delete and rebuild the index.\n")
    script_start = time.time()
    print(f"Script started at {time.strftime('%Y-%m-%d %H:%M:%S')} with model: {EMBED_MODEL}")
    build_index(reset=reset_flag)
    print(f"\nTotal time for model {EMBED_MODEL}: {time.time() - script_start:.1f}s")
