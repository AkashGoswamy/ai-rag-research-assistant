import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

from pypdf import PdfReader


@dataclass
class Chunk:
    doc_id: str
    doc_name: str
    page_number: int
    text: str
    chunk_index: int


def clean_text(text: str) -> str:
    # Remove control characters (except newlines/tabs which we'll normalize anyway)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    # Collapse multiple whitespace/newlines into single spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_pages(pdf_path: str) -> List[str]:
    """Extract raw text per page from a PDF. Returns a list where index i
    is the text of page i+1."""
    reader = PdfReader(pdf_path)
    pages = []
    for page in reader.pages:
        raw = page.extract_text() or ""
        pages.append(clean_text(raw))
    return pages


def _split_on_separator(text: str, separator: str) -> List[str]:
    if separator == "":
        return list(text)
    parts = text.split(separator)
    # Re-attach separator to keep boundaries meaningful except for the last piece
    result = []
    for i, part in enumerate(parts):
        if i < len(parts) - 1:
            result.append(part + separator)
        else:
            if part:
                result.append(part)
    return result


def _recursive_split(text: str, chunk_size: int, separators: List[str]) -> List[str]:
    """Recursively split text using a priority list of separators
    (paragraph -> sentence -> word -> char), only descending to the next
    separator when a piece is still too large."""
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    if not separators:
        # Fallback: hard character split
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    separator = separators[0]
    remaining_separators = separators[1:]

    pieces = _split_on_separator(text, separator)

    final_chunks = []
    for piece in pieces:
        if len(piece) <= chunk_size:
            if piece.strip():
                final_chunks.append(piece)
        else:
            final_chunks.extend(_recursive_split(piece, chunk_size, remaining_separators))

    return final_chunks


def _merge_with_overlap(pieces: List[str], chunk_size: int, overlap: int) -> List[str]:
    """Greedily pack small pieces into chunks close to chunk_size, carrying
    overlap characters of trailing context into the next chunk."""
    if not pieces:
        return []

    chunks = []
    current = ""

    for piece in pieces:
        if len(current) + len(piece) <= chunk_size:
            current += piece
        else:
            if current.strip():
                chunks.append(current.strip())
            # start new chunk, carry overlap from the end of the previous chunk
            overlap_text = current[-overlap:] if overlap > 0 else ""
            current = overlap_text + piece

    if current.strip():
        chunks.append(current.strip())

    return chunks


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
    """Split text into overlapping chunks, using recursive fallback:
    paragraph -> sentence -> word -> char."""
    separators = ["\n\n", ". ", " ", ""]
    pieces = _recursive_split(text, chunk_size, separators)
    return _merge_with_overlap(pieces, chunk_size, overlap)


def ingest_pdf(pdf_path: str, doc_id: str, chunk_size: int = 800, overlap: int = 150) -> List[Chunk]:
    """Parse a single PDF into a list of Chunk objects, tagged with
    doc_id, doc_name, and page_number."""
    doc_name = Path(pdf_path).name
    pages = extract_pages(pdf_path)

    chunks: List[Chunk] = []
    chunk_index = 0
    for page_number, page_text in enumerate(pages, start=1):
        if not page_text.strip():
            continue
        page_chunks = chunk_text(page_text, chunk_size=chunk_size, overlap=overlap)
        for c in page_chunks:
            chunks.append(Chunk(
                doc_id=doc_id,
                doc_name=doc_name,
                page_number=page_number,
                text=c,
                chunk_index=chunk_index,
            ))
            chunk_index += 1

    return chunks


class DocumentStore:
    """Holds a FAISS index plus per-chunk metadata for one or more
    ingested PDFs. Supports adding multiple documents into the same
    store, each tagged by their doc_id/doc_name."""

    _model = None  # class-level singleton, loaded once

    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.embedding_model_name = embedding_model_name
        self.index = None  # faiss.IndexFlatIP, created lazily on first add
        self.metadata: List[Chunk] = []  # parallel list, index i <-> vector i in FAISS

    @classmethod
    def _get_model(cls, embedding_model_name: str):
        if cls._model is None:
            from sentence_transformers import SentenceTransformer
            cls._model = SentenceTransformer(embedding_model_name)
        return cls._model

    def _embed(self, texts: List[str]):
        import numpy as np
        model = self._get_model(self.embedding_model_name)
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        # Normalize so inner product == cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12  # avoid division by zero
        embeddings = embeddings / norms
        return embeddings.astype('float32')

    def add_pdf(self, pdf_path: str, doc_id: str, chunk_size: int = 800, overlap: int = 150) -> int:
        """Ingest one PDF: parse, chunk, embed, and add to the index.
        Returns the number of chunks added."""
        import faiss
        import numpy as np

        chunks = ingest_pdf(pdf_path, doc_id=doc_id, chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            return 0

        texts = [c.text for c in chunks]
        embeddings = self._embed(texts)

        if self.index is None:
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)

        self.index.add(embeddings)
        self.metadata.extend(chunks)

        return len(chunks)

    def total_chunks(self) -> int:
        return len(self.metadata)

    def chunks_per_doc(self) -> dict:
        counts = {}
        for c in self.metadata:
            counts[c.doc_name] = counts.get(c.doc_name, 0) + 1
        return counts
