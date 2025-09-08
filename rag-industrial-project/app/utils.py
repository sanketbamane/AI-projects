import re
from typing import List, Dict, Any
import uuid

def simple_chunk_text(text: str, chunk_size_chars: int = 2000, overlap_chars: int = 200) -> List[str]:
    text = text.replace("\r\n", "\n")
    paragraphs = text.split("\n\n")
    chunks = []
    buffer = ""
    for p in paragraphs:
        if len(buffer) + len(p) + 2 <= chunk_size_chars:
            buffer = (buffer + "\n\n" + p).strip()
        else:
            if buffer:
                chunks.append(buffer.strip())
            if len(p) > chunk_size_chars:
                start = 0
                while start < len(p):
                    part = p[start:start + chunk_size_chars]
                    chunks.append(part.strip())
                    start += chunk_size_chars - overlap_chars
                buffer = ""
            else:
                buffer = p
    if buffer:
        chunks.append(buffer.strip())
    # Add overlap stitching
    stitched = []
    for i, c in enumerate(chunks):
        if i == 0:
            stitched.append(c)
        else:
            prev = stitched[-1]
            overlap = prev[-overlap_chars:] if len(prev) > overlap_chars else prev
            stitched.append((overlap + "\n" + c).strip())
    return stitched

def build_prompt(question: str, chunks: List[Dict[str, Any]], instructions: str = None) -> str:
    system = instructions or (
        "You are a helpful domain expert. Answer using only the provided document excerpts. Cite sources inline using [DOC<n>]. If not supported, say 'I don\\'t know'."
    )
    parts = ["System: " + system, "\nUser question: " + question, "\nProvided documents:"]
    for i, ch in enumerate(chunks, start=1):
        title = ch.get("metadata", {}).get("title") or f"doc_{i}"
        meta = f"Title: {title} | Source: {ch.get('metadata', {}).get('source','')}"
        parts.append(f"[DOC{i}] {meta}\n" + ch["text"][:3000])
    parts.append("\nAnswer (include citations like [DOC1]):")
    return "\n\n".join(parts)
