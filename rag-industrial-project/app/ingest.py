import os
from typing import List, Dict, Any
import uuid
from .utils import simple_chunk_text

try:
    import pdfplumber
except Exception:
    pdfplumber = None

try:
    import docx
except Exception:
    docx = None

try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None

def extract_text_from_file(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        if pdfplumber is None:
            raise RuntimeError("pdfplumber not installed")
        text_parts = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                p = page.extract_text()
                if p:
                    text_parts.append(p)
        return "\n\n".join(text_parts)
    elif ext in [".docx", ".doc"]:
        if docx is None:
            raise RuntimeError("python-docx not installed")
        doc = docx.Document(path)
        paras = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
        return "\n\n".join(paras)
    elif ext in [".html", ".htm"]:
        if BeautifulSoup is None:
            raise RuntimeError("beautifulsoup4 not installed")
        with open(path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")
        for s in soup(["script", "style"]):
            s.decompose()
        return soup.get_text(separator="\n")
    else:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

def ingest_file_to_chunks(path: str, title: str = None) -> List[Dict[str, Any]]:
    text = extract_text_from_file(path)
    # simple normalization
    text = text.replace("\r\n", "\n")
    chunks = simple_chunk_text(text)
    doc_id = str(uuid.uuid4())
    records = []
    for i, c in enumerate(chunks, start=1):
        rec = {
            "id": f"{doc_id}_{i}",
            "doc_id": doc_id,
            "text": c,
            "metadata": {"title": title or os.path.basename(path), "source": os.path.abspath(path), "chunk_index": i},
        }
        records.append(rec)
    return records
