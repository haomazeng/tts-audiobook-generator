import fitz  # PyMuPDF
import pdfplumber
from pathlib import Path
from typing import List
import re

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF file using PyMuPDF for better Chinese support."""
    doc = fitz.open(pdf_path)
    text_parts = []

    for page in doc:
        text = page.get_text()
        if text.strip():
            text_parts.append(text)

    doc.close()
    return "\n".join(text_parts)

def extract_text_from_md(md_path: str) -> str:
    """Extract text from Markdown file."""
    with open(md_path, 'r', encoding='utf-8') as f:
        return f.read()

def split_into_sentences(text: str, max_chunk_size: int = 500) -> List[str]:
    """
    Split text into sentences, respecting punctuation marks.
    Merge short sentences into chunks that don't exceed max_chunk_size.
    """
    # Chinese punctuation marks for sentence splitting
    sentence_endings = r'([。！？；])'
    sentences = re.split(sentence_endings, text)

    # Re-attach punctuation to sentences
    reconstructed = []
    i = 0
    while i < len(sentences):
        if i + 1 < len(sentences) and sentences[i + 1] in '。！？；':
            reconstructed.append(sentences[i] + sentences[i + 1])
            i += 2
        else:
            if sentences[i]:
                reconstructed.append(sentences[i])
            i += 1

    # Filter empty sentences
    sentences = [s.strip() for s in reconstructed if s.strip()]

    # Merge sentences into chunks
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chunk_size:
            current_chunk += sentence
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def extract_and_split(file_path: str, max_chunk_size: int = 500) -> List[str]:
    """Extract text from file and split into chunks based on file type."""
    path = Path(file_path)

    if path.suffix.lower() == '.pdf':
        text = extract_text_from_pdf(file_path)
    elif path.suffix.lower() in ['.md', '.markdown']:
        text = extract_text_from_md(file_path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    return split_into_sentences(text, max_chunk_size)
