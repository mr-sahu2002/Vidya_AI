import pdfplumber
import re
import logging
from pathlib import Path
from typing import List, Dict

def extract_content_universal(file_path: str, metadata: Dict) -> List[Dict]:
    file_extension = Path(file_path).suffix.lower()
    
    if file_extension == '.pdf':
        return extract_pdf_content(file_path, metadata)
    elif file_extension == '.txt':
        return extract_text_content(file_path, metadata)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

def extract_pdf_content(file_path: str, metadata: Dict) -> List[Dict]:
    all_chunks = []
    filename = Path(file_path).name
    
    try:
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text() or ""
                if text.strip():
                    text_chunks = chunk_text_with_metadata(
                        text, metadata, filename, page_num
                    )
                    all_chunks.extend(text_chunks)
        return all_chunks
    except Exception as e:
        logging.error(f"PDF extraction failed: {str(e)}")
        return []

def extract_text_content(file_path: str, metadata: Dict) -> List[Dict]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        filename = Path(file_path).name
        return chunk_text_with_metadata(text, metadata, filename, 1)
    except Exception as e:
        logging.error(f"Text extraction failed: {str(e)}")
        return []

def chunk_text_with_metadata(
    text: str, 
    metadata: Dict, 
    filename: str, 
    page_num: int
) -> List[Dict]:
    from .chunker import smart_chunk_text
    
    chunks = smart_chunk_text(text)
    enriched_chunks = []
    
    # Get subject from class registry
    class_id = metadata.get("class_id")
    subject = metadata.get("subject", "general")
    
    for idx, chunk in enumerate(chunks):
        chunk_data = {
            "text": chunk,
            "class_id": class_id,
            "subject": subject,
            "teacher_id": metadata.get("teacher_id", ""),
            "grade_level": metadata.get("grade_level", "unknown"),
            "filename": filename,
            "page_number": page_num,
            "chunk_index": idx,
            "language": detect_language(chunk),
            "difficulty": estimate_difficulty(chunk),
            "word_count": len(chunk.split()),
            "uploaded_at": metadata.get("uploaded_at"),
        }
        enriched_chunks.append(chunk_data)
    
    return enriched_chunks

def detect_language(text: str) -> str:
    """Detect English, Hindi, or Kannada"""
    if re.search(r'[\u0900-\u097F]', text):  # Hindi
        return "hindi"
    elif re.search(r'[\u0C80-\u0CFF]', text):  # Kannada
        return "kannada"
    else:
        return "english"

def estimate_difficulty(text: str) -> str:
    """Simplified difficulty estimation"""
    word_count = len(text.split())
    complex_words = len(re.findall(r'\b\w{8,}\b', text))
    
    if complex_words > 5 or word_count > 100:
        return "advanced"
    elif complex_words > 2 or word_count > 50:
        return "intermediate"
    else:
        return "basic"