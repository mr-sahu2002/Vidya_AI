import re
import tiktoken
from typing import List

def smart_chunk_text(text: str, max_tokens: int = 512) -> List[str]:
    # Improved handling for educational content
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Enhanced content type detection
    if is_mathematical_content(text):
        return chunk_mathematical_content(text, max_tokens)
    elif is_social_studies_content(text):
        return chunk_social_studies_content(text, max_tokens)
    elif is_STEM_content(text):
        return chunk_STEM_content(text, max_tokens)
    elif is_educational_content(text):
        return chunk_educational_content(text, max_tokens)
    else:
        return chunk_regular_text(text, max_tokens)

# New content detection functions
def is_social_studies_content(text: str) -> bool:
    """Detect social studies content (history, geography, civics)"""
    social_keywords = [
        'history', 'geography', 'civics', 'government', 'economics',
        'culture', 'society', 'historical', 'war', 'revolution',
        'country', 'nation', 'state', 'city', 'continent',
        'ancient', 'medieval', 'modern', 'century',
        'president', 'king', 'queen', 'emperor', 'prime minister',
        'law', 'constitution', 'amendment', 'rights',
        'population', 'migration', 'urban', 'rural'
    ]
    return any(keyword in text.lower() for keyword in social_keywords)

def is_STEM_content(text: str) -> bool:
    """Detect broader STEM content beyond pure math"""
    stem_keywords = [
        'science', 'technology', 'engineering', 'physics',
        'chemistry', 'biology', 'geology', 'astronomy',
        'experiment', 'hypothesis', 'theory', 'scientific method',
        'invention', 'innovation', 'design', 'prototype',
        'ecosystem', 'organism', 'cell', 'atom', 'molecule',
        'force', 'energy', 'motion', 'electricity',
        'computer', 'programming', 'algorithm', 'data',
        'system', 'model', 'simulation'
    ]
    return any(keyword in text.lower() for keyword in stem_keywords)

# Existing functions with enhancements
def is_mathematical_content(text: str) -> bool:
    """Check if content is primarily mathematical"""
    math_indicators = [
        r'\d+\s*[+\-*/=]\s*\d+',  # Basic math operations
        r'equation|formula|theorem|proof|solve|calculate',
        r'[∑∏∫∂≠≤≥±∞]',  # Math symbols
        r'\$.*?\$',  # LaTeX math
        r'\\[a-zA-Z]+',  # LaTeX commands
        r'graph|plot|chart|diagram',  # Visual math elements
        r'variable|constant|coefficient|polynomial'
    ]
    
    math_count = sum(len(re.findall(pattern, text.lower())) for pattern in math_indicators)
    return math_count > 2

def is_educational_content(text: str) -> bool:
    """Detect educational content patterns"""
    edu_keywords = [
        'chapter', 'exercise', 'solution', 'example', 
        'question', 'answer', 'formula', 'diagram',
        'lesson', 'unit', 'module', 'activity',
        'review', 'summary', 'key points', 'learning objective'
    ]
    return any(keyword in text.lower() for keyword in edu_keywords)


def chunk_social_studies_content(text: str, max_tokens: int) -> List[str]:
    """Special chunking for social studies content"""
    # Split by historical events, time periods, or conceptual sections
    sections = re.split(
        r'(\n\s*(?:Event|Period|Era|Topic|Concept|Case Study|Region)\s*[:.-])', 
        text
    )
    
    chunks = []
    current_chunk = ""
    
    for section in sections:
        if not section.strip():
            continue
            
        if count_tokens(current_chunk + section) <= max_tokens:
            current_chunk += section
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            # Handle long sections about specific events or concepts
            if count_tokens(section) > max_tokens:
                sub_sections = re.split(
                    r'(?:\n{2,}|[.!?]\s+(?=[A-Z]))', 
                    section
                )
                for sub in sub_sections:
                    if count_tokens(sub) > max_tokens:
                        sentences = split_into_sentences(sub)
                        chunks.extend(group_sentences_by_tokens(sentences, max_tokens))
                    else:
                        chunks.append(sub.strip())
            else:
                current_chunk = section
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks

def chunk_STEM_content(text: str, max_tokens: int) -> List[str]:
    """Special chunking for STEM content"""
    # Split by scientific concepts, experiments, or technical sections
    sections = re.split(
        r'(\n\s*(?:Experiment|Theory|Principle|Law|Concept|Process|System)\s*[:.-])', 
        text
    )
    
    chunks = []
    current_chunk = ""
    
    for section in sections:
        if not section.strip():
            continue
            
        if count_tokens(current_chunk + section) <= max_tokens:
            current_chunk += section
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            # Handle long technical descriptions
            if count_tokens(section) > max_tokens:
                # Try to preserve equations and formulas
                if is_mathematical_content(section):
                    math_chunks = chunk_mathematical_content(section, max_tokens)
                    chunks.extend(math_chunks)
                else:
                    sub_sections = re.split(
                        r'(?:\n{2,}|[.!?]\s+(?=[A-Z]))', 
                        section
                    )
                    for sub in sub_sections:
                        if count_tokens(sub) > max_tokens:
                            sentences = split_into_sentences(sub)
                            chunks.extend(group_sentences_by_tokens(sentences, max_tokens))
                        else:
                            chunks.append(sub.strip())
            else:
                current_chunk = section
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks


def chunk_mathematical_content(text: str, max_tokens: int) -> List[str]:
    """Improved chunking for mathematical content"""
    # Split by mathematical breaks (equations, examples, etc.)
    math_breaks = r'(\n(?=(?:Example|Problem|Solution|Theorem|Proof|Step \d+):))'
    parts = re.split(math_breaks, text)
    
    chunks = []
    current_chunk = ""
    
    for part in parts:
        if not part.strip():
            continue
            
        if count_tokens(current_chunk + part) <= max_tokens:
            current_chunk += part
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
           
            if count_tokens(part) > max_tokens:
                math_expr = extract_mathematical_expressions(part)
                for expr in math_expr:
                    if expr.strip():
                        chunks.append(expr.strip())
            else:
                current_chunk = part
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks

def chunk_educational_content(text: str, max_tokens: int) -> List[str]:
    """Enhanced chunking for educational material"""

    sections = re.split(r'(\n\s*(?:Chapter|Section|Unit|Exercise|Lesson)\s+\d+[:\.])', text)
    chunks = []
    current_chunk = ""
    
    for section in sections:
        if not section.strip():
            continue
            
        if count_tokens(current_chunk + section) <= max_tokens:
            current_chunk += section
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = section
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks

def extract_mathematical_expressions(text: str) -> List[str]:
    """Extract standalone mathematical expressions"""
  
    patterns = [
        r'\$.*?\$',  # LaTeX math
        r'\\[a-zA-Z]+\{.*?\}',  # LaTeX commands
        r'Equation:.*?(?=\n\n|$)',
        r'Proof:.*?(?=\n\n|$)',
        r'Solution:.*?(?=\n\n|$)',
        r'Theorem:.*?(?=\n\n|$)',
        r'Formula:.*?(?=\n\n|$)',
        r'Step \d+:.*?(?=\n\n|$)'
    ]
    
    expressions = []
    for pattern in patterns:
        expressions.extend(re.findall(pattern, text, re.DOTALL))
    
    if not expressions:
        # Split by equations or proofs
        expressions = re.split(r'(?:\n{2,}|\.\s+)(?=(?:Equation|Proof|Solution|Step)\s)', text)
    
    return expressions

def chunk_regular_text(text: str, max_tokens: int) -> List[str]:
    """Chunk by paragraphs first, then sentences"""
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        if not para.strip():
            continue
            
        if count_tokens(current_chunk + para) <= max_tokens:
            current_chunk += para + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            # If single paragraph is too long, split by sentences
            if count_tokens(para) > max_tokens:
                sentences = split_into_sentences(para)
                chunks.extend(group_sentences_by_tokens(sentences, max_tokens))
            else:
                current_chunk = para + "\n\n"
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks

def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences with better handling of abbreviations"""
    # Common abbreviations that shouldn't end sentences
    abbreviations = r'\b(?:Dr|Mr|Mrs|Ms|Prof|etc|vs|e\.g|i\.e|Fig|Ch|Sec|No|Vol|pp)\.'
    
    # Replace abbreviations temporarily
    text = re.sub(abbreviations, lambda m: m.group().replace('.', '<DOT>'), text)
    
    # Split by sentence endings
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Restore abbreviations
    sentences = [s.replace('<DOT>', '.') for s in sentences if s.strip()]
    
    return sentences

def group_sentences_by_tokens(sentences: List[str], max_tokens: int) -> List[str]:
    """Group sentences into chunks within token limit"""
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    chunks = []
    current_chunk = []
    current_token_count = 0
    
    for sentence in sentences:
        sentence_tokens = enc.encode(sentence)
        sentence_token_count = len(sentence_tokens)
        
        if current_token_count + sentence_token_count <= max_tokens:
            current_chunk.append(sentence)
            current_token_count += sentence_token_count
        else:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_token_count = sentence_token_count
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def count_tokens(text: str) -> int:
    """Count tokens using tiktoken"""
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(enc.encode(text))