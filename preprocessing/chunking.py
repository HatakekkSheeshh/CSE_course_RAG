"""
Text chunking module for RAG indexing.

Provides functions to chunk syllabus and material documents into
smaller text segments with preserved metadata.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import tiktoken
from .manifest import DocChunk, Metadata


def _get_encoding():
    """Get tiktoken encoding for token counting."""
    try:
        return tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer
    except Exception:
        # Fallback to approximate counting
        return None


def _count_tokens(text: str, encoding: Optional[tiktoken.Encoding]) -> int:
    """Count tokens in text."""
    if encoding:
        return len(encoding.encode(text))
    # Fallback: approximate 1 token = 4 characters
    return len(text) // 4


def chunk_text(
    text: str,
    metadata: Metadata,
    chunk_id_prefix: str,
    chunk_size: int = 512,
    overlap: int = 50,
    encoding: Optional[tiktoken.Encoding] = None
) -> List[DocChunk]:
    """
    Chunk text into overlapping segments.
    
    Args:
        text: Text to chunk
        metadata: Metadata to attach to each chunk
        chunk_id_prefix: Prefix for chunk IDs (e.g., "CO1005-slide-001")
        chunk_size: Target chunk size in tokens
        overlap: Overlap size in tokens between chunks
        encoding: tiktoken encoding for token counting
        
    Returns:
        List of DocChunk objects
    """
    if not text or not text.strip():
        return []
    
    chunks = []
    encoding = encoding or _get_encoding()
    
    # Simple character-based chunking with overlap
    # Approximate token count: 1 token ≈ 4 characters
    char_chunk_size = chunk_size * 4
    char_overlap = overlap * 4
    
    start = 0
    chunk_idx = 0
    
    while start < len(text):
        end = start + char_chunk_size
        
        # Extract chunk
        chunk_text_segment = text[start:end].strip()
        
        if chunk_text_segment:
            chunk_id = f"{chunk_id_prefix}-{chunk_idx:04d}"
            
            # Create chunk with same metadata
            chunk = DocChunk(
                id=chunk_id,
                text=chunk_text_segment,
                metadata=metadata
            )
            chunks.append(chunk)
            chunk_idx += 1
        
        # Move start position with overlap
        start = end - char_overlap
        if start >= len(text):
            break
    
    return chunks


def chunk_syllabus(syllabus_json: Dict[str, Any]) -> List[DocChunk]:
    """
    Chunk syllabus JSON into DocChunk objects.
    
    Args:
        syllabus_json: Parsed syllabus JSON dictionary
        
    Returns:
        List of DocChunk objects
    """
    chunks = []
    encoding = _get_encoding()
    
    # Extract metadata
    metadata_dict = syllabus_json.get("metadata", {})
    course_info = syllabus_json.get("course_info", {})
    course_id = course_info.get("course_id", "") or metadata_dict.get("course_id", "")
    
    # Create base metadata
    base_metadata = Metadata(
        doc_type="syllabus",
        course_id=course_id,
        source_file=metadata_dict.get("source_file"),
        page_index=metadata_dict.get("page_index", 0),
        language=metadata_dict.get("language", "en"),
        ocr_engine=metadata_dict.get("ocr_engine"),
        extractor_version=metadata_dict.get("extractor_version"),
        timestamp=metadata_dict.get("timestamp")
    )
    
    # Chunk course info section
    if course_info:
        course_info_text = f"Course: {course_info.get('title', '')} (ID: {course_id})\n"
        if course_info.get("credits"):
            course_info_text += f"Credits: {course_info.get('credits')}\n"
        if course_info.get("applied_semester"):
            course_info_text += f"Semester: {course_info.get('applied_semester')}\n"
        
        if course_info_text.strip():
            prefix = f"{course_id}-syllabus-info"
            chunks.extend(chunk_text(
                course_info_text,
                base_metadata,
                prefix,
                chunk_size=256,  # Smaller chunks for structured info
                overlap=25,
                encoding=encoding
            ))
    
    # Chunk assessments section
    assessments = syllabus_json.get("assessments", [])
    if assessments:
        assessment_texts = []
        for assessment in assessments:
            assessment_text = f"{assessment.get('name', '')}: "
            if assessment.get("hours"):
                assessment_text += f"{assessment.get('hours')} hours. "
            if assessment.get("ratio") is not None:
                assessment_text += f"Weight: {assessment.get('ratio')}%. "
            
            eval_types = assessment.get("evaluation_type", [])
            if eval_types:
                assessment_text += "Evaluation: "
                for eval_type in eval_types:
                    assessment_text += f"{eval_type.get('name', '')} ({eval_type.get('ratio', '')}%), "
            
            assessment_texts.append(assessment_text.strip())
        
        if assessment_texts:
            assessment_full_text = "\n".join(assessment_texts)
            prefix = f"{course_id}-syllabus-assessments"
            chunks.extend(chunk_text(
                assessment_full_text,
                base_metadata,
                prefix,
                chunk_size=512,
                overlap=50,
                encoding=encoding
            ))
    
    # Chunk course description
    course_des = syllabus_json.get("course_des", {})
    if course_des:
        desc_text = ""
        if course_des.get("description"):
            desc_text += f"Description: {course_des.get('description')}\n"
        if course_des.get("material"):
            desc_text += f"Materials: {course_des.get('material')}\n"
        
        if desc_text.strip():
            prefix = f"{course_id}-syllabus-description"
            chunks.extend(chunk_text(
                desc_text,
                base_metadata,
                prefix,
                chunk_size=512,
                overlap=50,
                encoding=encoding
            ))
    
    # Chunk raw OCR text (main content)
    raw_text = syllabus_json.get("raw_ocr_text", "")
    if raw_text and raw_text.strip():
        prefix = f"{course_id}-syllabus-raw"
        chunks.extend(chunk_text(
            raw_text,
            base_metadata,
            prefix,
            chunk_size=512,
            overlap=50,
            encoding=encoding
        ))
    
    return chunks


def chunk_material(material_json: Dict[str, Any]) -> List[DocChunk]:
    """
    Chunk material JSON into DocChunk objects.
    
    Args:
        material_json: Parsed material JSON dictionary
        
    Returns:
        List of DocChunk objects
    """
    chunks = []
    encoding = _get_encoding()
    
    course_name = material_json.get("course", "")
    course_id = material_json.get("course_id", "")
    slides = material_json.get("slides", [])
    
    for slide in slides:
        # Extract slide metadata
        slide_metadata_dict = slide.get("metadata", {})
        
        base_metadata = Metadata(
            doc_type="slide",
            course_id=course_id or slide_metadata_dict.get("course_id", ""),
            source_file=slide_metadata_dict.get("source_file") or slide.get("source_file"),
            page_index=slide.get("page_index", slide_metadata_dict.get("page_index", 0)),
            language=slide_metadata_dict.get("language", "en"),
            ocr_engine=slide_metadata_dict.get("ocr_engine"),
            extractor_version=slide_metadata_dict.get("extractor_version"),
            timestamp=slide_metadata_dict.get("timestamp")
        )
        
        # Get raw text from slide
        raw_text = slide.get("raw_text", "")
        if not raw_text or not raw_text.strip():
            continue
        
        # Create chunk prefix
        chapter_num = slide.get("chapter_num", 0)
        page_idx = slide.get("page_index", 0)
        prefix = f"{course_id}-chapter-{chapter_num}-slide-{page_idx:03d}"
        
        # Chunk the slide text
        slide_chunks = chunk_text(
            raw_text,
            base_metadata,
            prefix,
            chunk_size=512,
            overlap=50,
            encoding=encoding
        )
        
        chunks.extend(slide_chunks)
    
    return chunks

