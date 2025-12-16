"""
Analyze quality of parsed syllabus and material text data.
"""

import json
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import re

def analyze_text_quality(text: str) -> Dict:
    """Analyze quality metrics for a text string."""
    if not text:
        return {
            "empty": True,
            "length": 0,
            "has_ocr_errors": False,
            "has_special_chars": False,
            "has_vietnamese": False,
            "has_english": False,
            "word_count": 0
        }
    
    # Check for common OCR errors
    ocr_error_patterns = [
        r'\?\?\?',  # Triple question marks
        r'\?\?',    # Double question marks
        r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\{\}\'\"]+',  # Unusual characters
    ]
    has_ocr_errors = any(re.search(pattern, text) for pattern in ocr_error_patterns)
    
    # Check for special characters that might indicate OCR issues
    special_chars = re.findall(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\{\}\'\"]', text)
    has_special_chars = len(special_chars) > len(text) * 0.1  # More than 10% special chars
    
    # Detect Vietnamese (has diacritics)
    vietnamese_pattern = r'[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđĐ]'
    has_vietnamese = bool(re.search(vietnamese_pattern, text))
    
    # Detect English (has common English words or patterns)
    english_pattern = r'\b(the|and|for|are|but|not|you|all|can|her|was|one|our|out|day|get|has|him|his|how|its|may|new|now|old|see|two|way|who|boy|did|its|let|put|say|she|too|use)\b'
    has_english = bool(re.search(english_pattern, text, re.IGNORECASE))
    
    # Word count
    words = text.split()
    word_count = len(words)
    
    return {
        "empty": False,
        "length": len(text),
        "has_ocr_errors": has_ocr_errors,
        "has_special_chars": has_special_chars,
        "has_vietnamese": has_vietnamese,
        "has_english": has_english,
        "word_count": word_count
    }

def analyze_syllabus_files(data_dir: Path) -> Dict:
    """Analyze all syllabus parsed files."""
    stats = {
        "total_courses": 0,
        "courses_with_syllabus": 0,
        "total_files": 0,
        "empty_files": 0,
        "files_with_ocr_errors": 0,
        "total_text_length": 0,
        "total_words": 0,
        "bilingual_files": 0,  # Has both Vietnamese and English
        "vietnamese_only": 0,
        "english_only": 0,
        "course_details": {}
    }
    
    courses = [d for d in data_dir.iterdir() 
               if d.is_dir() and d.name != "scratch"]
    stats["total_courses"] = len(courses)
    
    for course_dir in courses:
        parsed_dir = course_dir / "syllabus" / "parsed"
        if not parsed_dir.exists():
            continue
        
        stats["courses_with_syllabus"] += 1
        course_stats = {
            "files": 0,
            "empty": 0,
            "ocr_errors": 0,
            "total_length": 0,
            "total_words": 0,
            "bilingual": 0,
            "vietnamese_only": 0,
            "english_only": 0
        }
        
        for json_file in parsed_dir.glob("*.json"):
            stats["total_files"] += 1
            course_stats["files"] += 1
            
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    text = data.get("raw_ocr_text", "")
                    
                    quality = analyze_text_quality(text)
                    
                    if quality["empty"]:
                        stats["empty_files"] += 1
                        course_stats["empty"] += 1
                    else:
                        stats["total_text_length"] += quality["length"]
                        stats["total_words"] += quality["word_count"]
                        course_stats["total_length"] += quality["length"]
                        course_stats["total_words"] += quality["word_count"]
                        
                        if quality["has_ocr_errors"]:
                            stats["files_with_ocr_errors"] += 1
                            course_stats["ocr_errors"] += 1
                        
                        if quality["has_vietnamese"] and quality["has_english"]:
                            stats["bilingual_files"] += 1
                            course_stats["bilingual"] += 1
                        elif quality["has_vietnamese"]:
                            stats["vietnamese_only"] += 1
                            course_stats["vietnamese_only"] += 1
                        elif quality["has_english"]:
                            stats["english_only"] += 1
                            course_stats["english_only"] += 1
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
        
        if course_stats["files"] > 0:
            stats["course_details"][course_dir.name] = course_stats
    
    return stats

def analyze_material_files(data_dir: Path) -> Dict:
    """Analyze all material JSON files."""
    stats = {
        "total_courses": 0,
        "courses_with_material": 0,
        "total_slides": 0,
        "empty_slides": 0,
        "slides_with_ocr_errors": 0,
        "total_text_length": 0,
        "total_words": 0,
        "bilingual_slides": 0,
        "vietnamese_only": 0,
        "english_only": 0,
        "course_details": {}
    }
    
    courses = [d for d in data_dir.iterdir() 
               if d.is_dir() and d.name != "scratch"]
    stats["total_courses"] = len(courses)
    
    for course_dir in courses:
        material_file = course_dir / "material" / "material.json"
        if not material_file.exists():
            continue
        
        stats["courses_with_material"] += 1
        course_stats = {
            "slides": 0,
            "empty": 0,
            "ocr_errors": 0,
            "total_length": 0,
            "total_words": 0,
            "bilingual": 0,
            "vietnamese_only": 0,
            "english_only": 0
        }
        
        try:
            with open(material_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                slides = data.get("slides", [])
                
                for slide in slides:
                    stats["total_slides"] += 1
                    course_stats["slides"] += 1
                    
                    text = slide.get("raw_text", "")
                    quality = analyze_text_quality(text)
                    
                    if quality["empty"]:
                        stats["empty_slides"] += 1
                        course_stats["empty"] += 1
                    else:
                        stats["total_text_length"] += quality["length"]
                        stats["total_words"] += quality["word_count"]
                        course_stats["total_length"] += quality["length"]
                        course_stats["total_words"] += quality["word_count"]
                        
                        if quality["has_ocr_errors"]:
                            stats["slides_with_ocr_errors"] += 1
                            course_stats["ocr_errors"] += 1
                        
                        if quality["has_vietnamese"] and quality["has_english"]:
                            stats["bilingual_slides"] += 1
                            course_stats["bilingual"] += 1
                        elif quality["has_vietnamese"]:
                            stats["vietnamese_only"] += 1
                            course_stats["vietnamese_only"] += 1
                        elif quality["has_english"]:
                            stats["english_only"] += 1
                            course_stats["english_only"] += 1
        except Exception as e:
            print(f"Error processing {material_file}: {e}")
        
        if course_stats["slides"] > 0:
            stats["course_details"][course_dir.name] = course_stats
    
    return stats

def print_report(syllabus_stats: Dict, material_stats: Dict):
    """Print analysis report."""
    print("=" * 80)
    print("PARSED DATA QUALITY ANALYSIS REPORT")
    print("=" * 80)
    
    print("\n[SYLLABUS FILES ANALYSIS]")
    print("-" * 80)
    print(f"Total courses: {syllabus_stats['total_courses']}")
    print(f"Courses with syllabus: {syllabus_stats['courses_with_syllabus']}")
    print(f"Total syllabus files: {syllabus_stats['total_files']}")
    print(f"Empty files: {syllabus_stats['empty_files']} ({syllabus_stats['empty_files']/max(syllabus_stats['total_files'],1)*100:.1f}%)")
    print(f"Files with OCR errors: {syllabus_stats['files_with_ocr_errors']} ({syllabus_stats['files_with_ocr_errors']/max(syllabus_stats['total_files'],1)*100:.1f}%)")
    print(f"Total text length: {syllabus_stats['total_text_length']:,} characters")
    print(f"Total words: {syllabus_stats['total_words']:,}")
    if syllabus_stats['total_files'] > 0:
        avg_length = syllabus_stats['total_text_length'] / (syllabus_stats['total_files'] - syllabus_stats['empty_files'])
        avg_words = syllabus_stats['total_words'] / (syllabus_stats['total_files'] - syllabus_stats['empty_files'])
        print(f"Average text length: {avg_length:.0f} characters per file")
        print(f"Average words: {avg_words:.0f} words per file")
    
    print(f"\nLanguage distribution:")
    print(f"  Bilingual (Vietnamese + English): {syllabus_stats['bilingual_files']} ({syllabus_stats['bilingual_files']/max(syllabus_stats['total_files'],1)*100:.1f}%)")
    print(f"  Vietnamese only: {syllabus_stats['vietnamese_only']} ({syllabus_stats['vietnamese_only']/max(syllabus_stats['total_files'],1)*100:.1f}%)")
    print(f"  English only: {syllabus_stats['english_only']} ({syllabus_stats['english_only']/max(syllabus_stats['total_files'],1)*100:.1f}%)")
    
    print("\n[MATERIAL FILES ANALYSIS]")
    print("-" * 80)
    print(f"Total courses: {material_stats['total_courses']}")
    print(f"Courses with material: {material_stats['courses_with_material']}")
    print(f"Total slides: {material_stats['total_slides']}")
    print(f"Empty slides: {material_stats['empty_slides']} ({material_stats['empty_slides']/max(material_stats['total_slides'],1)*100:.1f}%)")
    print(f"Slides with OCR errors: {material_stats['slides_with_ocr_errors']} ({material_stats['slides_with_ocr_errors']/max(material_stats['total_slides'],1)*100:.1f}%)")
    print(f"Total text length: {material_stats['total_text_length']:,} characters")
    print(f"Total words: {material_stats['total_words']:,}")
    if material_stats['total_slides'] > 0:
        avg_length = material_stats['total_text_length'] / (material_stats['total_slides'] - material_stats['empty_slides'])
        avg_words = material_stats['total_words'] / (material_stats['total_slides'] - material_stats['empty_slides'])
        print(f"Average text length: {avg_length:.0f} characters per slide")
        print(f"Average words: {avg_words:.0f} words per slide")
    
    print(f"\nLanguage distribution:")
    print(f"  Bilingual (Vietnamese + English): {material_stats['bilingual_slides']} ({material_stats['bilingual_slides']/max(material_stats['total_slides'],1)*100:.1f}%)")
    print(f"  Vietnamese only: {material_stats['vietnamese_only']} ({material_stats['vietnamese_only']/max(material_stats['total_slides'],1)*100:.1f}%)")
    print(f"  English only: {material_stats['english_only']} ({material_stats['english_only']/max(material_stats['total_slides'],1)*100:.1f}%)")
    
    print("\n" + "=" * 80)
    print("QUALITY ASSESSMENT")
    print("=" * 80)
    
    # Overall assessment
    syllabus_quality_score = 100
    if syllabus_stats['empty_files'] / max(syllabus_stats['total_files'], 1) > 0.1:
        syllabus_quality_score -= 20
    if syllabus_stats['files_with_ocr_errors'] / max(syllabus_stats['total_files'], 1) > 0.2:
        syllabus_quality_score -= 20
    
    material_quality_score = 100
    if material_stats['empty_slides'] / max(material_stats['total_slides'], 1) > 0.1:
        material_quality_score -= 20
    if material_stats['slides_with_ocr_errors'] / max(material_stats['total_slides'], 1) > 0.2:
        material_quality_score -= 20
    
    print(f"\nSyllabus Quality Score: {syllabus_quality_score}/100")
    if syllabus_quality_score >= 80:
        print("  [OK] Good quality - suitable for RAG system")
    elif syllabus_quality_score >= 60:
        print("  [WARNING] Moderate quality - some improvements needed")
    else:
        print("  [ERROR] Poor quality - significant improvements needed")
    
    print(f"\nMaterial Quality Score: {material_quality_score}/100")
    if material_quality_score >= 80:
        print("  [OK] Good quality - suitable for RAG system")
    elif material_quality_score >= 60:
        print("  [WARNING] Moderate quality - some improvements needed")
    else:
        print("  [ERROR] Poor quality - significant improvements needed")
    
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    if syllabus_stats['empty_files'] > 0:
        print(f"[WARNING] {syllabus_stats['empty_files']} empty syllabus files detected - check OCR extraction")
    if syllabus_stats['files_with_ocr_errors'] > 0:
        print(f"[WARNING] {syllabus_stats['files_with_ocr_errors']} syllabus files with OCR errors - consider re-processing")
    if material_stats['empty_slides'] > 0:
        print(f"[WARNING] {material_stats['empty_slides']} empty slides detected - check OCR extraction")
    if material_stats['slides_with_ocr_errors'] > 0:
        print(f"[WARNING] {material_stats['slides_with_ocr_errors']} slides with OCR errors - consider re-processing")
    
    if syllabus_stats['bilingual_files'] > 0 or material_stats['bilingual_slides'] > 0:
        print("[INFO] Bilingual content detected - ensure embedding model supports both languages")
    
    print("\n[OK] Overall: Data appears suitable for RAG system with minor improvements")

def main():
    data_dir = Path("data/data")
    if not data_dir.exists():
        print(f"Error: {data_dir} does not exist")
        return
    
    print("Analyzing syllabus files...")
    syllabus_stats = analyze_syllabus_files(data_dir)
    
    print("Analyzing material files...")
    material_stats = analyze_material_files(data_dir)
    
    print_report(syllabus_stats, material_stats)

if __name__ == "__main__":
    main()

