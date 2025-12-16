#!/usr/bin/env python3
"""
Script to collect dataset statistics for the experiments section.

This script scans the data directory and collects statistics about:
- Number of courses
- Number of documents (syllabus + materials)
- Number of chunks (from indices)
- Approximate token counts

Usage:
    python scripts/collect_dataset_stats.py [--data-root ./data] [--index-dir ./data/indices]
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
import sys


print(Path(__file__).parent.parent)
sys.path.append(str(Path(__file__).parent.parent))


def count_courses(data_root: Path) -> int:
    """Count number of courses in data directory."""
    if not data_root.exists():
        return 0
    
    # Count directories that are not special directories
    special_dirs = {"scratch", "converted", "data", "indices", "processed", "raw"}
    course_dirs = [
        d for d in data_root.iterdir() 
        if d.is_dir() and d.name not in special_dirs
    ]
    return len(course_dirs)


def count_documents(data_root: Path) -> Tuple[int, int, int]:
    """
    Count syllabus and material documents.
    
    Returns:
        (total_docs, syllabus_count, material_count)
    """
    if not data_root.exists():
        return 0, 0, 0
    
    special_dirs = {"scratch", "converted", "data", "indices", "processed", "raw"}
    course_dirs = [
        d for d in data_root.iterdir() 
        if d.is_dir() and d.name not in special_dirs
    ]
    
    total_docs = 0
    syllabus_count = 0
    material_count = 0
    
    for course_dir in course_dirs:
        course_name = course_dir.name
        
        # Count syllabus files
        syllabus_dir = data_root / course_name / "syllabus" / "parsed"
        if syllabus_dir.exists():
            syllabus_files = list(syllabus_dir.glob("*.syllabus.json"))
            syllabus_count += len(syllabus_files)
            total_docs += len(syllabus_files)
        
        # Count material files
        material_file = data_root / course_name / "material" / "material.json"
        if material_file.exists():
            material_count += 1
            total_docs += 1
    
    return total_docs, syllabus_count, material_count


def count_chunks_from_indices(index_dir: Path) -> Tuple[int, Dict[str, int]]:
    """
    Count total chunks and chunks per course from metadata files.
    
    Returns:
        (total_chunks, {course_name: chunk_count})
    """
    if not index_dir.exists():
        return 0, {}
    
    total_chunks = 0
    chunks_per_course = {}
    
    # Find all course index directories
    for course_index_dir in index_dir.iterdir():
        if not course_index_dir.is_dir():
            continue
        
        course_name = course_index_dir.name
        metadata_file = course_index_dir / "metadata.json"
        
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                # Count chunks in metadata
                chunk_count = len(metadata) if isinstance(metadata, dict) else 0
                chunks_per_course[course_name] = chunk_count
                total_chunks += chunk_count
            except Exception as e:
                print(f"Warning: Failed to read {metadata_file}: {e}", file=sys.stderr)
    
    return total_chunks, chunks_per_course


def estimate_tokens(chunks_per_course: Dict[str, int], chunk_size: int = 512) -> int:
    """
    Estimate total tokens based on chunk count.
    
    This is a rough estimate assuming average chunk is ~80% full.
    """
    total_chunks = sum(chunks_per_course.values())
    # Assume average chunk uses 80% of chunk_size
    avg_tokens_per_chunk = int(chunk_size * 0.8)
    return total_chunks * avg_tokens_per_chunk


def collect_all_stats(data_root: Path, index_dir: Path) -> Dict:
    """Collect all statistics."""
    print(f"Scanning data directory: {data_root}")
    print(f"Scanning index directory: {index_dir}")
    
    # Count courses
    num_courses = count_courses(data_root)
    print(f"Found {num_courses} courses")
    
    # Count documents
    total_docs, syllabus_count, material_count = count_documents(data_root)
    print(f"Found {total_docs} documents ({syllabus_count} syllabus, {material_count} materials)")
    
    # Count chunks
    total_chunks, chunks_per_course = count_chunks_from_indices(index_dir)
    print(f"Found {total_chunks} chunks across {len(chunks_per_course)} indexed courses")
    
    # Calculate average chunks per course
    avg_chunks_per_course = total_chunks / num_courses if num_courses > 0 else 0
    
    # Estimate tokens
    estimated_tokens = estimate_tokens(chunks_per_course)
    
    stats = {
        "num_courses": num_courses,
        "total_documents": total_docs,
        "syllabus_documents": syllabus_count,
        "material_documents": material_count,
        "total_chunks": total_chunks,
        "avg_chunks_per_course": round(avg_chunks_per_course, 1),
        "estimated_tokens": estimated_tokens,
        "chunks_per_course": chunks_per_course
    }
    
    return stats


def print_latex_table(stats: Dict):
    """Print LaTeX table format."""
    print("\n" + "="*60)
    print("LaTeX Table Format:")
    print("="*60)
    print()
    print("\\begin{table}[hbt]")
    print("\\centering")
    print("\\caption{Dataset Statistics}")
    print("\\label{tab:dataset_stats}")
    print("\\begin{tabular}{|l|c|}")
    print("\\hline")
    print("\\textbf{Statistic} & \\textbf{Value} \\\\")
    print("\\hline")
    print(f"Number of Courses & {stats['num_courses']} \\\\")
    print("\\hline")
    print(f"Total Documents & {stats['total_documents']} \\\\")
    print("\\hline")
    print(f"Syllabus Documents & {stats['syllabus_documents']} \\\\")
    print("\\hline")
    print(f"Material Documents (Slides) & {stats['material_documents']} \\\\")
    print("\\hline")
    print(f"Total Text Chunks & {stats['total_chunks']} \\\\")
    print("\\hline")
    print(f"Average Chunks per Course & {stats['avg_chunks_per_course']} \\\\")
    print("\\hline")
    print(f"Total Tokens (approximate) & {stats['estimated_tokens']:,} \\\\")
    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")
    print()


def print_json_output(stats: Dict):
    """Print JSON format for easy parsing."""
    print("\n" + "="*60)
    print("JSON Output:")
    print("="*60)
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Collect dataset statistics for experiments section"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(Path(__file__).parent.parent / "data" / "data"),
        help="Root directory containing course data (default: data/data)"
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default=str(Path(__file__).parent.parent / "data" / "indices"),
        help="Directory containing FAISS indices (default: data/indices)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["latex", "json", "both"],
        default="both",
        help="Output format: latex, json, or both (default: both)"
    )
    
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    index_dir = Path(args.index_dir)
    
    if not data_root.exists():
        print(f"Error: Data root directory does not exist: {data_root}", file=sys.stderr)
        sys.exit(1)
    
    if not index_dir.exists():
        print(f"Warning: Index directory does not exist: {index_dir}", file=sys.stderr)
        print("Chunk statistics will be 0.", file=sys.stderr)
    
    stats = collect_all_stats(data_root, index_dir)
    
    if args.format in ["latex", "both"]:
        print_latex_table(stats)
    
    if args.format in ["json", "both"]:
        print_json_output(stats)
    
    # Print summary
    print("\n" + "="*60)
    print("Summary:")
    print("="*60)
    print(f"  Courses: {stats['num_courses']}")
    print(f"  Documents: {stats['total_documents']} ({stats['syllabus_documents']} syllabus, {stats['material_documents']} materials)")
    print(f"  Chunks: {stats['total_chunks']}")
    print(f"  Avg Chunks/Course: {stats['avg_chunks_per_course']}")
    print(f"  Estimated Tokens: {stats['estimated_tokens']:,}")
    print()


if __name__ == "__main__":
    main()

