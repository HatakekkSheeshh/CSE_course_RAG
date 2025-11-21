from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from rag.query_pipeline import QueryPipeline, NO_INFO_MESSAGE  # noqa: E402
from rag.llm_client import LLMClient  # noqa: E402



def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Query the CSE RAG indices with reranking.")
    parser.add_argument("--question", required=True, help="User question to answer")
    parser.add_argument("--data-dir", default="data", help="Base data directory")
    parser.add_argument("--index-dir", default="data/indices", help="FAISS index directory")
    parser.add_argument("--course", help="Limit to a single course folder (optional)")
    parser.add_argument("--retrieval-k", type=int, default=8, help="Top K chunks per course to retrieve")
    parser.add_argument("--rerank-k", type=int, default=5, help="Top K chunks to rerank")
    parser.add_argument("--confidence-threshold", type=float, default=0.1, help="Confidence needed to call LLM")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM call, only show context")
    parser.add_argument("--dump-json", action="store_true", help="Print raw pipeline output as JSON")
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    pipeline = QueryPipeline(
        data_dir=Path(args.data_dir),
        index_dir=Path(args.index_dir),
        retrieval_k=args.retrieval_k,
        rerank_k=args.rerank_k,
        confidence_threshold=args.confidence_threshold,
        only_course=args.course,
    )

    result = pipeline.answer(args.question, course=args.course)

    if args.dump_json:
        print(json.dumps(result, default=lambda o: o.__dict__, indent=2))

    if result.get("status") != "ok":
        reason = result.get("reason", "unknown")
        print(f"{NO_INFO_MESSAGE} (reason: {reason})")
        return

    best = result["best_chunk"]
    confidence = result.get("confidence", 0.0)
    print(f"Top chunk: {best.chunk_id} | confidence={confidence:.2%}")
    if best.metadata:
        print(f"Metadata: {best.metadata}")
    print("\nContext Preview:\n")
    print(best.text.strip())
    print("\n---")

    if args.no_llm:
        print("LLM skipped (--no-llm).")
        return

    llm = LLMClient()
    if not llm.enabled:
        print("LLM disabled (missing OPENAI_API_KEY or dependency). Set --no-llm to skip.")
        return

    contexts = [best.text] + [chunk.text for chunk in result.get("reranked", [])[1:3]]
    answer = llm.generate_answer(args.question, contexts)
    print("\nAnswer:\n")
    print(answer)


if __name__ == "__main__":
    main()

