import json
from typing import Dict, Any

from src.llm.prompt_builder import PromptBuilder
from src.llm.client import LLMClient
from src.rag.pipeline import retrieve_context

prompt_builder = PromptBuilder()
llm_client = LLMClient()


def _safe_json_parse(response_text: str) -> Dict[str, Any]:
    """
    Try to parse the LLM response as JSON.
    If it fails, fall back to a simple dict with raw text.
    """
    try:
        data = json.loads(response_text)
        # Basic validation
        if not isinstance(data, dict):
            raise ValueError("Parsed JSON is not a dict")
        return data
    except Exception:
        # Fallback: wrap raw text
        return {
            "answer": response_text,
            "citations": [],
            "confidence": "unknown",
        }


def answer_question(
    question: str,
    use_rewrite: bool = True,
    top_k: int = 5,
    collection_name: str = "policies",
) -> Dict[str, Any]:
    """
    High-level QA function:
    1. Optionally rewrite the query for better retrieval.
    2. Retrieve context with RAG.
    3. Build answer prompt.
    4. Call LLM and parse JSON.

    Returns a dict:
    {
      "answer": str,
      "citations": List[str],
      "confidence": str
    }
    """
    original_question = question.strip()
    if not original_question:
        return {
            "answer": "Please provide a non-empty question.",
            "citations": [],
            "confidence": "low",
        }

    # 1) Optional query rewrite
    search_question = original_question
    if use_rewrite:
        rewrite_prompt = prompt_builder.build_rewrite_prompt(original_question)
        rewritten = llm_client.run(rewrite_prompt).strip()
        if rewritten:
            search_question = rewritten

    # 2) Retrieve context via RAG
    context = retrieve_context(
        question=search_question,
        top_k=top_k,
        collection_name=collection_name,
    )

    if not context.strip():
        # No context retrieved – be honest, no hallucinations
        return {
            "answer": "I couldn't find relevant information for this question in the indexed documents.",
            "citations": [],
            "confidence": "low",
        }

    # 3) Build answer prompt
    answer_prompt = prompt_builder.build_answer_prompt(
        question=original_question,
        context=context,
    )

    # 4) Call LLM
    raw_response = llm_client.run(answer_prompt)

    # 5) Parse JSON safely
    parsed = _safe_json_parse(raw_response)

    # Ensure some fields exist
    parsed.setdefault("answer", "")
    parsed.setdefault("citations", [])
    parsed.setdefault("confidence", "unknown")

    return parsed


if __name__ == "__main__":
    # Simple manual test (assumes you've already indexed something)
    q = "What is the late submission policy?"
    result = answer_question(q)
    print("Q:", q)
    print("Answer:", result["answer"])
    print("Citations:", result["citations"])
    print("Confidence:", result["confidence"])
