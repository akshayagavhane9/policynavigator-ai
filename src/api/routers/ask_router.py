from fastapi import APIRouter, HTTPException
from typing import List
import json
import re

from src.api.models.ask_models import (
    AskRequest,
    AskResponse,
    ScenarioRequest,
    ScenarioResponse,
    QuizRequest,
    QuizResponse,
    QuizItem,
)
from src.main import answer_question
from src.llm.client import LLMClient


router = APIRouter(prefix="/api", tags=["qa"])


# -------------------------------------------------------------------
# Helper: extract JSON array from LLM output (same logic as Streamlit)
# -------------------------------------------------------------------
def extract_json_array(raw: str) -> list:
    """
    Extract a JSON array from an LLM response that may contain extra text.
    Looks for the first '[' and the last ']' and tries to json.loads that slice.
    """
    if not raw or not raw.strip():
        raise ValueError("Model returned an empty response.")

    match = re.search(r"\[.*\]", raw, re.S)
    if not match:
        raise ValueError("Model did not return a JSON array.")
    json_str = match.group(0)
    return json.loads(json_str)


# -------------------------------------------------------------------
# /api/ask – main policy Q&A endpoint
# -------------------------------------------------------------------
@router.post("/ask", response_model=AskResponse)
async def ask_policy(request: AskRequest) -> AskResponse:
    try:
        res = answer_question(
            question=request.question,
            answer_style=request.answer_style,
            rewrite_query=request.rewrite_query,
            k=request.k,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return AskResponse(**res)


# -------------------------------------------------------------------
# /api/scenario – What-If scenario analysis
# -------------------------------------------------------------------
@router.post("/scenario", response_model=ScenarioResponse)
async def analyze_scenario(req: ScenarioRequest) -> ScenarioResponse:
    if not req.scenario.strip():
        raise HTTPException(status_code=400, detail="Scenario text is empty.")

    llm = LLMClient()
    system_prompt = (
        "You are PolicyNavigator AI. You reason about university policies, "
        "academic integrity, and student conduct in a careful, non-judgmental way. "
        "You do not invent specific NEU rules; instead you explain typical outcomes, "
        "risks, and next steps a student should take."
    )
    user_prompt = f"""
Here is a student's scenario:

\"\"\"{req.scenario}\"\"\"

Using general university policy principles, explain:

1. Which kinds of policies are probably relevant.
2. What risks or consequences might apply.
3. What steps the student should take next (e.g., who to contact, how to document things).
4. A short, encouraging closing note.
"""

    try:
        analysis = llm.chat(system_prompt, user_prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return ScenarioResponse(analysis=analysis)


# -------------------------------------------------------------------
# /api/quiz – generate multiple-choice quiz
# -------------------------------------------------------------------
@router.post("/quiz", response_model=QuizResponse)
async def generate_quiz(req: QuizRequest) -> QuizResponse:
    n = max(1, min(req.num_questions, 10))

    llm = LLMClient()
    system_prompt = (
        "You are an expert tutor on university academic integrity and "
        "student conduct policies. You generate short multiple-choice quizzes "
        "to help students understand key rules and consequences."
    )

    user_prompt = f"""
Generate {n} multiple-choice quiz questions about university academic integrity
and student conduct policies (cheating, plagiarism, collaboration rules, sanctions, appeals).

Return your answer ONLY as a JSON array with this exact schema:

[
  {{
    "question": "string",
    "options": ["option A", "option B", "option C", "option D"],
    "answer": "the correct option text (must match one element of 'options')",
    "explanation": "short explanation of why this is the correct answer"
  }},
  ...
]

Do not include any keys other than question, options, answer, explanation.
If unsure of a detail, prefer a more general question instead of inventing a specific rule.
"""

    try:
        raw = llm.run(system_prompt, user_prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    try:
        items_raw = extract_json_array(raw)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse quiz JSON: {e}")

    quiz_items: List[QuizItem] = []
    for item in items_raw:
        try:
            quiz_items.append(QuizItem(**item))
        except Exception:
            # skip malformed items
            continue

    return QuizResponse(quiz=quiz_items)
