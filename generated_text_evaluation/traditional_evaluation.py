# traditional_evaluation.py

from transformers import pipeline
from bert_score import score as bert_score
import language_tool_python
import textstat

# Load models/tools globally for efficiency
nli_pipeline = pipeline(
    "zero-shot-classification",
    model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
)
grammar_tool = language_tool_python.LanguageTool('en-US')

def evaluate_generated_text(generated_text: str, context_docs: str, grammar_penalty_per_error: int = 2) -> dict:
    """
    Evaluates a generated answer against a context (e.g., retrieved via RAG).
    Returns grammar evaluation, factual accuracy, and style metrics.
    """

    results = {}

    # --- 1. Grammar Evaluation ---
    grammar_matches = grammar_tool.check(generated_text)
    num_errors = len(grammar_matches)
    grammar_score = max(100 - num_errors * grammar_penalty_per_error, 0)
    grammar_suggestions = [m.message for m in grammar_matches[:3]]

    results["grammar"] = {
        "score": grammar_score,
        "errors": num_errors,
        "suggestions": grammar_suggestions
    }

    # --- 2. Factual Accuracy (NLI) ---
    try:
        nli_result = nli_pipeline(context_docs, candidate_labels=[generated_text])
        factual_confidence = round(nli_result['scores'][0] * 100, 2)
    except Exception as e:
        factual_confidence = 0.0

    results["factual_accuracy_based_on_context"] = {
        "confidence": factual_confidence
    }

    # --- 3. Style Metrics ---
    results["style"] = {
        "flesch_reading_ease": textstat.flesch_reading_ease(generated_text),
        "gunning_fog": textstat.gunning_fog(generated_text),
        "smog_index": textstat.smog_index(generated_text)
    }

    return results
