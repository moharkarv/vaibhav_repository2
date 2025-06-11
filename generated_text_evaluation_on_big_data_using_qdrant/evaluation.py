# evaluation.py

import google.generativeai as genai
import re
import json

# Configure Gemini API
genai.configure(api_key="")  # 🔁 Replace with your actual API key

llmmodel = genai.GenerativeModel("gemini-1.5-flash")

def evaluate_answer(
    query: str,
    generated_text: str,
    context_docs: list,
    context_input: str = "",
    reference_input: str = ""
) -> dict:
    """
    Evaluate generated_text for grammar, coherence, factuality (vs context), and style (vs reference).
    """
    # Combine context_docs + optional user-provided context
    #context_text = "\n\n".join([doc.page_content for doc in context_docs]) if context_docs else ""
    context_text=context_docs
    if context_input:
        context_text += f"\n\nAdditional user context:\n{context_input}"

    reference_input=reference_input+context_input

    # Build prompt with clearly separated sections
    prompt = f"""
You are a helpful assistant evaluating a generated answer along four quality dimensions: grammar, coherence, factual accuracy, and style.

Use the following rules:
- Grammar: Analyze only the generated answer.
- Coherence: Analyze only the generated answer.
- Factual accuracy: Compare the generated answer against the context section.
- Style: Compare the generated answer against the reference style section.

Query:
{query}

Context (for factual accuracy check):
{context_text}

Reference style guide (for tone/style comparison):
{reference_input}

Generated answer:
{generated_text}

Return your evaluation as JSON with this format:
{{
  "composite_score": <average of all scores>,
  "grammar": {{"score": 0-100, "notes": "..." }},
  "coherence": {{"score": 0-100, "notes": "..." }},
  "factual_accuracy": {{"score": 0-100, "notes": "..." }},
  "style": {{"score": 0-100, "notes": "..." }},
  "suggestions": ["list of concrete suggestions to improve the answer"]
}}
"""

    # Call Gemini
    response = llmmodel.generate_content(prompt)
    raw = response.candidates[0].content.parts[0].text

    # Try to extract JSON
    try:
        json_text_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_text_match:
            return json.loads(json_text_match.group())
        else:
            return {"error": "No JSON found in LLM response", "raw_response": raw}
    except json.JSONDecodeError as e:
        return {"error": f"JSON decode error: {e}", "raw_response": raw}
