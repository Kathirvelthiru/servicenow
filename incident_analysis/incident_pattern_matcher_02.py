import json
import re
from langchain_ollama import ChatOllama

# -----------------------------
# CONFIGF
# -----------------------------
PATTERN_FILE = "incident_patterns_top_15.json"
MODEL_NAME = "llama3.1"

llm = ChatOllama(
    model=MODEL_NAME,
    temperature=0.0,
    base_url="http://localhost:11434"
)

# -----------------------------
# Helpers
# -----------------------------
def load_patterns():
    with open(PATTERN_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_json(text: str):
    """Robust JSON extractor"""
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        return json.loads(match.group())
    raise ValueError("Invalid JSON from LLM")


# -----------------------------
# LLM Pattern Matcher
# -----------------------------
def match_incident_with_llm(incident, patterns):
    pattern_text = json.dumps(patterns, indent=2)

    prompt = f"""
You are a SENIOR ITSM INCIDENT CLASSIFICATION ENGINE.

TASK:
Given an incident and a list of predefined incident patterns,
identify the SINGLE BEST matching pattern.

RULES:
- Choose ONLY one pattern
- Match based on semantic meaning, not keywords only
- Avoid generic matches if a specific one exists
- If nothing matches well, return "No Match"

OUTPUT FORMAT (STRICT JSON ONLY):

{{
  "matched_pattern": "<pattern_name or No Match>",
  "confidence": "<High | Medium | Low>",
  "reason": "Short justification"
}}

INCIDENT:
Title: {incident['title']}
Description: {incident['description']}
Category: {incident['category']}
Sub Category: {incident['sub_category']}`

AVAILABLE PATTERNS:
{pattern_text}
"""

    response = llm.invoke(prompt)
    return extract_json(response.content)


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    print("\n Enter Incident Details")

    incident = {
        "title": input("Title: "),
        "description": input("Description: "),
        "category": input("Category: "),
        "sub_category": input("Sub Category: ")
    }

    patterns = load_patterns()

    print("\n Matching incident using LLM...\n")
    result = match_incident_with_llm(incident, patterns)

    print(" MATCH RESULT")
    print(json.dumps(result, indent=2))
