import json
import re
from langchain_ollama import ChatOllama

# Configuration
PATTERN_FILE = "incident_patterns_top_15.json"
MODEL_NAME = "llama3.1"

llm = ChatOllama(
    model=MODEL_NAME,
    temperature=0.0,
    base_url="http://localhost:11434"
)

# Helpers
def load_patterns():
    with open(PATTERN_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_json(text: str):
    """Robust JSON extractor"""
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        return json.loads(match.group())
    raise ValueError("Invalid JSON from LLM")


# LLM Pattern Matcher
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
Sub Category: {incident['sub_category']}

AVAILABLE PATTERNS:
{pattern_text}
"""

    response = llm.invoke(prompt)
    return extract_json(response.content)


# Test Incidents
test_incidents = [
    {
        "title": "User unable to login to banking portal",
        "description": "A customer is unable to access their account in the banking application",
        "category": "Banking",
        "sub_category": "Authentication"
    },
    {
        "title": "Server not responding to payment requests",
        "description": "The payment processing server is not responding to API calls, causing transaction failures",
        "category": "Banking",
        "sub_category": "Infrastructure"
    },
    {
        "title": "Customer requests password reset",
        "description": "User forgot password and needs assistance with reset",
        "category": "Banking",
        "sub_category": "Account Management"
    },
    {
        "title": "API error in transaction processing",
        "description": "The transaction API is returning 500 errors",
        "category": "Banking",
        "sub_category": "API"
    },
    {
        "title": "Transaction failed without explanation",
        "description": "Customer's payment transaction was declined",
        "category": "Banking",
        "sub_category": "Payments"
    }
]

# Main
if __name__ == "__main__":
    print("\n" + "="*80)
    print("INCIDENT PATTERN MATCHER - TEST RUNNER")
    print("="*80 + "\n")
    
    patterns = load_patterns()
    
    for idx, incident in enumerate(test_incidents, 1):
        print(f"\n{'─'*80}")
        print(f"TEST CASE #{idx}")
        print(f"{'─'*80}")
        print(f"Title: {incident['title']}")
        print(f"Description: {incident['description']}")
        print(f"Category: {incident['category']}")
        print(f"Sub Category: {incident['sub_category']}")
        
        try:
            print("\nMatching incident using LLM...")
            result = match_incident_with_llm(incident, patterns)
            
            print("\n✓ MATCH RESULT:")
            print(json.dumps(result, indent=2))
        except Exception as e:
            print(f"\n✗ ERROR: {str(e)}")
    
    print(f"\n{'='*80}")
    print("TEST COMPLETED")
    print(f"{'='*80}\n")
