import json
import requests
from typing import TypedDict
from langgraph.graph import StateGraph, END

OLLAMA_URL = "http://localhost:11434"  

# ----------------------------------
# Safe JSON parser for LLM responses
# ----------------------------------
def safe_parse_json(text: str):
    """Try to parse JSON even if extra newlines or text exist"""
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except Exception:
                continue
    return {}

# ----------------------------------
# Direct Ollama API call
# ----------------------------------
def call_ollama(prompt: str) -> str:
    response = requests.post(
        f"{OLLAMA_URL}/api/chat",  
        json={
            "model": "llama3.1",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1
        }
    )
    response.raise_for_status()
    data = response.json()
    content = data["choices"][0]["message"]["content"]
    return content.strip()

# ----------------------------------
# Load patterns from JSON
# ----------------------------------
def load_patterns(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ----------------------------------
# Define state schema
# ----------------------------------
class PatternState(TypedDict):
    pattern_title: str
    description: str
    resolution_notes: list

# ----------------------------------
# Build LangGraph state graph
# ----------------------------------
def build_pattern_graph(patterns):
    states = [
        {
            "pattern_title": pattern["pattern_title"],
            "description": f"Pattern: {pattern['pattern_title']}",
            "resolution_notes": pattern.get("resolution_notes", [])
        }
        for pattern in patterns
    ]

    graph = StateGraph(
        states=states,
        state_schema=PatternState,
        end_state=END
    )
    return graph

# ----------------------------------
# Match incident with patterns using LLM
# ----------------------------------
def match_incident_with_pattern(incident, patterns, graph):
    patterns_text = json.dumps(patterns, indent=2)

    prompt = f"""
You are an incident classification engine.

Incident:
Title: {incident['title']}
Description: {incident['short_description']}
Category: {incident.get('category', '')}

Available patterns (JSON):
{patterns_text}

Task:
- Understand the incident semantically.
- Compare it with ALL patterns.
- Select ONLY ONE best matching pattern.
- Treat synonyms as same meaning (e.g., permission denied = access denied).
- If none match, respond "NO_MATCH".

Respond ONLY in JSON format:
{{ "pattern_title": "<pattern_title or NO_MATCH>" }}
"""

    raw_response = call_ollama(prompt)
    result = safe_parse_json(raw_response)
    selected_pattern = result.get("pattern_title")

    
    if selected_pattern and any(s["pattern_title"] == selected_pattern for s in graph.states):
        return selected_pattern
    return "NO_MATCH"

# ----------------------------------
# Suggest resolution notes for matched pattern
# ----------------------------------
def get_resolution_notes(pattern_title, graph):
    for state in graph.states:
        if state["pattern_title"] == pattern_title:
            return state.get("resolution_notes", [])
    return []

# ----------------------------------
# Interactive runner
# ----------------------------------
if __name__ == "__main__":
    patterns = load_patterns("incident_patterns_final.json")
    graph = build_pattern_graph(patterns)

    while True:
        print("\n Enter New Incident Details")
        incident_id = input("Incident ID: ").strip()
        title = input("Title: ").strip()
        short_description = input("Short Description: ").strip()
        category = input("Category: ").strip()

        incident = {
            "incident_id": incident_id,
            "title": title,
            "short_description": short_description,
            "category": category
        }

        matched_pattern = match_incident_with_pattern(incident, patterns, graph)

        if matched_pattern and matched_pattern != "NO_MATCH":
            print("\n PRECISE MATCH FOUND\n")
            print(f"Pattern Title: {matched_pattern}")
            notes = get_resolution_notes(matched_pattern, graph)
            if notes:
                print("\n Suggested Resolution Notes:")
                for i, note in enumerate(notes, 1):
                    print(f"{i}. {note}")
        else:
            print("\n No precise pattern match found.")

        cont = input("\nDo you want to enter another incident? (y/n): ").strip().lower()
        if cont != 'y':
            break
