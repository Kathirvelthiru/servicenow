import json
import re
import pandas as pd
from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from collections import defaultdict

# =====================================================
# CONFIGURATION
# =====================================================

DATASET_FILE = "dataset_4500_similar.xlsx"
OUTPUT_FILE = "incident_patterns_top_15.json"
MODEL_NAME = "llama3.1:latest"
CHUNK_SIZE = 50           
TOP_PATTERN_COUNT = 15    
RETRY_LIMIT = 2           

# =====================================================
# STATE DEFINITION
# =====================================================

class PatternState(TypedDict):
    incidents: List[Dict]
    patterns: List[Dict]

# =====================================================
# LLM INITIALIZATION
# =====================================================

llm = ChatOllama(
    model=MODEL_NAME,
    temperature=0.1,
    base_url="http://localhost:11434"
)

# =====================================================
# ROBUST JSON EXTRACTOR
# =====================================================

def extract_patterns(text: str) -> List[Dict]:
    """Extract JSON array from LLaMA output even if wrapped in text or markdown."""
    text = text.strip()
    text = re.sub(r"```json|```", "", text, flags=re.IGNORECASE).strip()

    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            data = json.loads(text[start:end+1])
            if isinstance(data, list):
                return data
        except:
            pass

    
    objects = []
    matches = re.findall(r"\{[\s\S]*?\}", text)
    for m in matches:
        try:
            obj = json.loads(m)
            if "pattern_name" in obj:
                objects.append(obj)
        except:
            continue
    return objects

# =====================================================
# NODE 1 — LOAD DATA
# =====================================================

def load_data(state: PatternState):
    df = pd.read_excel(DATASET_FILE).fillna("")
    incidents = df.to_dict(orient="records")
    print(f"✅ Total Incidents Loaded: {len(incidents)}")
    state["incidents"] = incidents
    return state

# =====================================================
# NODE 2 — GENERATE PATTERNS PER CHUNK
# =====================================================

def generate_patterns(state: PatternState):
    print(f" Generating patterns in chunks of {CHUNK_SIZE} incidents...")

    all_patterns = []

    
    incidents = state["incidents"]
    for idx in range(0, len(incidents), CHUNK_SIZE):
        chunk = incidents[idx:idx + CHUNK_SIZE]
        print(f" Processing chunk {idx//CHUNK_SIZE + 1}/{(len(incidents)//CHUNK_SIZE)+1} (size={len(chunk)})")

       
        lines = []
        for i, inc in enumerate(chunk, 1):
            lines.append(
                f"{i}. Title: {inc.get('Title','')} | "
                f"Description: {inc.get('Short Description','')} | "
                f"Category: {inc.get('Category','')} | "
                f"Sub Category: {inc.get('Sub Category','')}"
            )

        prompt = f"""
You are a SENIOR ITSM INCIDENT PATTERN ENGINE.

TASK:
From the incidents below, generate recurring incident patterns.

RULES:
- Output patterns as JSON objects
- 1 pattern per root cause
- Ignore rare incidents
- Do NOT merge unrelated issues

OUTPUT FORMAT:
[
  {{
    "pattern_name": "Clear issue name",
    "description": "What incidents this pattern represents",
    "rules": [
      "Title contains ...",
      "Category is ...",
      "Description contains ..."
    ]
  }}
]

INCIDENT DATA:
{chr(10).join(lines)}
"""

        
        attempts = 0
        patterns_chunk = []
        while attempts <= RETRY_LIMIT and len(patterns_chunk) == 0:
            response = llm.invoke(prompt)
            patterns_chunk = extract_patterns(response.content)
            if len(patterns_chunk) == 0:
                print(f"⚠️ Empty patterns detected. Retrying ({attempts+1}/{RETRY_LIMIT})...")
            attempts += 1

        print(f"✅ Patterns extracted from chunk: {len(patterns_chunk)}")
        all_patterns.extend(patterns_chunk)

    # =====================================================
    # Merge & Deduplicate
    # =====================================================
    final_patterns = []
    seen_names = set()
    for p in all_patterns:
        name = p.get("pattern_name","").strip().lower()
        if name and name not in seen_names:
            final_patterns.append(p)
            seen_names.add(name)
        if len(final_patterns) >= TOP_PATTERN_COUNT:
            break

    print(f"\n🎯 Total Top Patterns Selected: {len(final_patterns)}")
    state["patterns"] = final_patterns
    return state

# =====================================================
# NODE 3 — SAVE OUTPUT
# =====================================================

def save_output(state: PatternState):
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(state["patterns"], f, indent=2, ensure_ascii=False)

    print("\n🎉 PROCESS COMPLETED SUCCESSFULLY")
    print(f"📄 OUTPUT FILE: {OUTPUT_FILE}")
    return state

# =====================================================
# BUILD LANGGRAPH PIPELINE
# =====================================================

graph = StateGraph(PatternState)
graph.add_node("load", load_data)
graph.add_node("generate", generate_patterns)
graph.add_node("save", save_output)
graph.set_entry_point("load")
graph.add_edge("load", "generate")
graph.add_edge("generate", "save")
graph.add_edge("save", END)
app = graph.compile()

# =====================================================
# RUN
# =====================================================

if __name__ == "__main__":
    app.invoke({"incidents": [], "patterns": []})
