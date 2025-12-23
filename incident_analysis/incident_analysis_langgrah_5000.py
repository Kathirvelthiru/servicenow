import json
import re
import pandas as pd
from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama

# =====================================================
# CONFIGURATION
# =====================================================

DATASET_FILE = "Updated Dataset.xlsx"
OUTPUT_FILE = "incident_patterns_final.json"

CHUNK_SIZE = 150           
MODEL_NAME = "llama3.1"    

# =====================================================
# STATE DEFINITION
# =====================================================

class PatternState(TypedDict):
    incidents: List[Dict]
    chunks: List[List[Dict]]
    patterns: List[Dict]

# =====================================================
# LLM INITIALIZATION
# =====================================================

llm = ChatOllama(
    model="llama3.1",
    temperature=0.1,
    base_url="http://localhost:11434"
)

# =====================================================
# HELPER FUNCTIONS
# =====================================================

def split_chunks(data: List[Dict], size: int):
    for i in range(0, len(data), size):
        yield data[i:i + size]

def extract_json_objects(text: str) -> List[Dict]:
    """
    Safely extract valid JSON objects from LLM output.
    This avoids total failure when the model adds extra text.
    """
    results = []
    matches = re.findall(r"\{[\s\S]*?\}", text)

    for m in matches:
        try:
            obj = json.loads(m)
            if (
                isinstance(obj, dict)
                and "pattern_title" in obj
                and "belongs_when" in obj
                and "resolution_notes" in obj
            ):
                results.append(obj)
        except Exception:
            continue

    return results

# =====================================================
# NODE 1 — LOAD DATA
# =====================================================

def load_data(state: PatternState):
    df = pd.read_excel(DATASET_FILE)
    df = df.fillna("")

    incidents = df.to_dict(orient="records")

    print(f"✅ Total Incidents Loaded: {len(incidents)}")

    state["incidents"] = incidents
    return state

# =====================================================
# NODE 2 — CHUNK DATA
# =====================================================

def chunk_data(state: PatternState):
    chunks = list(split_chunks(state["incidents"], CHUNK_SIZE))

    print(f"✅ Total Chunks: {len(chunks)}")

    state["chunks"] = chunks
    state["patterns"] = []
    return state

# =====================================================
# NODE 3 — PATTERN GENERATION
# =====================================================

def generate_patterns(state: PatternState):
    all_patterns = []

    for idx, chunk in enumerate(state["chunks"], start=1):
        print(f"🔹 Processing chunk {idx}/{len(state['chunks'])} (size={len(chunk)})")

        compact_incidents = []
        for i, inc in enumerate(chunk):
            compact_incidents.append(
                f"{i+1}. "
                f"{inc.get('Title','')} | "
                f"{inc.get('Description','')} | "
                f"{inc.get('Top Category','')} | "
                f"{inc.get('Sub Category','')} | "
                f"{inc.get('Resolution Notes','')}"
            )

        prompt = f"""
You are an ITSM INCIDENT PATTERN ENGINE.

OBJECTIVE:
Extract as MANY DISTINCT INCIDENT PATTERNS as possible.

STRICT RULES:
- One issue = one pattern
- Do NOT summarize incidents
- Do NOT merge unrelated problems
- Output ONLY JSON objects (multiple allowed)

FORMAT FOR EACH PATTERN:

{{
  "pattern_title": "Clear issue name",
  "belongs_when": [
    "Title contains ...",
    "Description mentions ...",
    "Top Category is ...",
    "Sub Category is ..."
  ],
  "resolution_notes": [
    "Step 1",
    "Step 2",
    "Step 3"
  ]
}}

INCIDENTS:
{chr(10).join(compact_incidents)}
"""

        response = llm.invoke(prompt)
        extracted = extract_json_objects(response.content)

        print(f"✅ Patterns extracted from chunk {idx}: {len(extracted)}")
        all_patterns.extend(extracted)

    state["patterns"] = all_patterns
    return state

# =====================================================
# NODE 4 — SAVE OUTPUT
# =====================================================

def save_output(state: PatternState):
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(state["patterns"], f, indent=2, ensure_ascii=False)

    print("\n🎉 PROCESS COMPLETED SUCCESSaFULLY")
    print(f"✅ TOTAL PATTERNS GENERATED: {len(state['patterns'])}")
    print(f"📄 OUTPUT FILE: {OUTPUT_FILE}")

    return state

# =====================================================
# BUILD LANGGRAPH PIPELINE
# =====================================================

graph = StateGraph(PatternState)

graph.add_node("load", load_data)
graph.add_node("chunk", chunk_data)
graph.add_node("generate", generate_patterns)
graph.add_node("save", save_output)

graph.set_entry_point("load")
graph.add_edge("load", "chunk")
graph.add_edge("chunk", "generate")
graph.add_edge("generate", "save")
graph.add_edge("save", END)

app = graph.compile()

# =====================================================
# RUN
# =====================================================

if __name__ == "__main__":
    app.invoke({
        "incidents": [],
        "chunks": [],
        "patterns": []
    })
