import json
import re 
import pandas as pd
from typing import TypedDict
from langgraph.graph import StateGraph, END 
from langchain_ollama import ChatOllama
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# -------------------------
# State definition
# -------------------------
class PatternState(TypedDict):
    df: pd.DataFrame
    texts: list
    vectors: object
    labels: list
    clusters: dict
    patterns: list

# -------------------------
# Config
# -------------------------
DATASET = "ServiceNow_Banking_Incidents_500.xlsx"
OUTPUT = "incident_patterns_graph.json"
NUM_CLUSTERS = 15    
SAMPLE_PER_CLUSTER = 6

# -------------------------
# Helpers
# -------------------------
def safe_extract_json(text: str):
    """Robustly extract first JSON array/object from LLM output."""

    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    
    m = re.search(r"\[\s*{.*}\s*\]", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass

    m = re.search(r"\{\s*\".*\}\s*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass

    return {"error": "parse_failed", "raw": text}

# -------------------------
# Node: load dataset
# -------------------------
def node_load(state: PatternState):
    df = pd.read_excel(DATASET)
   
    df.columns = [c.strip() for c in df.columns]
    state["df"] = df
    return state

# -------------------------
# Node: preprocess (combine text)
# -------------------------
def node_preprocess(state: PatternState):
    df = state["df"]
    
    texts = (df["Title"].astype(str) + ". " + df["Description"].astype(str) + ". " +
             df.get("Resolution Notes", "").astype(str)).str.replace(r"\s+", " ", regex=True).str.strip().tolist()
    state["texts"] = texts
    return state

# -------------------------
# Node: vectorize
# -------------------------
def node_vectorize(state: PatternState):
    texts = state["texts"]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    X = vectorizer.fit_transform(texts)
    state["vectors"] = X
    state["_vectorizer"] = vectorizer  
    return state

# -------------------------
# Node: cluster
# -------------------------
def node_cluster(state: PatternState):
    X = state["vectors"]
    if NUM_CLUSTERS <= 0 or X.shape[0] < 2:
        labels = [0] * X.shape[0]
    else:
        kmeans = KMeans(n_clusters=min(NUM_CLUSTERS, X.shape[0]), random_state=42)
        labels = kmeans.fit_predict(X)
    state["labels"] = labels


    clusters = {}
    for idx, lbl in enumerate(labels):
        clusters.setdefault(int(lbl), []).append(idx)
    state["clusters"] = clusters
    return state

# -------------------------
# Node: generate patterns (LLM)
# -------------------------

llm = ChatOllama(model="llama3:latest")  

def node_generate_patterns(state: PatternState):
    df = state["df"]
    clusters = state["clusters"]
    patterns = []

    for cluster_id in sorted(clusters.keys()):
        indices = clusters[cluster_id]
        sample_idxs = indices[:SAMPLE_PER_CLUSTER]
        sample_texts = []
        for i in sample_idxs:
            row = df.iloc[i]
            s = (
                f"Incident ID: {row.get('Incident ID', '')}\n"
                f"Title: {row.get('Title', '')}\n"
                f"Description: {row.get('Description', '')}\n"
                f"Top Category: {row.get('Top Category', '')}\n"
                f"Sub Category: {row.get('Sub Category', '')}\n"
                f"Resolution Notes: {row.get('Resolution Notes', '')}\n"
                f"Resolved By: {row.get('Resolved By', '')}\n"
                "----"
            )
            sample_texts.append(s)

        prompt = f"""
You are an expert ITSM incident analyst. Given the following resolved incidents (banking domain),
produce a single **incident pattern** that groups these incidents together.

Examples (sample incidents):
{chr(10).join(sample_texts)}

Required JSON output (ONLY JSON, no extra text). Provide a single JSON object:

{{
  "pattern_name": "short descriptive name",
  "description": "one-line description of the pattern",
  "rules": [
     "one rule per string (e.g. Title contains ...)",
     "assignment group contains ...",
     "category is ..."
  ],
  "indicators": {{
     "Title/Short Description": ["keyword1","keyword2"],
     "Description text": ["phrase1","phrase2"],
     "Top Category": ["..."],
     "Sub Category": ["..."],
     "Resolution Notes Keywords": ["..."],
     "Resolved By Roles": ["..."]
  }}
}}
"""

        try:
            resp = llm.invoke([("system", "You are an ITSM pattern mining assistant."), ("user", prompt)])
            
            text = getattr(resp, "content", None) or getattr(resp, "text", None) or str(resp)
        except Exception as e:
            text = f'{{"error":"llm_invoke_failed","detail":"{str(e)}"}}'

        parsed = safe_extract_json(text)
        
        if isinstance(parsed, list) and parsed:
            parsed_obj = parsed[0]
        else:
            parsed_obj = parsed

        
        parsed_obj["_cluster_id"] = int(cluster_id)
        parsed_obj["_sample_count"] = len(sample_idxs)
        parsed_obj["_example_indices"] = sample_idxs
        patterns.append(parsed_obj)

    state["patterns"] = patterns
    return state

# -------------------------
# Node: save output
# -------------------------
def node_save(state: PatternState):
    patterns = state.get("patterns", [])
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(patterns, f, indent=2, ensure_ascii=False)
    return state

# -------------------------
# Build LangGraph
# -------------------------
graph = StateGraph(PatternState)

graph.add_node("load", node_load)
graph.add_node("preprocess", node_preprocess)
graph.add_node("vectorize", node_vectorize)
graph.add_node("cluster", node_cluster)
graph.add_node("generate", node_generate_patterns)
graph.add_node("save", node_save)

graph.set_entry_point("load")
graph.add_edge("load", "preprocess")
graph.add_edge("preprocess", "vectorize")
graph.add_edge("vectorize", "cluster")
graph.add_edge("cluster", "generate")
graph.add_edge("generate", "save")
graph.add_edge("save", END)

app = graph.compile()

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    initial_state = {"df": None, "texts": [], "vectors": None, "labels": [], "clusters": {}, "patterns": []}
    result = app.invoke(initial_state)
    print(f"\nCompleted — patterns saved to {OUTPUT}")
