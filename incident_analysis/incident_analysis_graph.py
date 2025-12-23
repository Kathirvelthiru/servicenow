import pandas as pd
import json
from langchain_ollama import ChatOllama  # Correct import for Ollama + LangChain
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import os

# --- 1. Load dataset
dataset_file = "ServiceNow_Banking_Incidents_500.xlsx"
df = pd.read_excel(dataset_file)
print(f"Loaded {len(df)} rows")

# --- 2. Pre‑process: combine Title + Description into one text column
df["full_text"] = (df["Title"].astype(str) + ". " + df["Description"].astype(str)).str.strip()

# --- 3. Vectorize text for clustering
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X = vectorizer.fit_transform(df["full_text"].tolist())

# --- 4. Cluster incidents
num_clusters = 15  # adjust as needed
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df["cluster_label"] = kmeans.fit_predict(X)
print("Clustering done — cluster counts:")
print(df["cluster_label"].value_counts())

# --- 5. Initialize Ollama (local Llama 3/2) for pattern generation
llm = ChatOllama(model="llama3", temperature=0)

# --- 6. For each cluster, sample incidents and ask Ollama to generate a pattern
patterns = []
for cluster in sorted(df["cluster_label"].unique()):
    subset = df[df["cluster_label"] == cluster]
    sample = subset.head(5)  # first 5 as sample for pattern prompt
    incident_texts = "\n\n".join(
        f"Title: {row['Title']}\nDescription: {row['Description']}\nTop Category: {row['Top Category']}\nSub Category: {row['Sub Category']}\nResolution Notes: {row.get('Resolution Notes','')}\nResolved By: {row.get('Resolved By','')}"
        for _, row in sample.iterrows()
    )
    prompt = f"""
You are an experienced incident‑management analyst. Below are a few resolved incident examples from a banking ServiceNow system:

{incident_texts}

Based on these, define a general **incident pattern**. Provide JSON output in this format:

{{
  "pattern_name": "...",
  "description": "...",
  "rules": [
     "rule 1",
     "rule 2"
  ],
  "indicators": {{
     "Title/Short Description": ["..."],
     "Description text": ["..."],
     "Top Category": ["..."],
     "Sub Category": ["..."],
     "Resolution Notes Keywords": ["..."],
     "Resolved By Roles": ["..."]
  }}
}}
Make sure output is valid JSON only (no commentary).
"""
    resp = llm.invoke([("system", "You are an IT‑SM pattern mining assistant."), ("user", prompt)])
    try:
        pattern = json.loads(resp.content)
    except json.JSONDecodeError:
        print(f"⚠️  Could not parse JSON for cluster {cluster}, raw response:\n{resp.content}")
        continue
    patterns.append(pattern)
    print(f"Generated pattern for cluster {cluster}: {pattern.get('pattern_name')}")

# --- 7. Save patterns
with open("incident_patterns_auto.json", "w", encoding="utf-8") as f:
    json.dump(patterns, f, indent=2, ensure_ascii=False)

print("All patterns saved to incident_patterns_auto.json")
