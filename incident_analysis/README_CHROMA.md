# ChromaDB Incident Matcher - Quick Start Guide

## 🚀 What's New

The incident matcher now uses **ChromaDB** for ultra-fast vector search, making classification **10-20x faster**!

### Key Improvements

| Feature | Old Version | ChromaDB Version |
|---------|-------------|------------------|
| Vector Search | ~100-200ms | ~5-10ms |
| Embeddings | Recomputed every run | Cached in ChromaDB |
| Startup Time | Slow (compute embeddings) | Fast (load from cache) |
| Similarity Display | ❌ No | ✅ Yes with scores |
| Timing Logs | ❌ Basic | ✅ Detailed |

## 📦 Installation

```bash
# Install dependencies
pip install -r requirements_prod.txt

# The dependencies include:
# - chromadb>=0.4.0
# - sentence-transformers>=2.2.0
# - scikit-learn>=1.3.0
# - langchain-ollama>=0.1.0
```

## 🎯 Usage Options

### 1. Streamlit Web UI (Recommended)

```bash
streamlit run streamlit_app.py
```

**Features:**
- ⚡ Fast classification with ChromaDB
- 📊 Real-time performance metrics (vector search, LLM, total time)
- 📈 Similar patterns visualization with similarity scores
- 📋 Classification history with timing
- 💾 Export results to CSV/JSON

**When you click "🔎 Classify Incident":**
- Shows vector search time (~5-10ms)
- Shows LLM inference time
- Shows total classification time
- Displays top 5 similar patterns with scores

### 2. Command Line - Interactive Mode

```bash
python incident_matcher_chroma.py
```

**Output includes:**
- Detailed timing logs for every operation
- Top similar patterns with similarity scores
- Visual similarity bars
- Complete timing summary

### 3. Command Line - Batch Mode

```bash
python incident_matcher_chroma.py batch incidents.json results.json
```

Process multiple incidents with performance stats.

### 4. Rebuild ChromaDB Index

```bash
python incident_matcher_chroma.py rebuild
```

Force rebuild the ChromaDB index (useful if patterns file changed).

## 📊 Performance Metrics

### Timing Breakdown

Every classification shows:
- **Vector Search Time**: ChromaDB similarity search (~5-10ms)
- **LLM Inference Time**: AI classification (~2-3 seconds)
- **Total Time**: End-to-end classification

### Example Output

```
[16:35:42.123] ⏱️  STARTED: Vector similarity search (ChromaDB)
[16:35:42.128] ✅ COMPLETED: Vector similarity search (ChromaDB) [0.0052s]

📊 TOP SIMILAR PATTERNS (by cosine similarity):
--------------------------------------------------
   1. [0.8234] ████████████████
      User Login Issue
   2. [0.7891] ███████████████
      Permission Denied
--------------------------------------------------

⏱️  TIMING SUMMARY
============================================================
  Vector similarity search (ChromaDB): 0.0052s
  LLM inference: 2.3456s
  Total Time: 2.3508s
============================================================
```

## 🗂️ ChromaDB Storage

- **Location**: `./chroma_patterns_db/`
- **Purpose**: Persistent vector storage for pattern embeddings
- **Benefit**: Embeddings computed once, reused forever
- **Size**: ~10-50MB depending on pattern count

## 🧪 Testing

```bash
# Test single incident with detailed output
python test_chroma_performance.py single

# Test multiple incidents
python test_chroma_performance.py multi

# Compare with/without caching
python test_chroma_performance.py compare
```

## 📝 Files Overview

| File | Purpose |
|------|---------|
| `incident_matcher_chroma.py` | Main ChromaDB-based matcher |
| `streamlit_app.py` | Web UI (uses ChromaDB matcher) |
| `test_chroma_performance.py` | Performance testing |
| `requirements_prod.txt` | Dependencies |
| `chroma_patterns_db/` | ChromaDB vector storage |

## 🎨 Streamlit UI Features

### Performance Metrics Display
- ⚡ Vector Search time in milliseconds
- 🤖 LLM Inference time
- ⏱️ Total classification time

### Similar Patterns Visualization
- Top 5 most similar patterns
- Similarity scores (0-1 range)
- Progress bars for visual representation

### Classification History
- Timestamp of each classification
- Performance time for each incident
- Export to CSV/JSON

## 🔧 Configuration

Edit `incident_matcher_chroma.py` Config class:

```python
class Config:
    PATTERN_FILE: str = "incident_patterns_final.json"
    MODEL_NAME: str = "llama3.1"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    TOP_K_PATTERNS: int = 15  # Number of patterns to send to LLM
    CHROMA_PERSIST_DIR: str = "./chroma_patterns_db"
```

## 🚨 Troubleshooting

### ChromaDB not found
```bash
pip install chromadb
```

### Slow first run
- First run builds ChromaDB index (one-time cost)
- Subsequent runs use cached embeddings (much faster)

### Force rebuild index
```bash
python incident_matcher_chroma.py rebuild
```

### Ollama not running
Make sure Ollama is running:
```bash
ollama serve
```

## 📈 Performance Comparison

**Before (sklearn cosine similarity):**
- Startup: ~2-3 seconds (compute embeddings)
- Classification: ~3-5 seconds
- Vector search: ~100-200ms

**After (ChromaDB):**
- Startup: ~0.5-1 second (load from cache)
- Classification: ~2-3 seconds
- Vector search: ~5-10ms ⚡

**Result: 10-20x faster vector search!**

## 🎯 Next Steps

1. Run Streamlit app: `streamlit run streamlit_app.py`
2. Click "🔎 Classify Incident" button
3. See fast results with timing metrics
4. View similar patterns with scores
5. Check classification history with performance data

---

**Version**: 2.0 (ChromaDB Optimized)  
**Last Updated**: 2026-03-05
