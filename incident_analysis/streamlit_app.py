"""
Streamlit UI for Incident Matcher
=====================================
Interactive web-based interface for incident matching.
Matches test incidents against train incidents using vector similarity.

Usage:
1. Place train.csv and test.csv in the incident_analysis folder
2. Run: streamlit run streamlit_app.py
3. Select a test incident from dropdown to find matching train incidents
"""

import streamlit as st
import json
import time
import io
import csv
import pandas as pd
from pathlib import Path
from datetime import datetime

try:
    import chromadb
except ImportError:
    raise ImportError("Please install: pip install chromadb")

try:
    from sentence_transformers import CrossEncoder, SentenceTransformer
    import torch
except ImportError:
    raise ImportError("Please install: pip install sentence-transformers torch")

try:
    import plotly.graph_objects as go
    import networkx as nx
except ImportError:
    raise ImportError("Please install: pip install plotly networkx")

# Configuration
class Config:
    TRAIN_FILE = "train_with_embeddings.json"
    TEST_FILE = "test_with_embeddings.json"
    CHROMA_PERSIST_DIR = "./chroma_incidents_db"
    CHROMA_COLLECTION_NAME = "train_incidents"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    HIGH_CONFIDENCE_THRESHOLD = 0.75
    MEDIUM_CONFIDENCE_THRESHOLD = 0.60
    EXACT_MATCH_THRESHOLD = 0.95
    # CrossEncoder thresholds (different scale than cosine similarity)
    CE_HIGH_THRESHOLD = 0.7
    CE_MEDIUM_THRESHOLD = 0.4
    # Default score thresholds for filtering
    DEFAULT_SS_THRESHOLD_MIN = 0.90
    DEFAULT_SS_THRESHOLD_MAX = 0.99


class ConfidenceLevel:
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


# Page configuration
st.set_page_config(
    page_title="Incident Matcher",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
    }
    .exact-match {
        background-color: #d4edda;
        border: 2px solid #28a745;
        border-radius: 5px;
        padding: 10px;
        margin: 5px 0;
    }
    .high-match {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 10px;
        margin: 5px 0;
    }
    .medium-match {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        border-radius: 5px;
        padding: 10px;
        margin: 5px 0;
    }
    .low-match {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 10px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_data():
    """Load train and test data with embeddings"""
    base_path = Path(__file__).parent
    
    train_file = base_path / Config.TRAIN_FILE
    test_file = base_path / Config.TEST_FILE
    
    if not train_file.exists():
        raise FileNotFoundError(f"Train file not found: {train_file}")
    if not test_file.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")
    
    with open(train_file, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    return train_data, test_data


@st.cache_resource
def init_chroma_db(_train_data):
    """Initialize ChromaDB with train embeddings"""
    base_path = Path(__file__).parent
    persist_dir = str(base_path / Config.CHROMA_PERSIST_DIR)
    
    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path=persist_dir)
    
    # Check if collection exists and has correct count
    existing_collections = [c.name for c in client.list_collections()]
    
    if Config.CHROMA_COLLECTION_NAME in existing_collections:
        collection = client.get_collection(name=Config.CHROMA_COLLECTION_NAME)
        if collection.count() == len(_train_data):
            return client, collection
        # Delete and recreate if count mismatch
        client.delete_collection(Config.CHROMA_COLLECTION_NAME)
    
    # Create new collection
    collection = client.create_collection(
        name=Config.CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    
    # Add train embeddings
    ids = [f"train_{i}" for i in range(len(_train_data))]
    embeddings = [inc['embedding'] for inc in _train_data]
    metadatas = [
        {
            "number": inc.get('number', ''),
            "short_description": inc.get('short_description', ''),
            "problem_id": str(inc.get('problem_id', '')),
            "category": inc.get('category', ''),
            "subcategory": inc.get('subcategory', ''),
            "index": i
        }
        for i, inc in enumerate(_train_data)
    ]
    documents = [inc.get('context', '') for inc in _train_data]
    
    collection.add(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=documents
    )
    
    return client, collection


def get_confidence(similarity_score: float) -> str:
    """Determine confidence level based on cosine similarity score"""
    if similarity_score >= Config.EXACT_MATCH_THRESHOLD:
        return ConfidenceLevel.HIGH
    elif similarity_score >= Config.HIGH_CONFIDENCE_THRESHOLD:
        return ConfidenceLevel.HIGH
    elif similarity_score >= Config.MEDIUM_CONFIDENCE_THRESHOLD:
        return ConfidenceLevel.MEDIUM
    else:
        return ConfidenceLevel.LOW


def get_ce_confidence(ce_score: float) -> str:
    """Determine confidence level based on CrossEncoder score"""
    if ce_score >= Config.CE_HIGH_THRESHOLD:
        return ConfidenceLevel.HIGH
    elif ce_score >= Config.CE_MEDIUM_THRESHOLD:
        return ConfidenceLevel.MEDIUM
    else:
        return ConfidenceLevel.LOW


@st.cache_resource
def load_cross_encoder():
    """Load CrossEncoder model for reranking"""
    # Note: ms-marco model outputs logits, we'll apply sigmoid manually for better control
    return CrossEncoder(Config.CROSS_ENCODER_MODEL)


def match_incident(test_embedding: list, test_context: str, collection, train_data: list, 
                   cross_encoder, top_k: int = 5, use_reranking: bool = True) -> dict:
    """
    Match test incident against train embeddings in ChromaDB.
    
    Two-stage process:
    1. Retrieve top-K candidates using cosine similarity (fast)
    2. Rerank using CrossEncoder for better accuracy (slower but more accurate)
    """
    start_time = time.time()
    
    # Stage 1: Retrieve more candidates than needed for reranking
    retrieve_k = top_k * 3 if use_reranking else top_k  # Get 3x candidates for reranking
    
    results = collection.query(
        query_embeddings=[test_embedding],
        n_results=retrieve_k,
        include=["metadatas", "distances", "documents"]
    )
    
    retrieval_time = time.time() - start_time
    
    # Build candidate list
    candidates = []
    for i, (metadata, distance, document) in enumerate(zip(
        results['metadatas'][0], 
        results['distances'][0],
        results['documents'][0]
    )):
        cosine_sim = 1 - distance
        idx = metadata['index']
        train_inc = train_data[idx]
        
        candidates.append({
            "number": metadata['number'],
            "short_description": metadata['short_description'],
            "problem_id": metadata['problem_id'],
            "category": metadata['category'],
            "subcategory": metadata['subcategory'],
            "cosine_score": round(cosine_sim, 4),
            "context": document,
            "full_data": train_inc
        })
    
    # Stage 2: Rerank with CrossEncoder
    rerank_time = 0
    if use_reranking and cross_encoder is not None:
        rerank_start = time.time()
        
        # Create pairs for CrossEncoder: (query, candidate)
        pairs = [(test_context, c['context']) for c in candidates]
        
        # Debug: Check if pairs have content
        if not test_context or not test_context.strip():
            # Fallback: use short_description if context is empty
            for c in candidates:
                c['ce_score'] = None
                c['confidence'] = get_confidence(c['cosine_score'])
        else:
            # Get CrossEncoder scores (raw logits)
            ce_scores = cross_encoder.predict(pairs)
            
            # Apply sigmoid to normalize logits to [0, 1]
            import math
            def sigmoid(x):
                return 1 / (1 + math.exp(-x))
            
            # Add CE scores to candidates
            for i, score in enumerate(ce_scores):
                raw_logit = float(score)
                normalized_score = sigmoid(raw_logit)
                candidates[i]['ce_score'] = normalized_score
                candidates[i]['ce_logit'] = raw_logit  # Keep raw logit for debugging
                candidates[i]['confidence'] = get_ce_confidence(normalized_score)
            
            # Sort by CrossEncoder score (descending)
            candidates.sort(key=lambda x: x['ce_score'], reverse=True)
        
        rerank_time = time.time() - rerank_start
    else:
        # No reranking - use cosine similarity
        for c in candidates:
            c['ce_score'] = None
            c['confidence'] = get_confidence(c['cosine_score'])
    
    # Take top K after reranking
    top_matches = candidates[:top_k]
    
    # Add ranks
    for i, match in enumerate(top_matches):
        match['rank'] = i + 1
        match['is_exact_match'] = (match['ce_score'] or match['cosine_score']) >= Config.CE_HIGH_THRESHOLD
    
    total_time = time.time() - start_time
    
    return {
        "top_matches": top_matches,
        "timing": {
            "retrieval_ms": round(retrieval_time * 1000, 2),
            "rerank_ms": round(rerank_time * 1000, 2),
            "total_ms": round(total_time * 1000, 2)
        },
        "reranking_used": use_reranking
    }


# Initialize data, ChromaDB, and CrossEncoder
try:
    with st.spinner("⚡ Loading data, ChromaDB, and CrossEncoder..."):
        train_data, test_data = load_data()
        chroma_client, chroma_collection = init_chroma_db(train_data)
        cross_encoder = load_cross_encoder()
    data_ready = True
except Exception as e:
    data_ready = False
    init_error = str(e)

# Force reload button
if st.sidebar.button("🔄 Reload Data"):
    st.cache_resource.clear()
    st.rerun()

# Header
st.markdown("""
<div class="header">
    <h1>🔍 Incident Matcher</h1>
    <p>Match test incidents against train incidents using vector similarity</p>
</div>
""", unsafe_allow_html=True)

# Check if data is ready
if not data_ready:
    st.error(f"❌ Failed to initialize: {init_error}")
    st.info("Make sure train_with_embeddings.json and test_with_embeddings.json exist.")
    st.stop()

# Sidebar
with st.sidebar:
    st.header("📊 Data Statistics")
    st.metric("Train Incidents", len(train_data))
    st.metric("Test Incidents", len(test_data))
    st.metric("ChromaDB Collection", chroma_collection.count())
    
    st.divider()
    st.header("⚙️ Settings")
    top_k = st.slider("Top K Results", min_value=1, max_value=10, value=5)
    use_reranking = st.checkbox("Use CrossEncoder Reranking", value=True, 
                                 help="Rerank results using CrossEncoder for better accuracy")

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Select Test Incident")
    
    # Create dropdown options from test incidents
    test_options = {f"{inc['number']}: {inc['short_description'][:50]}...": i for i, inc in enumerate(test_data)}
    
    selected_test = st.selectbox(
        "Test Incident ID",
        options=[""] + list(test_options.keys()),
        help="Select a test incident to find matching train incidents"
    )
    
    if selected_test:
        test_idx = test_options[selected_test]
        test_incident = test_data[test_idx]
        
        st.markdown("### Selected Incident Details")
        st.write(f"**Number:** {test_incident['number']}")
        st.write(f"**Short Description:** {test_incident['short_description']}")
        st.write(f"**Description:** {test_incident['description']}")
        st.write(f"**Category:** {test_incident['category']}")
        st.write(f"**Subcategory:** {test_incident['subcategory']}")
        
        match_btn = st.button("⚡ Find Matching Incidents", type="primary", use_container_width=True)
    else:
        match_btn = False

with col2:
    st.subheader("📊 Matching Results")
    
    if selected_test and match_btn:
        with st.spinner("🔄 Finding matching incidents..."):
            try:
                # Get test incident embedding and context
                test_embedding = test_incident['embedding']
                test_context = test_incident['context']
                
                # Match against train incidents in ChromaDB with optional reranking
                result = match_incident(
                    test_embedding, 
                    test_context,
                    chroma_collection, 
                    train_data, 
                    cross_encoder,
                    top_k=top_k,
                    use_reranking=use_reranking
                )
                
                # Display result
                rerank_label = " (with CrossEncoder reranking)" if result['reranking_used'] else ""
                st.success(f"✅ Found {len(result['top_matches'])} matches!{rerank_label}")
                
                # Debug info
                if result['reranking_used'] and result['top_matches']:
                    ce_scores_list = [m.get('ce_score', 0) for m in result['top_matches'] if m.get('ce_score') is not None]
                    if ce_scores_list:
                        st.info(f"🔍 CE Score range: {min(ce_scores_list):.4f} - {max(ce_scores_list):.4f}")
                
                # Performance metrics
                perf_col1, perf_col2, perf_col3 = st.columns(3)
                with perf_col1:
                    st.metric(
                        "🔍 Retrieval",
                        f"{result['timing']['retrieval_ms']:.1f}ms",
                        help="Time for ChromaDB similarity search"
                    )
                with perf_col2:
                    st.metric(
                        "🔄 Rerank",
                        f"{result['timing']['rerank_ms']:.1f}ms",
                        help="Time for CrossEncoder reranking"
                    )
                with perf_col3:
                    st.metric(
                        "⏱️ Total",
                        f"{result['timing']['total_ms']:.1f}ms",
                        help="Total matching time"
                    )
                
                st.markdown("---")
                
                # Show top matches
                st.write(f"**📊 Top {top_k} Matching Train Incidents:**")
                
                for match in result['top_matches']:
                    # Determine styling based on confidence
                    if match['is_exact_match']:
                        conf_emoji = "⭐"
                        match_class = "exact-match"
                    elif match['confidence'] == ConfidenceLevel.HIGH:
                        conf_emoji = "🟢"
                        match_class = "high-match"
                    elif match['confidence'] == ConfidenceLevel.MEDIUM:
                        conf_emoji = "🟡"
                        match_class = "medium-match"
                    else:
                        conf_emoji = "🔴"
                        match_class = "low-match"
                    
                    with st.container():
                        col_rank, col_info = st.columns([1, 5])
                        
                        with col_rank:
                            st.markdown(f"### {match['rank']}")
                        
                        with col_info:
                            exact_label = " **EXACT MATCH**" if match['is_exact_match'] else ""
                            st.markdown(f"**{match['number']}** {conf_emoji}{exact_label}")
                            st.write(f"{match['short_description']}")
                            # Show appropriate score based on reranking
                            if match.get('ce_score') is not None:
                                # CrossEncoder with sigmoid outputs [0, 1]
                                st.progress(match['ce_score'])
                                logit_info = f" (logit: {match.get('ce_logit', 0):.2f})" if 'ce_logit' in match else ""
                                st.caption(f"CE Score: {match['ce_score']:.4f}{logit_info} | Cosine: {match['cosine_score']:.4f} | Problem ID: {match['problem_id']}")
                            else:
                                # Cosine is already in [0, 1]
                                st.progress(match['cosine_score'])
                                st.caption(f"Cosine: {match['cosine_score']:.4f} | Problem ID: {match['problem_id']} | {match['category']}/{match['subcategory']}")
                        
                        st.markdown("---")
            
            except Exception as e:
                st.error(f"❌ Matching failed: {str(e)}")
    elif not selected_test:
        st.info("👈 Select a test incident from the dropdown to find matches")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: gray; font-size: 12px;">
    <p>⚡ Incident Matcher v5.0 | ChromaDB + CrossEncoder Reranking</p>
    <p>Two-stage matching: Fast retrieval → Accurate reranking</p>
    <p>To update data: Replace train_with_embeddings.json and test_with_embeddings.json, then click Reload</p>
</div>
""", unsafe_allow_html=True)
