"""
Streamlit UI for Incident Matcher - Enhanced Version
=====================================================
Features:
- Single incident matching mode
- Batch processing mode with CSV upload
- Score threshold filtering (CE and Cosine)
- Graph visualization of incident-problem relationships
- Export functionality (CSV, JSON)
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

class ConfidenceLevel:
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"

# Page configuration
st.set_page_config(
    page_title="Incident Matcher - Enhanced",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'mode' not in st.session_state:
    st.session_state.mode = 'single'
if 'batch_results' not in st.session_state:
    st.session_state.batch_results = None
if 'show_results' not in st.session_state:
    st.session_state.show_results = False

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
    .mode-button {
        padding: 10px 20px;
        margin: 5px;
        border-radius: 5px;
        border: 2px solid #667eea;
        background-color: white;
        color: #667eea;
        font-weight: bold;
        cursor: pointer;
    }
    .mode-button-active {
        background-color: #667eea;
        color: white;
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
    
    client = chromadb.PersistentClient(path=persist_dir)
    
    existing_collections = [c.name for c in client.list_collections()]
    
    if Config.CHROMA_COLLECTION_NAME in existing_collections:
        collection = client.get_collection(name=Config.CHROMA_COLLECTION_NAME)
        if collection.count() == len(_train_data):
            return client, collection
        client.delete_collection(Config.CHROMA_COLLECTION_NAME)
    
    collection = client.create_collection(
        name=Config.CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    
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


@st.cache_resource
def load_cross_encoder():
    """Load CrossEncoder model for reranking"""
    return CrossEncoder(Config.CROSS_ENCODER_MODEL)


@st.cache_resource
def load_embedding_model():
    """Load SentenceTransformer for generating embeddings"""
    return SentenceTransformer(Config.EMBEDDING_MODEL)


def match_incident(test_embedding: list, test_context: str, collection, train_data: list, 
                   cross_encoder, top_k: int = 5, use_reranking: bool = True) -> dict:
    """
    Match test incident against train embeddings.
    Returns top_k matches with all scores (no filtering here - filtering done at display time).
    """
    start_time = time.time()
    
    retrieve_k = top_k * 3 if use_reranking else top_k
    
    results = collection.query(
        query_embeddings=[test_embedding],
        n_results=retrieve_k,
        include=["metadatas", "distances", "documents"]
    )
    
    retrieval_time = time.time() - start_time
    
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
    
    rerank_time = 0
    if use_reranking and cross_encoder is not None:
        rerank_start = time.time()
        
        if not test_context or not test_context.strip():
            for c in candidates:
                c['ce_score'] = None
        else:
            pairs = [(test_context, c['context']) for c in candidates]
            ce_scores = cross_encoder.predict(pairs)
            
            import math
            def sigmoid(x):
                try:
                    return 1 / (1 + math.exp(-x))
                except:
                    return 0.0 if x < 0 else 1.0
            
            for i, score in enumerate(ce_scores):
                raw_logit = float(score)
                normalized_score = sigmoid(raw_logit)
                candidates[i]['ce_score'] = normalized_score
                candidates[i]['ce_logit'] = raw_logit
            
            candidates.sort(key=lambda x: x['ce_score'], reverse=True)
        
        rerank_time = time.time() - rerank_start
    else:
        for c in candidates:
            c['ce_score'] = None
    
    # Take top_k matches (no threshold filtering - that's done at display time)
    top_matches = candidates[:top_k]
    
    for i, match in enumerate(top_matches):
        match['rank'] = i + 1
    
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


def process_batch_csv(uploaded_file, embedding_model, collection, train_data, cross_encoder,
                      top_k, use_reranking):
    """
    Process uploaded CSV file and match all incidents.
    Collects top_k matches per incident with ALL scores (no threshold filtering).
    Threshold filtering is applied at display/graph time.
    """
    
    # Read CSV
    df = pd.read_csv(uploaded_file)
    
    required_cols = ['number', 'short_description', 'description']
    for col in required_cols:
        if col not in df.columns:
            st.error(f"Missing required column: {col}")
            return None
    
    batch_results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, row in df.iterrows():
        status_text.text(f"Processing {idx + 1}/{len(df)}: {row['number']}")
        
        # Create context
        context = f"{row['short_description']} {row['description']}".strip()
        
        # Generate embedding
        embedding = embedding_model.encode(context).tolist()
        
        # Match - get top_k without threshold filtering
        result = match_incident(
            embedding, context, collection, train_data, cross_encoder,
            top_k=top_k, use_reranking=use_reranking
        )
        
        # Store ALL top_k results with scores
        for match in result['top_matches']:
            batch_results.append({
                'test_incident': row['number'],
                'test_description': row['short_description'],
                'matched_incident': match['number'],
                'matched_description': match['short_description'],
                'problem_id': match['problem_id'],
                'ce_score': match.get('ce_score'),
                'cosine_score': match['cosine_score'],
                'rank': match['rank']
            })
        
        progress_bar.progress((idx + 1) / len(df))
    
    progress_bar.empty()
    status_text.empty()
    
    return batch_results


def get_score(result, use_ce=True):
    """Get the appropriate score (CE or Cosine) for a result."""
    if use_ce and result.get('ce_score') is not None:
        return result['ce_score']
    return result['cosine_score']


def filter_results_by_threshold(batch_results, threshold_min=0.0, use_ce=True, threshold_max=1.0):
    """
    Filter batch results by score threshold range.
    Returns results where threshold_min <= score <= threshold_max.
    """
    return [r for r in batch_results 
            if threshold_min <= get_score(r, use_ce) <= threshold_max]


def apply_topk_per_incident(results, top_k, use_ce=True):
    """
    Apply top-K filtering per incident.
    Returns at most top_k results per unique test_incident, sorted by score.
    """
    if not results or top_k is None:
        return results
    
    # Group by incident
    incident_groups = {}
    for r in results:
        inc = r['test_incident']
        if inc not in incident_groups:
            incident_groups[inc] = []
        incident_groups[inc].append(r)
    
    # Take top-K per incident
    filtered = []
    for inc, group in incident_groups.items():
        sorted_group = sorted(group, key=lambda x: get_score(x, use_ce), reverse=True)
        filtered.extend(sorted_group[:top_k])
    
    return filtered


def get_score_statistics(batch_results, use_ce=True):
    """Calculate score statistics for batch results."""
    if not batch_results:
        return {'min': 0, 'max': 0, 'avg': 0, 'count': 0}
    
    scores = [get_score(r, use_ce) for r in batch_results]
    return {
        'min': min(scores),
        'max': max(scores),
        'avg': sum(scores) / len(scores),
        'count': len(scores)
    }


def create_graph_visualization(batch_results, selected_incident=None, threshold_min=0.0, 
                                use_ce=True, top_k=None, threshold_max=1.0):
    """
    Create network graph visualization of incident-problem relationships.
    
    Args:
        batch_results: All batch results
        selected_incident: If set, show only this incident's bipartite graph
        threshold_min: Minimum score threshold for filtering edges
        threshold_max: Maximum score threshold for filtering edges
        use_ce: Use CE score if available, otherwise cosine
        top_k: Maximum edges per incident (applied after threshold filtering)
    
    Returns:
        tuple: (figure, num_test_incidents, num_problem_ids, num_edges)
    """
    
    # Filter by threshold first
    filtered_results = filter_results_by_threshold(batch_results, threshold_min, use_ce, threshold_max)
    
    # Filter by selected incident if specified
    if selected_incident and selected_incident != "All Incidents":
        filtered_results = [r for r in filtered_results if r['test_incident'] == selected_incident]
    
    # Apply top-K per incident
    filtered_results = apply_topk_per_incident(filtered_results, top_k, use_ce)
    
    if not filtered_results:
        return None, 0, 0, 0
    
    # Create network graph
    G = nx.Graph()
    
    # Track unique nodes
    test_incidents = set()
    problem_ids = set()
    
    # Add nodes and edges
    for result in filtered_results:
        test_inc = result['test_incident']
        problem_id = result['problem_id']
        
        test_incidents.add(test_inc)
        problem_ids.add(problem_id)
        
        # Get score for edge weight
        score = result.get('ce_score') if use_ce and result.get('ce_score') is not None else result['cosine_score']
        
        # Add nodes
        G.add_node(test_inc, node_type='test', label=result['test_description'][:30])
        G.add_node(problem_id, node_type='problem', label=problem_id)
        
        # Add edge with weight and details
        edge_key = (test_inc, problem_id)
        if G.has_edge(*edge_key):
            # Update if this edge has higher score
            if score > G.edges[edge_key]['weight']:
                G.edges[edge_key]['weight'] = score
                G.edges[edge_key]['matched_inc'] = result['matched_incident']
                G.edges[edge_key]['ce_score'] = result.get('ce_score')
                G.edges[edge_key]['cosine_score'] = result['cosine_score']
        else:
            G.add_edge(test_inc, problem_id, 
                      weight=score, 
                      matched_inc=result['matched_incident'],
                      ce_score=result.get('ce_score'),
                      cosine_score=result['cosine_score'])
    
    if len(G.nodes()) == 0:
        return None, 0, 0
    
    # Create Plotly figure
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Edge traces with color based on weight
    edge_traces = []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        
        weight = edge[2]['weight']
        # Color based on score: green for high, yellow for medium, red for low
        if weight >= 0.7:
            color = '#28a745'  # Green
            width = 3
        elif weight >= 0.4:
            color = '#ffc107'  # Yellow
            width = 2
        else:
            color = '#dc3545'  # Red
            width = 1
        
        ce_info = f"CE: {edge[2]['ce_score']:.4f}" if edge[2].get('ce_score') is not None else "CE: N/A"
        hover_text = f"Score: {weight:.4f}<br>{ce_info}<br>Cosine: {edge[2]['cosine_score']:.4f}<br>Via: {edge[2]['matched_inc']}"
        
        edge_traces.append(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=width, color=color),
                hoverinfo='text',
                text=hover_text,
                showlegend=False
            )
        )
    
    # Node traces
    test_nodes_x = []
    test_nodes_y = []
    test_nodes_text = []
    test_nodes_hover = []
    
    problem_nodes_x = []
    problem_nodes_y = []
    problem_nodes_text = []
    problem_nodes_hover = []
    
    for node in G.nodes(data=True):
        x, y = pos[node[0]]
        if node[1]['node_type'] == 'test':
            test_nodes_x.append(x)
            test_nodes_y.append(y)
            test_nodes_text.append(node[0])
            test_nodes_hover.append(f"{node[0]}<br>{node[1]['label']}")
        else:
            problem_nodes_x.append(x)
            problem_nodes_y.append(y)
            problem_nodes_text.append(node[0])
            problem_nodes_hover.append(f"Problem: {node[0]}")
    
    test_node_trace = go.Scatter(
        x=test_nodes_x, y=test_nodes_y,
        mode='markers+text',
        text=test_nodes_text,
        textposition="top center",
        textfont=dict(size=10),
        marker=dict(size=25, color='#667eea', line=dict(width=2, color='white')),
        name='Test Incidents',
        hoverinfo='text',
        hovertext=test_nodes_hover
    )
    
    problem_node_trace = go.Scatter(
        x=problem_nodes_x, y=problem_nodes_y,
        mode='markers+text',
        text=problem_nodes_text,
        textposition="top center",
        textfont=dict(size=10),
        marker=dict(size=20, color='#f093fb', symbol='square', line=dict(width=2, color='white')),
        name='Problem IDs',
        hoverinfo='text',
        hovertext=problem_nodes_hover
    )
    
    title = "Incident-Problem Relationship Graph"
    if selected_incident and selected_incident != "All Incidents":
        title = f"Bipartite Graph: {selected_incident}"
    
    fig = go.Figure(data=edge_traces + [test_node_trace, problem_node_trace],
                    layout=go.Layout(
                        title=dict(text=title, font=dict(size=16)),
                        showlegend=True,
                        hovermode='closest',
                        margin=dict(b=20, l=20, r=20, t=50),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        height=600,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    ))
    
    return fig, len(test_incidents), len(problem_ids), G.number_of_edges()


def export_results(batch_results, format='csv'):
    """Export batch results to CSV or JSON"""
    
    if not batch_results:
        return ""
    
    if format == 'csv':
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=batch_results[0].keys())
        writer.writeheader()
        writer.writerows(batch_results)
        return output.getvalue()
    
    elif format == 'json':
        return json.dumps(batch_results, indent=2)
    
    elif format == 'edge_csv':
        # Export as edge list: incident1, problem1, incident1, problem2, etc.
        edges = []
        for result in batch_results:
            edges.append({
                'source_incident': result['test_incident'],
                'target_problem': result['problem_id'],
                'via_incident': result['matched_incident'],
                'ce_score': result.get('ce_score'),
                'cosine_score': result['cosine_score'],
                'rank': result['rank']
            })
        
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=edges[0].keys())
        writer.writeheader()
        writer.writerows(edges)
        return output.getvalue()


# Initialize data
try:
    with st.spinner("⚡ Loading data, ChromaDB, and models..."):
        train_data, test_data = load_data()
        chroma_client, chroma_collection = init_chroma_db(train_data)
        cross_encoder = load_cross_encoder()
        embedding_model = load_embedding_model()
    data_ready = True
except Exception as e:
    data_ready = False
    init_error = str(e)

# Header with mode toggle
st.markdown("""
<div class="header">
    <h1>🔍 Incident Matcher - Enhanced</h1>
    <p>Single & Batch Processing with Advanced Filtering</p>
</div>
""", unsafe_allow_html=True)

if not data_ready:
    st.error(f"❌ Failed to initialize: {init_error}")
    st.stop()

# Mode selection buttons
col1, col2, col3 = st.columns([1, 1, 3])
with col1:
    if st.button("📝 Single Mode", use_container_width=True, 
                 type="primary" if st.session_state.mode == 'single' else "secondary"):
        st.session_state.mode = 'single'
        st.session_state.show_results = False
        st.rerun()

with col2:
    if st.button("📊 Batch Mode", use_container_width=True,
                 type="primary" if st.session_state.mode == 'batch' else "secondary"):
        st.session_state.mode = 'batch'
        st.session_state.show_results = False
        st.rerun()

st.divider()

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    top_k = st.slider("Top K Results", min_value=1, max_value=20, value=5)
    use_reranking = st.checkbox("Use CrossEncoder Reranking", value=True)
    
    st.divider()
    st.header("🎯 Batch Mode Settings")
    
    st.caption("Filter matches by score threshold (SS Threshold)")
    
    # SS Threshold range slider (same as before)
    ss_threshold_range = st.slider(
        "SS Threshold Range",
        min_value=0.0,
        max_value=1.0,
        value=(0.90, 0.99),
        step=0.01,
        help="Score threshold range. Only matches with scores in this range are shown."
    )
    
    # Use minimum of range as threshold
    batch_threshold_min = ss_threshold_range[0]
    batch_threshold_max = ss_threshold_range[1]
    
    # Checkbox to choose between CE and Cosine
    use_ce_for_threshold = st.checkbox(
        "Use CrossEncoder Score (if available)", 
        value=True,
        help="If checked, uses CE score for threshold filtering. Otherwise uses Cosine similarity."
    )
    
    st.divider()
    st.header("📊 Data Statistics")
    st.metric("Train Incidents", len(train_data))
    st.metric("Test Incidents", len(test_data))
    
    if st.button("🔄 Reload Data"):
        st.cache_resource.clear()
        st.rerun()

# SINGLE MODE
if st.session_state.mode == 'single':
    st.subheader("📝 Single Incident Matching")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Select Test Incident")
        
        test_options = {f"{inc['number']}: {inc['short_description'][:50]}...": i 
                       for i, inc in enumerate(test_data)}
        
        selected_test = st.selectbox(
            "Test Incident ID",
            options=[""] + list(test_options.keys()),
            help="Select a test incident to find matching train incidents"
        )
        
        if selected_test:
            test_idx = test_options[selected_test]
            test_incident = test_data[test_idx]
            
            st.markdown("#### Incident Details")
            st.write(f"**Number:** {test_incident['number']}")
            st.write(f"**Short Description:** {test_incident['short_description']}")
            st.write(f"**Description:** {test_incident['description']}")
            
            match_btn = st.button("⚡ Find Matching Incidents", type="primary", use_container_width=True)
        else:
            match_btn = False
    
    with col2:
        st.markdown("### Matching Results")
        
        if selected_test and match_btn:
            with st.spinner("🔄 Finding matching incidents..."):
                try:
                    test_embedding = test_incident['embedding']
                    test_context = test_incident['context']
                    
                    result = match_incident(
                        test_embedding, test_context, chroma_collection, train_data, 
                        cross_encoder, top_k=top_k, use_reranking=use_reranking
                    )
                    
                    st.success(f"✅ Found {len(result['top_matches'])} matches")
                    
                    perf_col1, perf_col2, perf_col3 = st.columns(3)
                    with perf_col1:
                        st.metric("🔍 Retrieval", f"{result['timing']['retrieval_ms']:.1f}ms")
                    with perf_col2:
                        st.metric("🔄 Rerank", f"{result['timing']['rerank_ms']:.1f}ms")
                    with perf_col3:
                        st.metric("⏱️ Total", f"{result['timing']['total_ms']:.1f}ms")
                    
                    st.markdown("---")
                    
                    for match in result['top_matches']:
                        with st.container():
                            col_rank, col_info = st.columns([1, 5])
                            
                            with col_rank:
                                st.markdown(f"### {match['rank']}")
                            
                            with col_info:
                                st.markdown(f"**{match['number']}**")
                                st.write(f"{match['short_description']}")
                                
                                if match.get('ce_score') is not None:
                                    st.progress(match['ce_score'])
                                    logit_info = f" (logit: {match.get('ce_logit', 0):.2f})" if 'ce_logit' in match else ""
                                    st.caption(f"CE: {match['ce_score']:.4f}{logit_info} | Cosine: {match['cosine_score']:.4f} | Problem: {match['problem_id']}")
                                else:
                                    st.progress(match['cosine_score'])
                                    st.caption(f"Cosine: {match['cosine_score']:.4f} | Problem: {match['problem_id']}")
                            
                            st.markdown("---")
                
                except Exception as e:
                    st.error(f"❌ Matching failed: {str(e)}")
        elif not selected_test:
            st.info("👈 Select a test incident to find matches")

# BATCH MODE
elif st.session_state.mode == 'batch':
    
    if not st.session_state.show_results:
        st.subheader("📊 Batch Processing Mode")
        
        st.markdown("""
        Upload a CSV file with test incidents to process in batch.
        
        **Required columns:**
        - `number`: Incident ID
        - `short_description`: Short description
        - `description`: Full description
        """)
        
        uploaded_file = st.file_uploader("Upload Test CSV", type=['csv'])
        
        if uploaded_file is not None:
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            
            # Preview
            df_preview = pd.read_csv(uploaded_file)
            st.markdown("### Preview (first 5 rows)")
            st.dataframe(df_preview.head())
            
            # Reset file pointer
            uploaded_file.seek(0)
            
            if st.button("🚀 Start Batch Processing", type="primary", use_container_width=True):
                with st.spinner("Processing batch..."):
                    batch_results = process_batch_csv(
                        uploaded_file, embedding_model, chroma_collection, train_data,
                        cross_encoder, top_k, use_reranking
                    )
                    
                    if batch_results:
                        st.session_state.batch_results = batch_results
                        st.session_state.show_results = True
                        st.rerun()
    
    else:
        # Show results screen
        st.subheader("📊 Batch Processing Results")
        
        if st.button("⬅️ Back to Upload"):
            st.session_state.show_results = False
            st.rerun()
        
        batch_results = st.session_state.batch_results
        
        # Get unique test incidents for dropdown
        unique_incidents = sorted(set(r['test_incident'] for r in batch_results))
        num_total_incidents = len(unique_incidents)
        
        # Large dataset threshold
        LARGE_DATASET_THRESHOLD = 50
        is_large_dataset = num_total_incidents > LARGE_DATASET_THRESHOLD
        
        st.success(f"✅ Processed {num_total_incidents} incidents with {len(batch_results)} total matches (Top-K per incident)")
        
        # Show score statistics to help user set threshold
        score_stats = get_score_statistics(batch_results, use_ce_for_threshold)
        stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
        with stats_col1:
            st.metric("Min Score", f"{score_stats['min']:.3f}")
        with stats_col2:
            st.metric("Max Score", f"{score_stats['max']:.3f}")
        with stats_col3:
            st.metric("Avg Score", f"{score_stats['avg']:.3f}")
        with stats_col4:
            st.metric("Total Matches", score_stats['count'])
        
        st.caption(f"💡 Score statistics using {'CE' if use_ce_for_threshold else 'Cosine'} scores. Adjust threshold in sidebar based on these values.")
        
                
        # Graph Controls Section
        st.markdown("### 📈 Relationship Graph")
        
        st.info(f"**Filtering Logic:** Threshold filter applied first (score ≥ {batch_threshold_min:.2f}), then Top-K (≤ {top_k} edges per incident)")
        
        # Controls in columns
        ctrl_col1, ctrl_col2 = st.columns([3, 2])
        
        with ctrl_col1:
            # Dropdown to select specific incident
            if is_large_dataset:
                # For large datasets, no "All Incidents" option
                selected_incident = st.selectbox(
                    "🔍 Select Incident (Bipartite View)",
                    options=unique_incidents,
                    help="Select an incident to see its bipartite graph. Full graph disabled for large datasets."
                )
            else:
                # For small datasets, allow "All Incidents"
                selected_incident = st.selectbox(
                    "🔍 Filter by Incident",
                    options=["All Incidents"] + unique_incidents,
                    help="Select a specific incident to see its bipartite graph, or 'All Incidents' for full graph"
                )
        
        with ctrl_col2:
            # Display current settings from sidebar
            st.metric("Threshold", f"{batch_threshold_min:.2f}")
            st.caption(f"Score Type: {'CE' if use_ce_for_threshold else 'Cosine'}")
        
        # Create and display graph with sidebar settings
        fig, num_incidents, num_problems, num_edges = create_graph_visualization(
            batch_results, 
            selected_incident=selected_incident,
            threshold_min=batch_threshold_min,
            use_ce=use_ce_for_threshold,
            top_k=top_k,
            threshold_max=batch_threshold_max
        )
        
        if fig:
            # Show graph stats
            stats_col1, stats_col2, stats_col3 = st.columns(3)
            with stats_col1:
                st.metric("Test Incidents", num_incidents)
            with stats_col2:
                st.metric("Problem IDs", num_problems)
            with stats_col3:
                st.metric("Edges", num_edges)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Legend
            st.markdown("""
            **Legend:** 🟢 High score (≥0.7) | 🟡 Medium (0.4-0.7) | 🔴 Low (<0.4)  
            **Nodes:** 🔵 Test Incidents | 🟣 Problem IDs
            """)
        else:
            st.warning(f"⚠️ No matches found with threshold ≥ {batch_threshold_min:.2f}. Try lowering the threshold in the sidebar.")
            st.info("💡 Tip: Adjust 'Minimum Score Threshold' in sidebar to see more matches.")
        
        st.divider()
        
        # Export options
        st.markdown("### 💾 Export Results")
        
        # Filter results for export based on current settings
        export_results_filtered = filter_results_by_threshold(
            batch_results, batch_threshold_min, use_ce_for_threshold, batch_threshold_max
        )
        export_results_filtered = apply_topk_per_incident(
            export_results_filtered, top_k, use_ce_for_threshold
        )
        
        if export_results_filtered:
            exp_col1, exp_col2, exp_col3, exp_col4 = st.columns(4)
            
            with exp_col1:
                csv_data = export_results(export_results_filtered, 'csv')
                st.download_button(
                    label="📥 CSV (Filtered)",
                    data=csv_data,
                    file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with exp_col2:
                json_data = export_results(export_results_filtered, 'json')
                st.download_button(
                    label="📥 JSON (Filtered)",
                    data=json_data,
                    file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            
            with exp_col3:
                edge_csv_data = export_results(export_results_filtered, 'edge_csv')
                st.download_button(
                    label="📥 Edge List",
                    data=edge_csv_data,
                    file_name=f"edge_list_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with exp_col4:
                # Export ALL results (unfiltered)
                all_csv_data = export_results(batch_results, 'csv')
                st.download_button(
                    label="📥 All Results",
                    data=all_csv_data,
                    file_name=f"all_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        else:
            st.warning("No results to export with current threshold")
        
        st.divider()
        
        # Detailed results table
        st.markdown("### 📋 Detailed Results")
        
        # Apply filters to table
        table_results = batch_results
        
        # Filter by selected incident
        if selected_incident and selected_incident != "All Incidents":
            table_results = [r for r in table_results if r['test_incident'] == selected_incident]
        
        # Filter by threshold
        table_results = filter_results_by_threshold(
            table_results, batch_threshold_min, use_ce_for_threshold, batch_threshold_max
        )
        
        # Apply top-K per incident
        table_results = apply_topk_per_incident(table_results, top_k, use_ce_for_threshold)
        
        if table_results:
            df = pd.DataFrame(table_results)
            st.dataframe(df, use_container_width=True)
            st.caption(f"Showing {len(table_results)} matches (threshold ≥ {batch_threshold_min:.2f}, ≤ {top_k} per incident)")
        else:
            st.info("No results match the current filters")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: gray; font-size: 12px;">
    <p>⚡ Incident Matcher v7.0 | Single & Batch Processing with Graph Visualization</p>
</div>
""", unsafe_allow_html=True)
