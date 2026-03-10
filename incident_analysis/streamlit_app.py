"""
Streamlit UI for Incident Pattern Matcher
==========================================
Interactive web-based interface for incident classification
"""

import streamlit as st
import json
from datetime import datetime
from pathlib import Path
import pandas as pd
from incident_matcher_vector_only import VectorOnlyMatcher, ConfidenceLevel

# Page configuration
st.set_page_config(
    page_title="Incident Pattern Matcher",
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
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'matcher' not in st.session_state:
    try:
        with st.spinner("⚡ Initializing Vector-Only Matcher..."):
            st.session_state.matcher = VectorOnlyMatcher()
        st.session_state.matcher_ready = True
    except Exception as e:
        st.session_state.matcher_ready = False
        st.session_state.error = str(e)

# Force reload button for development
if st.sidebar.button("🔄 Reload Matcher"):
    if 'matcher' in st.session_state:
        del st.session_state.matcher
    st.rerun()

# Header
st.markdown("""
<div class="header">
    <h1>🔍 Incident Pattern Matcher</h1>
</div>
""", unsafe_allow_html=True)

# Check if matcher is ready
if not st.session_state.matcher_ready:
    st.error(f"❌ Failed to initialize Pattern Matcher: {st.session_state.error}")
    st.info("Make sure Ollama is running and the pattern file exists.")
    st.stop()

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Load patterns info
    patterns = st.session_state.matcher.patterns
    st.metric("Available Patterns", len(patterns))
    
    with st.expander("📋 View Available Patterns"):
        for i, pattern in enumerate(patterns, 1):
            pattern_name = pattern.get('name', 'Unknown')
            pattern_desc = pattern.get('description', 'No description')
            st.write(f"**{i}. {pattern_name}**")
            st.caption(pattern_desc)
    
    # Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Incident Details")
    
    # Incident Title Dropdown
    title_options = ["", "server issue", "database issue", "network issue", "application issue"]
    selected_title = st.selectbox(
        "Incident Title",
        title_options,
        help="Select the incident title from predefined options"
    )
    
    # Set default values based on selection
    if selected_title == "server issue":
        default_description = "A server issue occurs when the server becomes unavailable, slow, or fails to process requests properly, causing disruption to applications, services, or user access. This can be due to hardware failure, network problems, high load, or software errors."
        default_category = "banking"
        default_sub_category = "banking"
    elif selected_title == "network issue":
        default_description = "A network issue occurs when there is a disruption or failure in network connectivity, preventing devices or systems from communicating properly. This can be caused by configuration errors, hardware faults, bandwidth problems, or connectivity failures."
        default_category = "banking"
        default_sub_category = "banking"
    elif selected_title == "application issue":
        default_description = "An application issue occurs when a software application fails to function as expected, such as crashes, errors, slow performance, or incorrect outputs, affecting user operations or business processes."
        default_category = "banking"
        default_sub_category = "banking"
    else:
        default_description = ""
        default_category = ""
        default_sub_category = ""
    
    with st.form("incident_form"):
        description = st.text_area(
            "Description",
            value=default_description,
            placeholder="Detailed description of the incident...",
            height=120,
            help="Provide a detailed description"
        )
        
        col_a, col_b = st.columns(2)
        with col_a:
            category = st.text_input(
                "Category",
                value=default_category,
                placeholder="e.g., Infrastructure",
                help="Main category of the incident"
            )
        
        with col_b:
            sub_category = st.text_input(
                "Sub Category",
                value=default_sub_category,
                placeholder="e.g., Database",
                help="Subcategory of the incident"
            )
        
        submit_btn = st.form_submit_button(
            "⚡ Match Instantly",
            width="stretch",
            type="primary"
        )

with col2:
    st.subheader("📊 Classification Result")
    
    if submit_btn:
        # Validate input
        if not all([selected_title, description, category, sub_category]):
            st.error("❌ All fields are required!")
        else:
            with st.spinner("🔄 Classifying incident..."):
                incident = {
                    "title": selected_title,
                    "description": description,
                    "category": category,
                    "sub_category": sub_category
                }
                
                try:
                    result = st.session_state.matcher.match(incident, return_top_k=5)
                    
                    # Display result
                    st.success("✅ Matches Found Instantly!")
                    
                    # Performance metrics
                    perf_col1, perf_col2 = st.columns(2)
                    with perf_col1:
                        st.metric(
                            "⚡ Vector Search",
                            f"{result['timing']['vector_search_ms']:.1f}ms",
                            help="Time for ChromaDB similarity search"
                        )
                    with perf_col2:
                        st.metric(
                            "⏱️ Total Time",
                            f"{result['timing']['total_ms']:.1f}ms",
                            help="Total matching time"
                        )
                    
                    st.markdown("---")
                    
                    result_col1, result_col2 = st.columns(2)
                    
                    with result_col1:
                        st.metric(
                            "Matched Pattern",
                            result['matched_pattern'],
                            help="The incident pattern that best matches this incident"
                        )
                    
                    with result_col2:
                        confidence = result['confidence']
                        if confidence == ConfidenceLevel.HIGH:
                            confidence_color = "🟢"
                        elif confidence == ConfidenceLevel.MEDIUM:
                            confidence_color = "🟡"
                        else:
                            confidence_color = "🔴"
                        
                        st.metric(
                            "Confidence Level",
                            f"{confidence_color} {confidence}",
                            help="How confident the match is"
                        )
                    
                    st.markdown("---")
                    
                    # Show top matches
                    if 'top_matches' in result and result['top_matches']:
                        st.write("**📊 Top 5 Matching Patterns:**")
                        
                        for match in result['top_matches']:
                            with st.container():
                                col_rank, col_info = st.columns([1, 5])
                                
                                with col_rank:
                                    st.markdown(f"### {match['rank']}")
                                
                                with col_info:
                                    # Confidence emoji
                                    if match['confidence'] == ConfidenceLevel.HIGH:
                                        conf_emoji = "🟢"
                                    elif match['confidence'] == ConfidenceLevel.MEDIUM:
                                        conf_emoji = "🟡"
                                    else:
                                        conf_emoji = "🔴"
                                    
                                    # Display with incident ID if available
                                    incident_id = match.get('incident_ID', 'N/A')
                                    st.markdown(f"**ID: {incident_id} - {match['pattern_title']}** {conf_emoji}")
                                    st.progress(match['similarity_score'])
                                    st.caption(f"Similarity: {match['similarity_score']:.4f} | Confidence: {match['confidence']}")
                                
                                st.markdown("---")
                
                except Exception as e:
                    st.error(f"❌ Classification failed: {str(e)}")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: gray; font-size: 12px;">
    <p>⚡ Incident Pattern Matcher v3.0 | Vector-Only</p>
    <p>Last updated: 2026-03-05 | Results in milliseconds!</p>
</div>
""", unsafe_allow_html=True)
