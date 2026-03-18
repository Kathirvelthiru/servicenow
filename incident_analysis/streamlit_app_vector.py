"""
Streamlit UI for Vector-Only Incident Matcher
==============================================
Ultra-fast classification using ONLY vector similarity (no LLM)
Results in milliseconds instead of seconds!
"""

import streamlit as st
import json
from datetime import datetime
from pathlib import Path
import pandas as pd
from incident_matcher_vector_only import VectorOnlyMatcher, ConfidenceLevel

# Page configuration
st.set_page_config(
    page_title="Vector-Only Incident Matcher",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .header {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
    }
    .speed-badge {
        background-color: #28a745;
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'matcher' not in st.session_state:
    try:
        with st.spinner("⚡ Initializing Vector-Only Matcher (No LLM)..."):
            # load only the top 15 patterns for matching
            st.session_state.matcher = VectorOnlyMatcher(pattern_file="incident_patterns_top_15.json")
        st.session_state.matcher_ready = True
    except Exception as e:
        st.session_state.matcher_ready = False
        st.session_state.error = str(e)

if 'history' not in st.session_state:
    st.session_state.history = []

# Header
st.markdown("""
<div class="header">
    <h1>⚡ Vector-Only Incident Matcher</h1>
    <p>🚀 Instant classification using pure vector similarity - No LLM needed!</p>
</div>
""", unsafe_allow_html=True)

# Check if matcher is ready
if not st.session_state.matcher_ready:
    st.error(f"❌ Failed to initialize Matcher: {st.session_state.error}")
    st.stop()

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Load patterns from JSON file
    try:
        with open("incident_patterns_top_15.json", "r", encoding="utf-8") as f:
            patterns = json.load(f)

        st.subheader("Available Patterns (Top 15):")
        with st.expander("View Patterns"):
            for pattern in patterns:
                st.markdown(f"**{pattern['pattern_name']}**")
                st.caption(pattern['description'])
    except Exception as e:
        st.error(f"Failed to load patterns: {str(e)}")
    
    st.divider()
    
    # Top K selector
    top_k = st.slider(
        "Number of Top Matches",
        min_value=1,
        max_value=10,
        value=5,
        help="How many top similar patterns to return"
    )
    
    st.divider()
    
    # History
    st.subheader("📊 Classification History")
    history_count = len(st.session_state.history)
    st.metric("Total Classifications", history_count)
    
    if history_count > 0:
        avg_time = sum(h['total_time_ms'] for h in st.session_state.history) / history_count
        st.metric("Avg Response Time", f"{avg_time:.1f}ms")
    
    if st.button("🗑️ Clear History", key="clear_history"):
        st.session_state.history = []
        st.rerun()
    
    st.divider()
    
    # Performance comparison
    # st.subheader("⚡ Speed Comparison")
    # st.write("**Vector-Only:** ~10-50ms")
    # st.write("**With LLM:** ~2000-3000ms")
    # st.success("**100x faster!**")

# Add buttons for matching methods
st.subheader("Select Matching Method")

col1, col2 = st.columns(2)

with col1:
    if st.button("🔍 LLM-Based Matching"):
        st.info("LLM-Based Matching selected.")

with col2:
    if st.button("⚡ Vector-Based Matching"):
        st.info("Vector-Based Matching selected.")

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
            "🔍 Search",
            use_container_width=True,
            type="primary"
        )

with col2:
    st.subheader("📊 Top Matching Patterns")
    
    if submit_btn:
        # Validate input
        if not all([selected_title, description, category, sub_category]):
            st.error("❌ All fields are required!")
        else:
            with st.spinner("⚡ Finding matches..."):
                incident = {
                    "title": selected_title,
                    "description": description,
                    "category": category,
                    "sub_category": sub_category
                }
                
                try:
                    result = st.session_state.matcher.match(incident, return_top_k=top_k)
                    
                    # Add to history
                    history_entry = {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "title": selected_title,
                        "category": category,
                        "matched_pattern": result.get('matched_pattern', ""),
                        "confidence": result['confidence'],
                        "similarity_score": result['similarity_score'],
                        "total_time_ms": result['timing']['total_ms']
                    }
                    st.session_state.history.append(history_entry)
                    
                    # Display result
                    if result.get('is_new'):
                        st.warning(result.get('message'))
                    else:
                        st.success("✅ Matches Found Instantly!")
                        
                        # Performance metric
                        st.metric(
                            "⚡ Response Time",
                            f"{result['timing']['total_ms']:.1f}ms",
                            help="Total time including vector search and processing"
                        )
                        
                        st.markdown("---")
                        
                        # Display top matches
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
                                    
                                    st.markdown(f"**{match['pattern_title']}** {conf_emoji}")
                                    
                                    # Similarity score with progress bar
                                    st.progress(match['similarity_score'])
                                    st.caption(f"Similarity: {match['similarity_score']:.4f} | Confidence: {match['confidence']}")
                                
                                st.markdown("---")
                except Exception as e:
                    st.error(f"❌ Matching failed: {str(e)}")

# History tab
st.divider()
st.subheader("📋 Classification History")

if st.session_state.history:
    # Convert history to DataFrame
    df = pd.DataFrame(st.session_state.history)
    
    # Display stats
    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
    
    with stats_col1:
        st.metric("Total", len(df))
    
    with stats_col2:
        high_conf = (df['confidence'] == ConfidenceLevel.HIGH).sum()
        st.metric("High Confidence", high_conf)
    
    with stats_col3:
        avg_time = df['total_time_ms'].mean()
        st.metric("Avg Time", f"{avg_time:.1f}ms")
    
    with stats_col4:
        avg_similarity = df['similarity_score'].mean()
        st.metric("Avg Similarity", f"{avg_similarity:.3f}")
    
    # Display table
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "timestamp": st.column_config.TextColumn("Time", width="medium"),
            "title": st.column_config.TextColumn("Incident Title", width="large"),
            "category": st.column_config.TextColumn("Category", width="medium"),
            "matched_pattern": st.column_config.TextColumn("Top Match", width="medium"),
            "confidence": st.column_config.TextColumn("Confidence", width="small"),
            "similarity_score": st.column_config.NumberColumn("Similarity", width="small", format="%.4f"),
            "total_time_ms": st.column_config.NumberColumn("Time (ms)", width="small", format="%.1f")
        }
    )
    
    # Export option
    col_exp1, col_exp2 = st.columns(2)
    
    with col_exp1:
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name=f"vector_matches_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col_exp2:
        json_data = df.to_json(orient='records', indent=2)
        st.download_button(
            label="📥 Download as JSON",
            data=json_data,
            file_name=f"vector_matches_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
else:
    st.info("💡 No classifications yet. Submit an incident above to get started!")


# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: gray; font-size: 12px;">
    <p>⚡ Vector-Only Incident Matcher v3.0 | Pure Vector Similarity (No LLM)</p>
</div>
""", unsafe_allow_html=True)
