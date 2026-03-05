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
from incident_pattern_matcher_prod import PatternMatcher, ConfidenceLevel

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
        st.session_state.matcher = PatternMatcher()
        st.session_state.matcher_ready = True
    except Exception as e:
        st.session_state.matcher_ready = False
        st.session_state.error = str(e)

if 'history' not in st.session_state:
    st.session_state.history = []

# Header
st.markdown("""
<div class="header">
    <h1>🔍 Incident Pattern Matcher</h1>
    <p>Classify incidents against predefined patterns using AI</p>
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
    
    st.divider()
    
    # History
    st.subheader("📊 Classification History")
    history_count = len(st.session_state.history)
    st.metric("Total Classifications", history_count)
    
    if st.button("🗑️ Clear History", key="clear_history"):
        st.session_state.history = []
        st.rerun()

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Incident Details")
    
    with st.form("incident_form"):
        title = st.text_input(
            "Incident Title",
            placeholder="e.g., Database connection timeout",
            help="Brief title of the incident"
        )
        
        description = st.text_area(
            "Description",
            placeholder="Detailed description of the incident...",
            height=120,
            help="Provide a detailed description"
        )
        
        col_a, col_b = st.columns(2)
        with col_a:
            category = st.text_input(
                "Category",
                placeholder="e.g., Infrastructure",
                help="Main category of the incident"
            )
        
        with col_b:
            sub_category = st.text_input(
                "Sub Category",
                placeholder="e.g., Database",
                help="Subcategory of the incident"
            )
        
        submit_btn = st.form_submit_button(
            "🔎 Classify Incident",
            use_container_width=True,
            type="primary"
        )

with col2:
    st.subheader("📊 Classification Result")
    
    if submit_btn:
        # Validate input
        if not all([title, description, category, sub_category]):
            st.error("❌ All fields are required!")
        else:
            with st.spinner("🔄 Classifying incident..."):
                incident = {
                    "title": title,
                    "description": description,
                    "category": category,
                    "sub_category": sub_category
                }
                
                try:
                    result = st.session_state.matcher.match(incident)
                    
                    # Add to history
                    history_entry = {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "title": title,
                        "category": category,
                        "matched_pattern": result['matched_pattern'],
                        "confidence": result['confidence'],
                        "reason": result['reason']
                    }
                    st.session_state.history.append(history_entry)
                    
                    # Display result
                    st.success("✅ Classification Complete!")
                    
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
                    st.write("**Reason:**")
                    st.info(result['reason'])
                
                except Exception as e:
                    st.error(f"❌ Classification failed: {str(e)}")

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
        matches = (df['matched_pattern'] != 'No Match').sum()
        st.metric("Matched", matches)
    
    with stats_col3:
        high_conf = (df['confidence'] == ConfidenceLevel.HIGH).sum()
        st.metric("High Confidence", high_conf)
    
    with stats_col4:
        no_match = (df['matched_pattern'] == 'No Match').sum()
        st.metric("No Match", no_match)
    
    # Display table
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "timestamp": st.column_config.TextColumn("Time", width="medium"),
            "title": st.column_config.TextColumn("Incident Title", width="large"),
            "category": st.column_config.TextColumn("Category", width="medium"),
            "matched_pattern": st.column_config.TextColumn("Pattern", width="medium"),
            "confidence": st.column_config.TextColumn("Confidence", width="small"),
            "reason": st.column_config.TextColumn("Reason", width="large")
        }
    )
    
    # Export option
    col_exp1, col_exp2 = st.columns(2)
    
    with col_exp1:
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name=f"incident_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col_exp2:
        json_data = df.to_json(orient='records', indent=2)
        st.download_button(
            label="📥 Download as JSON",
            data=json_data,
            file_name=f"incident_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )

else:
    st.info("💡 No classifications yet. Submit an incident above to get started!")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: gray; font-size: 12px;">
    <p>Incident Pattern Matcher v1.0 | Powered by LLM Classification</p>
    <p>Last updated: 2026-03-05</p>
</div>
""", unsafe_allow_html=True)
