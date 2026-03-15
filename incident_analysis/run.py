#!/usr/bin/env python3
"""
Run Script - All-in-one Incident Matching Pipeline
===================================================
Prepares data, generates embeddings, and launches Streamlit app.

Usage:
    python run.py my_dataset.xlsx           # Fresh start with new dataset
    python run.py my_dataset.xlsx --new     # Same as above (explicit fresh start)
    python run.py my_dataset2.xlsx --append # Append to existing train/test data
    
Examples:
    python run.py ../my_dataset.xlsx        # Process Excel file and start app
    python run.py data.xlsx --append        # Add new data to existing dataset
    python run.py data.xlsx --new           # Fresh start, replace all existing data
"""

import sys
import json
import subprocess
import shutil
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List
import time

# Try to import pandas for better Excel reading
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# Always import these for fallback Excel reading
import zipfile
import xml.etree.ElementTree as ET


@dataclass
class IncidentData:
    """Incident data object with context and other attributes"""
    number: str
    context: str
    short_description: str
    description: str
    priority: str
    cmdb_ci: str
    category: str
    subcategory: str
    problem_id: Optional[str]
    assigned_to: str
    assignment_group: str
    state: str
    u_issue: str
    caller_id: str
    sys_created_on: str
    resolved_at: str
    embedding: Optional[List[float]] = None


def read_excel_with_pandas(filepath: str) -> list[dict]:
    """Read Excel file using pandas"""
    try:
        df = pd.read_excel(filepath)
        return df.to_dict('records')
    except ImportError:
        # openpyxl not installed, fall back to manual parsing
        return None


def read_excel_without_pandas(filepath: str) -> list[dict]:
    """Read Excel file without pandas using zipfile and XML parsing"""
    with zipfile.ZipFile(filepath, 'r') as z:
        # Read shared strings
        shared_strings = []
        try:
            shared_strings_xml = z.read('xl/sharedStrings.xml')
            root = ET.fromstring(shared_strings_xml)
            ns = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
            for si in root.findall(f'.//{ns}si'):
                t = si.find(f'.//{ns}t')
                if t is not None:
                    shared_strings.append(t.text if t.text else '')
                else:
                    texts = []
                    for r in si.findall(f'.//{ns}r'):
                        t = r.find(f'.//{ns}t')
                        if t is not None and t.text:
                            texts.append(t.text)
                    shared_strings.append(''.join(texts))
        except Exception as e:
            print(f'Error reading shared strings: {e}')
            return []
        
        # Read worksheet
        sheet_xml = z.read('xl/worksheets/sheet1.xml')
        root = ET.fromstring(sheet_xml)
        ns = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
        
        rows_data = []
        for row in root.findall(f'.//{ns}row'):
            row_data = {}
            for cell in row.findall(f'.//{ns}c'):
                cell_ref = cell.get('r')
                cell_value = cell.find(f'{ns}v')
                cell_type = cell.get('t')
                
                if cell_value is not None and cell_value.text:
                    if cell_type == 's':
                        idx = int(cell_value.text)
                        if idx < len(shared_strings):
                            row_data[cell_ref] = shared_strings[idx]
                    else:
                        row_data[cell_ref] = cell_value.text
                
            if row_data:
                rows_data.append(row_data)
        
        # Convert to records
        if not rows_data:
            return []
        
        # Parse headers from first row
        def parse_cell_ref(cell_ref):
            col = ''
            for char in cell_ref:
                if char.isalpha():
                    col += char
                else:
                    break
            return col
        
        header_row = rows_data[0]
        headers = {}
        for cell_ref, value in header_row.items():
            col = parse_cell_ref(cell_ref)
            headers[col] = value
        
        # Convert remaining rows
        records = []
        for row in rows_data[1:]:
            record = {}
            for cell_ref, value in row.items():
                col = parse_cell_ref(cell_ref)
                if col in headers:
                    record[headers[col]] = value
            if record:
                records.append(record)
        
        return records


def read_excel(filepath: str) -> list[dict]:
    """Read Excel file - uses pandas if available, otherwise manual parsing"""
    if HAS_PANDAS:
        result = read_excel_with_pandas(filepath)
        if result is not None:
            return result
    # Fall back to manual parsing if pandas fails or not available
    return read_excel_without_pandas(filepath)


def create_incident_object(record: dict) -> IncidentData:
    """Create IncidentData object from record"""
    short_desc = str(record.get('short_description', '') or '')
    desc = str(record.get('description', '') or '')
    context = f"{short_desc} {desc}".strip()
    
    return IncidentData(
        number=str(record.get('number', '') or ''),
        context=context,
        short_description=short_desc,
        description=desc,
        priority=str(record.get('priority', '') or ''),
        cmdb_ci=str(record.get('cmdb_ci', '') or ''),
        category=str(record.get('category', '') or ''),
        subcategory=str(record.get('subcategory', '') or ''),
        problem_id=str(record.get('problem_id', '') or '') if record.get('problem_id') else None,
        assigned_to=str(record.get('assigned_to', '') or ''),
        assignment_group=str(record.get('assignment_group', '') or ''),
        state=str(record.get('state', '') or ''),
        u_issue=str(record.get('u_issue', '') or ''),
        caller_id=str(record.get('caller_id', '') or ''),
        sys_created_on=str(record.get('sys_created_on', '') or ''),
        resolved_at=str(record.get('resolved_at', '') or '')
    )


def read_csv_file(filepath: str) -> list[dict]:
    """Read CSV file and return list of records"""
    import csv
    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(dict(row))
    return records


def prepare_data(excel_path: str, append_mode: bool = False) -> tuple[list, list]:
    """
    Prepare data using prepare_data.py script.
    
    Args:
        excel_path: Path to Excel file
        append_mode: If True, append to existing data; if False, fresh start
    
    Returns:
        tuple: (train_incidents, test_incidents)
    """
    output_dir = Path(__file__).parent
    
    # Run prepare_data.py script
    print(f"\n📂 Running prepare_data.py to process Excel file...")
    prepare_script = output_dir / 'prepare_data.py'
    
    if not prepare_script.exists():
        print(f"❌ prepare_data.py not found!")
        return [], []
    
    # Run the prepare_data.py script
    result = subprocess.run(
        ['python', str(prepare_script)],
        cwd=str(output_dir),
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"❌ prepare_data.py failed!")
        print(result.stderr)
        return [], []
    
    print(result.stdout)
    
    # Load the generated JSON files
    train_json = output_dir / 'train_incidents.json'
    test_json = output_dir / 'test_incidents.json'
    
    if not train_json.exists() or not test_json.exists():
        print(f"❌ train_incidents.json or test_incidents.json not found!")
        return [], []
    
    with open(train_json, 'r', encoding='utf-8') as f:
        new_train_incidents = json.load(f)
    
    with open(test_json, 'r', encoding='utf-8') as f:
        new_test_incidents = json.load(f)
    
    print(f"\n📊 Loaded from prepare_data.py:")
    print(f"   Train: {len(new_train_incidents)} incidents")
    print(f"   Test: {len(new_test_incidents)} incidents")
    
    # Handle append mode (if needed in future)
    if append_mode:
        print("\n⚠️ Append mode not yet implemented with prepare_data.py")
        print("   Using fresh data from prepare_data.py")
    
    return new_train_incidents, new_test_incidents


def generate_embeddings(train_incidents: list, test_incidents: list, append_mode: bool = False):
    """Generate embeddings for incidents"""
    
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("❌ Please install: pip install sentence-transformers")
        return False
    
    output_dir = Path(__file__).parent
    
    print(f"\n⏳ Loading embedding model: all-MiniLM-L6-v2")
    start = time.time()
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print(f"✅ Model loaded in {time.time() - start:.2f}s")
    
    # In append mode, we need to generate embeddings only for new incidents
    if append_mode:
        train_emb_json = output_dir / 'train_with_embeddings.json'
        test_emb_json = output_dir / 'test_with_embeddings.json'
        
        # Load existing embeddings
        existing_train = []
        existing_test = []
        
        if train_emb_json.exists():
            with open(train_emb_json, 'r', encoding='utf-8') as f:
                existing_train = json.load(f)
        
        if test_emb_json.exists():
            with open(test_emb_json, 'r', encoding='utf-8') as f:
                existing_test = json.load(f)
        
        # Get existing numbers with embeddings
        existing_train_numbers = {inc['number'] for inc in existing_train if inc.get('embedding')}
        existing_test_numbers = {inc['number'] for inc in existing_test if inc.get('embedding')}
        
        # Find incidents needing embeddings
        train_needing_emb = [inc for inc in train_incidents if inc['number'] not in existing_train_numbers]
        test_needing_emb = [inc for inc in test_incidents if inc['number'] not in existing_test_numbers]
        
        print(f"\n📊 Generating embeddings for new incidents:")
        print(f"   Train: {len(train_needing_emb)} new")
        print(f"   Test: {len(test_needing_emb)} new")
        
        # Generate embeddings for new train
        if train_needing_emb:
            print("\n📊 Processing new TRAIN incidents...")
            train_contexts = [inc['context'] for inc in train_needing_emb]
            train_embeddings = model.encode(train_contexts, show_progress_bar=True).tolist()
            for i, inc in enumerate(train_needing_emb):
                inc['embedding'] = train_embeddings[i]
        
        # Generate embeddings for new test
        if test_needing_emb:
            print("\n📊 Processing new TEST incidents...")
            test_contexts = [inc['context'] for inc in test_needing_emb]
            test_embeddings = model.encode(test_contexts, show_progress_bar=True).tolist()
            for i, inc in enumerate(test_needing_emb):
                inc['embedding'] = test_embeddings[i]
        
        # Merge with existing
        existing_train_map = {inc['number']: inc for inc in existing_train}
        existing_test_map = {inc['number']: inc for inc in existing_test}
        
        # Update with new embeddings
        for inc in train_needing_emb:
            existing_train_map[inc['number']] = inc
        for inc in test_needing_emb:
            existing_test_map[inc['number']] = inc
        
        # Also ensure all incidents from current list are included
        for inc in train_incidents:
            if inc['number'] in existing_train_map:
                inc['embedding'] = existing_train_map[inc['number']].get('embedding')
        for inc in test_incidents:
            if inc['number'] in existing_test_map:
                inc['embedding'] = existing_test_map[inc['number']].get('embedding')
        
        final_train = train_incidents
        final_test = test_incidents
    else:
        # Fresh start - generate all embeddings
        print("\n📊 Processing TRAIN set...")
        train_contexts = [inc['context'] for inc in train_incidents]
        if train_contexts:
            train_embeddings = model.encode(train_contexts, show_progress_bar=True).tolist()
            for i, inc in enumerate(train_incidents):
                inc['embedding'] = train_embeddings[i]
        
        print("\n📊 Processing TEST set...")
        test_contexts = [inc['context'] for inc in test_incidents]
        if test_contexts:
            test_embeddings = model.encode(test_contexts, show_progress_bar=True).tolist()
            for i, inc in enumerate(test_incidents):
                inc['embedding'] = test_embeddings[i]
        
        final_train = train_incidents
        final_test = test_incidents
    
    # Save with embeddings
    train_emb_json = output_dir / 'train_with_embeddings.json'
    test_emb_json = output_dir / 'test_with_embeddings.json'
    
    with open(train_emb_json, 'w', encoding='utf-8') as f:
        json.dump(final_train, f, indent=2)
    
    with open(test_emb_json, 'w', encoding='utf-8') as f:
        json.dump(final_test, f, indent=2)
    
    print(f"\n✅ Saved: {train_emb_json}")
    print(f"✅ Saved: {test_emb_json}")
    
    if final_train and final_train[0].get('embedding'):
        print(f"\nEmbedding dimension: {len(final_train[0]['embedding'])}")
    
    return True


def clear_chroma_db():
    """Clear ChromaDB to force reindexing"""
    output_dir = Path(__file__).parent
    chroma_dir = output_dir / 'chroma_incidents_db'
    
    if chroma_dir.exists():
        print(f"\n🗑️ Clearing ChromaDB cache: {chroma_dir}")
        shutil.rmtree(chroma_dir)
        print("✅ ChromaDB cache cleared")


def run_streamlit():
    """Launch Streamlit app"""
    output_dir = Path(__file__).parent
    app_path = output_dir / 'streamlit_app_v2.py'
    
    if not app_path.exists():
        app_path = output_dir / 'streamlit_app.py'
    
    print(f"\n🚀 Launching Streamlit: {app_path}")
    print("=" * 60)
    
    subprocess.run(['streamlit', 'run', str(app_path)], cwd=str(output_dir))


def print_usage():
    """Print usage information"""
    print("""
Usage:
    python run.py <excel_file>              # Fresh start with new dataset
    python run.py <excel_file> --new        # Same as above (explicit fresh start)
    python run.py <excel_file> --append     # Append to existing train/test data

Examples:
    python run.py ../my_dataset.xlsx        # Process Excel file and start app
    python run.py data.xlsx --append        # Add new data to existing dataset
    python run.py data.xlsx --new           # Fresh start, replace all existing data

Options:
    --new       Fresh start - clears existing data and ChromaDB
    --append    Append mode - adds new data to existing train/test (skips duplicates)
    """)


def main():
    print("=" * 60)
    print("🚀 INCIDENT MATCHING PIPELINE")
    print("=" * 60)
    
    # Parse arguments
    args = sys.argv[1:]
    
    if not args or args[0] in ['-h', '--help']:
        # Default to ../my_dataset.xlsx if no arguments provided
        args = ['../my_dataset.xlsx']
    
    excel_path = args[0]
    append_mode = '--append' in args
    new_mode = '--new' in args
    
    # Validate Excel file
    excel_file = Path(excel_path)
    if not excel_file.exists():
        # Try relative to parent directory
        excel_file = Path(__file__).parent.parent / excel_path
    
    if not excel_file.exists():
        print(f"❌ Excel file not found: {excel_path}")
        return
    
    print(f"\n📁 Excel file: {excel_file}")
    print(f"📌 Mode: {'APPEND' if append_mode else 'NEW (Fresh Start)'}")
    
    # Step 1: Prepare data
    print("\n" + "=" * 60)
    print("📊 STEP 1: DATA PREPARATION")
    print("=" * 60)
    
    train_incidents, test_incidents = prepare_data(str(excel_file), append_mode=append_mode)
    
    if not train_incidents and not test_incidents:
        print("❌ No data to process!")
        return
    
    # Step 2: Generate embeddings
    print("\n" + "=" * 60)
    print("🔢 STEP 2: EMBEDDING GENERATION")
    print("=" * 60)
    
    success = generate_embeddings(train_incidents, test_incidents, append_mode=append_mode)
    
    if not success:
        print("❌ Embedding generation failed!")
        return
    
    # Step 3: Clear ChromaDB if new mode or fresh start
    if not append_mode or new_mode:
        clear_chroma_db()
    else:
        # In append mode, also clear ChromaDB to reindex with new data
        clear_chroma_db()
    
    # Step 4: Launch Streamlit
    print("\n" + "=" * 60)
    print("🌐 STEP 3: LAUNCHING STREAMLIT APP")
    print("=" * 60)
    
    run_streamlit()


if __name__ == "__main__":
    main()
