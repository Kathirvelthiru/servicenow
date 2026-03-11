"""
Data Preparation Script
========================
Prepares data for incident matching:
1. Reads from Excel (my_dataset.xlsx) OR CSV files (train.csv, test.csv)
2. Splits by problem_id: with problem_id -> train, without -> test
3. Creates objects with 'context' attribute (short_description + description)
4. Generates embeddings using SentenceTransformer

Usage:
    python prepare_data.py              # Process from Excel
    python prepare_data.py --from-csv   # Process from existing CSV files

To use with a new dataset:
1. Place your Excel file as my_dataset.xlsx OR
2. Place train.csv and test.csv directly
3. Run this script
4. Run generate_embeddings.py
5. Run streamlit run streamlit_app.py
"""

import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List
import zipfile
import xml.etree.ElementTree as ET


def read_excel_data(filepath: str) -> tuple[list[str], list[list[str]]]:
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
                    # Handle rich text
                    texts = []
                    for r in si.findall(f'.//{ns}r'):
                        t = r.find(f'.//{ns}t')
                        if t is not None and t.text:
                            texts.append(t.text)
                    shared_strings.append(''.join(texts))
        except Exception as e:
            print(f'Error reading shared strings: {e}')
            return [], []
        
        # Read worksheet
        sheet_xml = z.read('xl/worksheets/sheet1.xml')
        root = ET.fromstring(sheet_xml)
        ns = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
        
        rows_data = []
        for row in root.findall(f'.//{ns}row'):
            row_data = {}
            for cell in row.findall(f'.//{ns}c'):
                cell_ref = cell.get('r')  # e.g., 'A1', 'B2'
                cell_value = cell.find(f'{ns}v')
                cell_type = cell.get('t')
                
                if cell_value is not None and cell_value.text:
                    if cell_type == 's':  # Shared string
                        idx = int(cell_value.text)
                        if idx < len(shared_strings):
                            row_data[cell_ref] = shared_strings[idx]
                    else:
                        row_data[cell_ref] = cell_value.text
            
            if row_data:
                rows_data.append(row_data)
        
        return shared_strings, rows_data


def col_letter_to_index(col: str) -> int:
    """Convert column letter to index (A=0, B=1, etc.)"""
    result = 0
    for char in col:
        if char.isalpha():
            result = result * 26 + (ord(char.upper()) - ord('A') + 1)
    return result - 1


def parse_cell_ref(cell_ref: str) -> tuple[str, int]:
    """Parse cell reference like 'A1' into column letter and row number"""
    col = ''
    row = ''
    for char in cell_ref:
        if char.isalpha():
            col += char
        else:
            row += char
    return col, int(row) if row else 0


def convert_rows_to_records(rows_data: list[dict]) -> list[dict]:
    """Convert row data to list of records with column headers"""
    if not rows_data:
        return []
    
    # First row contains headers
    header_row = rows_data[0]
    headers = {}
    for cell_ref, value in header_row.items():
        col, _ = parse_cell_ref(cell_ref)
        headers[col] = value
    
    # Convert remaining rows to records
    records = []
    for row in rows_data[1:]:
        record = {}
        for cell_ref, value in row.items():
            col, _ = parse_cell_ref(cell_ref)
            if col in headers:
                record[headers[col]] = value
        if record:
            records.append(record)
    
    return records


@dataclass
class IncidentData:
    """Incident data object with context and other attributes"""
    number: str
    context: str  # Combined short_description + description
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


def create_incident_object(record: dict) -> IncidentData:
    """Create IncidentData object from record"""
    short_desc = record.get('short_description', '')
    desc = record.get('description', '')
    context = f"{short_desc} {desc}".strip()
    
    return IncidentData(
        number=record.get('number', ''),
        context=context,
        short_description=short_desc,
        description=desc,
        priority=record.get('priority', ''),
        cmdb_ci=record.get('cmdb_ci', ''),
        category=record.get('category', ''),
        subcategory=record.get('subcategory', ''),
        problem_id=record.get('problem_id'),
        assigned_to=record.get('assigned_to', ''),
        assignment_group=record.get('assignment_group', ''),
        state=record.get('state', ''),
        u_issue=record.get('u_issue', ''),
        caller_id=record.get('caller_id', ''),
        sys_created_on=record.get('sys_created_on', ''),
        resolved_at=record.get('resolved_at', '')
    )


def save_to_csv(records: list[dict], filepath: str):
    """Save records to CSV file"""
    if not records:
        return
    
    headers = list(records[0].keys())
    
    with open(filepath, 'w', encoding='utf-8') as f:
        # Write header
        f.write(','.join(f'"{h}"' for h in headers) + '\n')
        
        # Write rows
        for record in records:
            values = []
            for h in headers:
                val = str(record.get(h, '')).replace('"', '""')
                values.append(f'"{val}"')
            f.write(','.join(values) + '\n')


def main():
    print("=" * 60)
    print("📊 DATA PREPARATION SCRIPT")
    print("=" * 60)
    
    # Read Excel file
    excel_path = Path(__file__).parent.parent / 'my_dataset.xlsx'
    print(f"\n📂 Reading: {excel_path}")
    
    shared_strings, rows_data = read_excel_data(str(excel_path))
    print(f"   Found {len(shared_strings)} shared strings")
    print(f"   Found {len(rows_data)} rows")
    
    # Convert to records
    records = convert_rows_to_records(rows_data)
    print(f"   Converted to {len(records)} records")
    
    if not records:
        print("❌ No records found!")
        return
    
    # Show sample columns
    print(f"\n📋 Columns: {list(records[0].keys())}")
    
    # Split by problem_id
    train_records = []  # With problem_id
    test_records = []   # Without problem_id
    
    for record in records:
        problem_id = record.get('problem_id', '')
        if problem_id and problem_id.strip():
            train_records.append(record)
        else:
            test_records.append(record)
    
    print(f"\n📊 Split Results:")
    print(f"   Train (with problem_id): {len(train_records)} records")
    print(f"   Test (without problem_id): {len(test_records)} records")
    
    # Save to CSV
    output_dir = Path(__file__).parent
    
    train_csv = output_dir / 'train.csv'
    test_csv = output_dir / 'test.csv'
    
    save_to_csv(train_records, str(train_csv))
    save_to_csv(test_records, str(test_csv))
    
    print(f"\n✅ Saved: {train_csv}")
    print(f"✅ Saved: {test_csv}")
    
    # Create incident objects with context
    print("\n🔧 Creating incident objects...")
    
    train_incidents = [create_incident_object(r) for r in train_records]
    test_incidents = [create_incident_object(r) for r in test_records]
    
    # Save as JSON for further processing
    train_json = output_dir / 'train_incidents.json'
    test_json = output_dir / 'test_incidents.json'
    
    with open(train_json, 'w', encoding='utf-8') as f:
        json.dump([asdict(inc) for inc in train_incidents], f, indent=2)
    
    with open(test_json, 'w', encoding='utf-8') as f:
        json.dump([asdict(inc) for inc in test_incidents], f, indent=2)
    
    print(f"✅ Saved: {train_json}")
    print(f"✅ Saved: {test_json}")
    
    print("\n" + "=" * 60)
    print("✅ DATA PREPARATION COMPLETE")
    print("=" * 60)
    print("\nNext step: Run generate_embeddings.py to create embeddings")


def read_csv_file(filepath: str) -> list[dict]:
    """Read CSV file and return list of records"""
    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        if not lines:
            return []
        
        # Parse header
        header_line = lines[0].strip()
        headers = []
        in_quote = False
        current = ""
        for char in header_line:
            if char == '"':
                in_quote = not in_quote
            elif char == ',' and not in_quote:
                headers.append(current.strip().strip('"'))
                current = ""
            else:
                current += char
        headers.append(current.strip().strip('"'))
        
        # Parse data rows
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            
            values = []
            in_quote = False
            current = ""
            for char in line:
                if char == '"':
                    in_quote = not in_quote
                elif char == ',' and not in_quote:
                    values.append(current.strip().strip('"'))
                    current = ""
                else:
                    current += char
            values.append(current.strip().strip('"'))
            
            record = {}
            for i, header in enumerate(headers):
                if i < len(values):
                    record[header] = values[i]
            records.append(record)
    
    return records


def main_from_csv():
    """Process from existing CSV files"""
    print("=" * 60)
    print("📊 DATA PREPARATION FROM CSV")
    print("=" * 60)
    
    output_dir = Path(__file__).parent
    train_csv = output_dir / 'train.csv'
    test_csv = output_dir / 'test.csv'
    
    if not train_csv.exists():
        print(f"❌ Train CSV not found: {train_csv}")
        return
    if not test_csv.exists():
        print(f"❌ Test CSV not found: {test_csv}")
        return
    
    print(f"\n📂 Reading: {train_csv}")
    train_records = read_csv_file(str(train_csv))
    print(f"   Found {len(train_records)} train records")
    
    print(f"\n📂 Reading: {test_csv}")
    test_records = read_csv_file(str(test_csv))
    print(f"   Found {len(test_records)} test records")
    
    # Create incident objects with context
    print("\n🔧 Creating incident objects...")
    
    train_incidents = [create_incident_object(r) for r in train_records]
    test_incidents = [create_incident_object(r) for r in test_records]
    
    # Save as JSON for further processing
    train_json = output_dir / 'train_incidents.json'
    test_json = output_dir / 'test_incidents.json'
    
    with open(train_json, 'w', encoding='utf-8') as f:
        json.dump([asdict(inc) for inc in train_incidents], f, indent=2)
    
    with open(test_json, 'w', encoding='utf-8') as f:
        json.dump([asdict(inc) for inc in test_incidents], f, indent=2)
    
    print(f"✅ Saved: {train_json}")
    print(f"✅ Saved: {test_json}")
    
    print("\n" + "=" * 60)
    print("✅ DATA PREPARATION COMPLETE")
    print("=" * 60)
    print("\nNext step: Run generate_embeddings.py to create embeddings")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--from-csv":
        main_from_csv()
    else:
        main()
