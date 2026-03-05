"""
Incident Pattern Matcher - ChromaDB Optimized Version
======================================================
Fast incident classification using ChromaDB vector store with detailed timing logs.

Features:
- ChromaDB for persistent vector storage (fast retrieval)
- Detailed timing logs for every operation
- Shows similar patterns with similarity scores
- Pre-computed embeddings cached in ChromaDB
- Real-time progress and performance metrics
"""

import json
import re
import time
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from enum import Enum
import sqlite3

try:
    from langchain_ollama import ChatOllama
except ImportError:
    raise ImportError("Please install langchain_ollama: pip install langchain-ollama")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("Please install: pip install sentence-transformers")

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    raise ImportError("Please install: pip install chromadb")


# =====================================================
# CONFIGURATION
# =====================================================

class Config:
    """Application configuration"""
    PATTERN_FILE: str = "incident_patterns_final.json"
    MODEL_NAME: str = "llama3.1"
    LLM_BASE_URL: str = "http://localhost:11434"
    LLM_TEMPERATURE: float = 0.0
    RETRY_LIMIT: int = 3
    REQUEST_TIMEOUT: int = 30
    LOG_FILE: str = "incident_matcher_chroma.log"
    LOG_LEVEL: int = logging.INFO
    DB_FILE: str = "incident_matches.db"
    ENABLE_DB: bool = True
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    TOP_K_PATTERNS: int = 15
    CHROMA_PERSIST_DIR: str = "./chroma_patterns_db"
    CHROMA_COLLECTION_NAME: str = "incident_patterns"


class ConfidenceLevel(str, Enum):
    """Confidence levels for pattern matches"""
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"
    NO_MATCH = "No Match"


# =====================================================
# TIMING LOGGER
# =====================================================

class TimingLogger:
    """Utility class for detailed timing logs"""
    
    def __init__(self, name: str = "TimingLogger"):
        self.name = name
        self.steps = []
        self.start_time = None
        
    def start(self, operation: str):
        """Start timing an operation"""
        self.start_time = time.time()
        self._log(f"⏱️  STARTED: {operation}")
        
    def end(self, operation: str) -> float:
        """End timing and return duration"""
        if self.start_time is None:
            return 0.0
        duration = time.time() - self.start_time
        self.steps.append({"operation": operation, "duration": duration})
        self._log(f"✅ COMPLETED: {operation} [{duration:.4f}s]")
        self.start_time = None
        return duration
    
    def log_step(self, message: str):
        """Log an intermediate step"""
        self._log(f"   → {message}")
    
    def _log(self, message: str):
        """Print and log message"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"[{timestamp}] {message}")
    
    def summary(self) -> str:
        """Get timing summary"""
        total = sum(s["duration"] for s in self.steps)
        lines = ["\n" + "="*60, "⏱️  TIMING SUMMARY", "="*60]
        for step in self.steps:
            lines.append(f"  {step['operation']}: {step['duration']:.4f}s")
        lines.append("-"*60)
        lines.append(f"  TOTAL TIME: {total:.4f}s")
        lines.append("="*60)
        return "\n".join(lines)


# =====================================================
# LOGGING SETUP
# =====================================================

def setup_logging(log_file: str = Config.LOG_FILE, log_level: int = Config.LOG_LEVEL):
    """Configure logging to file and console"""
    logger = logging.getLogger("chroma_matcher")
    logger.setLevel(log_level)
    
    if logger.handlers:
        return logger
    
    formatter = logging.Formatter(
        '%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


logger = setup_logging()


# =====================================================
# CHROMA VECTOR STORE
# =====================================================

class ChromaPatternStore:
    """ChromaDB-based vector store for pattern embeddings"""
    
    def __init__(self, persist_dir: str = Config.CHROMA_PERSIST_DIR,
                 collection_name: str = Config.CHROMA_COLLECTION_NAME,
                 embedding_model: str = Config.EMBEDDING_MODEL):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        self.client = None
        self.collection = None
        self.embedding_model = None
        self.timer = TimingLogger("ChromaStore")
        
    def initialize(self, patterns: List[Dict], force_rebuild: bool = False) -> float:
        """
        Initialize ChromaDB with pattern embeddings.
        Returns total initialization time.
        """
        total_start = time.time()
        
        # Step 1: Load embedding model
        self.timer.start("Loading embedding model")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        self.timer.end("Loading embedding model")
        
        # Step 2: Initialize ChromaDB client
        self.timer.start("Initializing ChromaDB client")
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.timer.end("Initializing ChromaDB client")
        
        # Step 3: Check if collection exists and has correct count
        self.timer.start("Checking existing collection")
        existing_collections = [c.name for c in self.client.list_collections()]
        collection_exists = self.collection_name in existing_collections
        
        if collection_exists and not force_rebuild:
            self.collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=None  # We handle embeddings ourselves
            )
            existing_count = self.collection.count()
            self.timer.log_step(f"Found existing collection with {existing_count} patterns")
            
            if existing_count == len(patterns):
                self.timer.end("Checking existing collection")
                self.timer.log_step("✓ Using cached embeddings from ChromaDB")
                return time.time() - total_start
        
        self.timer.end("Checking existing collection")
        
        # Step 4: Create/recreate collection
        self.timer.start("Creating new collection")
        if collection_exists:
            self.client.delete_collection(self.collection_name)
            self.timer.log_step("Deleted old collection")
        
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.timer.end("Creating new collection")
        
        # Step 5: Prepare pattern texts
        self.timer.start("Preparing pattern texts")
        pattern_texts = []
        pattern_ids = []
        pattern_metadata = []
        
        for i, pattern in enumerate(patterns):
            pattern_text = f"{pattern.get('pattern_title', '')} {pattern.get('pattern_name', '')} "
            pattern_text += " ".join(pattern.get('belongs_when', []))
            pattern_texts.append(pattern_text)
            pattern_ids.append(f"pattern_{i}")
            pattern_metadata.append({
                "index": i,
                "title": pattern.get('pattern_title', pattern.get('pattern_name', f'Pattern {i}'))
            })
        
        self.timer.log_step(f"Prepared {len(pattern_texts)} pattern texts")
        self.timer.end("Preparing pattern texts")
        
        # Step 6: Compute embeddings
        self.timer.start("Computing embeddings")
        embeddings = self.embedding_model.encode(pattern_texts, show_progress_bar=False)
        self.timer.log_step(f"Computed embeddings shape: {embeddings.shape}")
        self.timer.end("Computing embeddings")
        
        # Step 7: Add to ChromaDB
        self.timer.start("Adding to ChromaDB")
        self.collection.add(
            ids=pattern_ids,
            embeddings=embeddings.tolist(),
            metadatas=pattern_metadata,
            documents=pattern_texts
        )
        self.timer.log_step(f"Added {len(pattern_ids)} patterns to ChromaDB")
        self.timer.end("Adding to ChromaDB")
        
        return time.time() - total_start
    
    def query_similar(self, incident_text: str, k: int = Config.TOP_K_PATTERNS) -> Tuple[List[int], List[float], float]:
        """
        Query ChromaDB for similar patterns.
        Returns (indices, similarity_scores, query_time)
        """
        query_start = time.time()
        
        # Compute incident embedding
        self.timer.start("Computing incident embedding")
        incident_embedding = self.embedding_model.encode([incident_text], show_progress_bar=False)
        self.timer.end("Computing incident embedding")
        
        # Query ChromaDB
        self.timer.start("Querying ChromaDB")
        results = self.collection.query(
            query_embeddings=incident_embedding.tolist(),
            n_results=k,
            include=["metadatas", "distances", "documents"]
        )
        self.timer.end("Querying ChromaDB")
        
        # Extract results
        indices = [m["index"] for m in results["metadatas"][0]]
        # ChromaDB returns distances, convert to similarities (1 - distance for cosine)
        distances = results["distances"][0]
        similarities = [1 - d for d in distances]
        
        query_time = time.time() - query_start
        return indices, similarities, query_time


# =====================================================
# INCIDENT DATABASE
# =====================================================

class IncidentDatabase:
    """Manages persistence of incident match results"""
    
    def __init__(self, db_file: str = Config.DB_FILE):
        self.db_file = db_file
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema with migration support"""
        with sqlite3.connect(self.db_file) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS incident_matches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    incident_title TEXT NOT NULL,
                    incident_description TEXT,
                    incident_category TEXT,
                    incident_sub_category TEXT,
                    matched_pattern TEXT,
                    confidence TEXT,
                    reason TEXT,
                    llm_response_time REAL,
                    vector_query_time REAL,
                    total_time REAL
                )
            """)
            
            # Check if old schema exists and add new columns if needed
            cursor = conn.execute("PRAGMA table_info(incident_matches)")
            columns = [row[1] for row in cursor.fetchall()]
            
            if 'vector_query_time' not in columns:
                conn.execute("ALTER TABLE incident_matches ADD COLUMN vector_query_time REAL")
            
            if 'total_time' not in columns:
                conn.execute("ALTER TABLE incident_matches ADD COLUMN total_time REAL")
            
            conn.commit()
    
    def save_match(self, incident: Dict, result: Dict, times: Dict):
        """Save match result to database"""
        with sqlite3.connect(self.db_file) as conn:
            conn.execute("""
                INSERT INTO incident_matches
                (timestamp, incident_title, incident_description, 
                 incident_category, incident_sub_category,
                 matched_pattern, confidence, reason, 
                 llm_response_time, vector_query_time, total_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                incident.get('title', ''),
                incident.get('description', ''),
                incident.get('category', ''),
                incident.get('sub_category', ''),
                result.get('matched_pattern', ''),
                result.get('confidence', ''),
                result.get('reason', ''),
                times.get('llm_time', 0),
                times.get('vector_time', 0),
                times.get('total_time', 0)
            ))
            conn.commit()


# =====================================================
# PATTERN MATCHER WITH CHROMA
# =====================================================

class ChromaPatternMatcher:
    """Fast incident pattern matching using ChromaDB"""
    
    def __init__(self, pattern_file: str = Config.PATTERN_FILE,
                 enable_db: bool = Config.ENABLE_DB,
                 force_rebuild: bool = False):
        self.pattern_file = pattern_file
        self.patterns = None
        self.llm = None
        self.db = None
        self.chroma_store = None
        self.timer = TimingLogger("PatternMatcher")
        
        print("\n" + "="*60)
        print("🚀 INITIALIZING CHROMA PATTERN MATCHER")
        print("="*60)
        
        init_start = time.time()
        
        self._load_patterns()
        self._init_chroma(force_rebuild)
        self._init_llm()
        
        if enable_db:
            self.timer.start("Initializing SQLite database")
            self.db = IncidentDatabase()
            self.timer.end("Initializing SQLite database")
        
        total_init = time.time() - init_start
        print(f"\n✅ Initialization complete in {total_init:.2f}s")
        print("="*60 + "\n")
    
    def _load_patterns(self):
        """Load patterns from JSON file"""
        self.timer.start("Loading patterns from JSON")
        
        if not Path(self.pattern_file).exists():
            raise FileNotFoundError(f"Pattern file not found: {self.pattern_file}")
        
        with open(self.pattern_file, "r", encoding="utf-8") as f:
            self.patterns = json.load(f)
        
        self.timer.log_step(f"Loaded {len(self.patterns)} patterns from {self.pattern_file}")
        self.timer.end("Loading patterns from JSON")
    
    def _init_chroma(self, force_rebuild: bool = False):
        """Initialize ChromaDB vector store"""
        self.timer.start("Initializing ChromaDB vector store")
        
        self.chroma_store = ChromaPatternStore()
        init_time = self.chroma_store.initialize(self.patterns, force_rebuild)
        
        self.timer.log_step(f"ChromaDB ready in {init_time:.2f}s")
        self.timer.end("Initializing ChromaDB vector store")
    
    def _init_llm(self):
        """Initialize LLM connection"""
        self.timer.start("Initializing LLM connection")
        
        self.llm = ChatOllama(
            model=Config.MODEL_NAME,
            temperature=Config.LLM_TEMPERATURE,
            base_url=Config.LLM_BASE_URL,
            timeout=Config.REQUEST_TIMEOUT
        )
        
        self.timer.log_step(f"LLM model: {Config.MODEL_NAME}")
        self.timer.end("Initializing LLM connection")
    
    def _extract_json(self, text: str) -> Dict:
        """Extract JSON from LLM response"""
        text = re.sub(r'```json|```', '', text, flags=re.IGNORECASE).strip()
        text = text.replace('\\n', '\n').replace('\\t', '\t')
        text = re.sub(r'\\(?!["\\/bfnrtu])', '', text)
        
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            json_str = match.group()
            try:
                return json.loads(json_str.encode().decode('utf-8', errors='ignore'))
            except json.JSONDecodeError:
                pattern_match = re.search(r'"matched_pattern"\s*:\s*"([^"]*)"', json_str)
                confidence_match = re.search(r'"confidence"\s*:\s*"([^"]*)"', json_str)
                reason_match = re.search(r'"reason"\s*:\s*"([^"]*)"', json_str)
                
                if pattern_match and confidence_match and reason_match:
                    return {
                        "matched_pattern": pattern_match.group(1),
                        "confidence": confidence_match.group(1),
                        "reason": reason_match.group(1)
                    }
        
        raise ValueError("Could not extract valid JSON from LLM response")
    
    def match(self, incident: Dict, show_similar: bool = True) -> Dict:
        """
        Match an incident against patterns using ChromaDB.
        
        Args:
            incident: Dictionary with title, description, category, sub_category
            show_similar: Whether to display similar patterns with scores
            
        Returns:
            Dictionary with matched_pattern, confidence, reason, similar_patterns, timing
        """
        total_start = time.time()
        timing = {}
        
        print("\n" + "-"*60)
        print(f"🔍 CLASSIFYING INCIDENT: {incident['title'][:50]}...")
        print("-"*60)
        
        # Step 1: Create incident text
        self.timer.start("Preparing incident text")
        incident_text = f"{incident['title']} {incident['description']} {incident['category']} {incident['sub_category']}"
        self.timer.end("Preparing incident text")
        
        # Step 2: Query ChromaDB for similar patterns
        self.timer.start("Vector similarity search (ChromaDB)")
        indices, similarities, vector_time = self.chroma_store.query_similar(
            incident_text, 
            k=Config.TOP_K_PATTERNS
        )
        timing['vector_time'] = vector_time
        self.timer.end("Vector similarity search (ChromaDB)")
        
        # Step 3: Get filtered patterns
        self.timer.start("Filtering top patterns")
        filtered_patterns = [self.patterns[i] for i in indices]
        similar_patterns_info = []
        
        for i, (idx, sim) in enumerate(zip(indices, similarities)):
            pattern = self.patterns[idx]
            title = pattern.get('pattern_title', pattern.get('pattern_name', f'Pattern {idx}'))
            similar_patterns_info.append({
                "rank": i + 1,
                "pattern_title": title,
                "similarity_score": round(sim, 4)
            })
        
        self.timer.log_step(f"Selected top {len(filtered_patterns)} patterns")
        self.timer.end("Filtering top patterns")
        
        # Step 4: Display similar patterns if requested
        if show_similar:
            print("\n📊 TOP SIMILAR PATTERNS (by cosine similarity):")
            print("-"*50)
            for info in similar_patterns_info[:10]:  # Show top 10
                score_bar = "█" * int(info['similarity_score'] * 20)
                print(f"  {info['rank']:2d}. [{info['similarity_score']:.4f}] {score_bar}")
                print(f"      {info['pattern_title'][:45]}")
            print("-"*50)
        
        # Step 5: Call LLM with filtered patterns
        self.timer.start("LLM inference")
        llm_start = time.time()
        
        pattern_text = json.dumps(filtered_patterns, indent=2)
        prompt = f"""You are a SENIOR ITSM INCIDENT CLASSIFICATION ENGINE.

TASK:
Given an incident and a list of predefined incident patterns,
identify the SINGLE BEST matching pattern.

RULES:
- Choose ONLY one pattern
- Match based on semantic meaning, not keywords only
- Avoid generic matches if a specific one exists
- If nothing matches well, return "No Match"

OUTPUT FORMAT (STRICT JSON ONLY):
{{
  "matched_pattern": "<pattern_name or No Match>",
  "confidence": "<High | Medium | Low>",
  "reason": "Short justification"
}}

INCIDENT:
Title: {incident['title']}
Description: {incident['description']}
Category: {incident['category']}
Sub Category: {incident['sub_category']}

AVAILABLE PATTERNS:
{pattern_text}
"""
        
        response = self.llm.invoke(prompt)
        llm_time = time.time() - llm_start
        timing['llm_time'] = llm_time
        
        self.timer.log_step(f"LLM responded in {llm_time:.2f}s")
        self.timer.end("LLM inference")
        
        # Step 6: Parse response
        self.timer.start("Parsing LLM response")
        result = self._extract_json(response.content)
        self.timer.end("Parsing LLM response")
        
        # Step 7: Calculate total time
        total_time = time.time() - total_start
        timing['total_time'] = total_time
        
        # Add similar patterns and timing to result
        result['similar_patterns'] = similar_patterns_info
        result['timing'] = {
            'vector_search_ms': round(timing['vector_time'] * 1000, 2),
            'llm_inference_ms': round(timing['llm_time'] * 1000, 2),
            'total_ms': round(total_time * 1000, 2)
        }
        
        # Step 8: Save to database
        if self.db:
            self.timer.start("Saving to database")
            self.db.save_match(incident, result, timing)
            self.timer.end("Saving to database")
        
        # Print timing summary
        print(self.timer.summary())
        
        return result


# =====================================================
# INTERACTIVE MODE
# =====================================================

def interactive_mode():
    """Interactive mode with detailed logging"""
    print("\n" + "="*60)
    print("🎯 INCIDENT PATTERN MATCHER - ChromaDB Optimized")
    print("="*60)
    
    matcher = ChromaPatternMatcher()
    
    while True:
        print("\n" + "="*60)
        print("Enter Incident Details (or 'quit' to exit)")
        print("="*60)
        
        try:
            title = input("📝 Title: ").strip()
            if title.lower() == 'quit':
                print("\n👋 Exiting...")
                break
            
            incident = {
                "title": title,
                "description": input("📄 Description: ").strip(),
                "category": input("📁 Category: ").strip(),
                "sub_category": input("📂 Sub Category: ").strip()
            }
            
            # Validate
            if not all(incident.values()):
                print("❌ All fields are required!")
                continue
            
            # Match
            result = matcher.match(incident, show_similar=True)
            
            # Display result
            print("\n" + "="*60)
            print("🎯 CLASSIFICATION RESULT")
            print("="*60)
            print(f"  Matched Pattern: {result['matched_pattern']}")
            print(f"  Confidence:      {result['confidence']}")
            print(f"  Reason:          {result['reason']}")
            print("-"*60)
            print("⏱️  PERFORMANCE:")
            print(f"  Vector Search:   {result['timing']['vector_search_ms']:.2f}ms")
            print(f"  LLM Inference:   {result['timing']['llm_inference_ms']:.2f}ms")
            print(f"  Total Time:      {result['timing']['total_ms']:.2f}ms")
            print("="*60)
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user.")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            logger.exception("Error in interactive mode")
    
    return 0


def batch_mode(input_file: str, output_file: Optional[str] = None):
    """Batch mode with timing for each incident"""
    print("\n" + "="*60)
    print("📦 BATCH PROCESSING MODE")
    print("="*60)
    
    with open(input_file, 'r', encoding='utf-8') as f:
        incidents = json.load(f)
    
    print(f"📋 Loaded {len(incidents)} incidents from {input_file}")
    
    matcher = ChromaPatternMatcher()
    results = []
    total_times = []
    
    for i, incident in enumerate(incidents, 1):
        print(f"\n[{i}/{len(incidents)}] Processing: {incident.get('title', 'Unknown')[:40]}...")
        
        result = matcher.match(incident, show_similar=False)
        results.append({
            "incident": incident,
            "match": result
        })
        total_times.append(result['timing']['total_ms'])
    
    # Save results
    output_path = output_file or f"matches_chroma_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Summary
    print("\n" + "="*60)
    print("📊 BATCH PROCESSING SUMMARY")
    print("="*60)
    print(f"  Total Incidents:    {len(incidents)}")
    print(f"  Average Time:       {sum(total_times)/len(total_times):.2f}ms")
    print(f"  Min Time:           {min(total_times):.2f}ms")
    print(f"  Max Time:           {max(total_times):.2f}ms")
    print(f"  Results saved to:   {output_path}")
    print("="*60)
    
    return 0


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "batch" and len(sys.argv) >= 3:
            exit(batch_mode(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None))
        elif sys.argv[1] == "rebuild":
            # Force rebuild ChromaDB
            print("🔄 Rebuilding ChromaDB index...")
            matcher = ChromaPatternMatcher(force_rebuild=True)
            print("✅ ChromaDB index rebuilt successfully!")
            exit(0)
    
    exit(interactive_mode())
