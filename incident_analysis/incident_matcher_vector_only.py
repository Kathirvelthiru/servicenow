"""
Incident Pattern Matcher - Vector Similarity Only (No LLM)
===========================================================
Ultra-fast incident classification using ONLY vector similarity.
No LLM involved - results in milliseconds instead of seconds!

Performance: ~10-50ms total (100x faster than LLM version)

Supports two modes:
1. Pattern-based: Match against predefined patterns (incident_patterns_final.json)
2. Incident-based: Match against train incidents with embeddings (train_with_embeddings.json)
"""

import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from enum import Enum
import sqlite3
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("Please install: pip install sentence-transformers")

try:
    import chromadb
except ImportError:
    raise ImportError("Please install: pip install chromadb")


# =====================================================
# CONFIGURATION
# =====================================================

class Config:
    """Application configuration"""
    PATTERN_FILE: str = "incident_patterns_final.json"
    TRAIN_EMBEDDINGS_FILE: str = "train_with_embeddings.json"
    LOG_FILE: str = "incident_matcher_vector.log"
    LOG_LEVEL: int = logging.INFO
    DB_FILE: str = "incident_matches_vector.db"
    ENABLE_DB: bool = True
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    TOP_K_RESULTS: int = 5
    CHROMA_PERSIST_DIR: str = "./chroma_patterns_db"
    CHROMA_COLLECTION_NAME: str = "incident_patterns"
    CHROMA_INCIDENTS_COLLECTION: str = "train_incidents"
    
    # Confidence thresholds based on similarity scores
    HIGH_CONFIDENCE_THRESHOLD: float = 0.75
    MEDIUM_CONFIDENCE_THRESHOLD: float = 0.60
    EXACT_MATCH_THRESHOLD: float = 0.95  # For near-exact matches


class ConfidenceLevel(str, Enum):
    """Confidence levels based on similarity scores"""
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


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
            lines.append(f"  {step['operation']}: {step['duration']:.4f}s ({step['duration']*1000:.1f}ms)")
        lines.append("-"*60)
        lines.append(f"  TOTAL TIME: {total:.4f}s ({total*1000:.1f}ms)")
        lines.append("="*60)
        return "\n".join(lines)


# =====================================================
# LOGGING SETUP
# =====================================================

def setup_logging(log_file: str = Config.LOG_FILE, log_level: int = Config.LOG_LEVEL):
    """Configure logging to file and console"""
    logger = logging.getLogger("vector_matcher")
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
        """Initialize ChromaDB with pattern embeddings"""
        total_start = time.time()
        
        self.timer.start("Loading embedding model")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        self.timer.end("Loading embedding model")
        
        self.timer.start("Initializing ChromaDB client")
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.timer.end("Initializing ChromaDB client")
        
        self.timer.start("Checking existing collection")
        existing_collections = [c.name for c in self.client.list_collections()]
        collection_exists = self.collection_name in existing_collections
        
        if collection_exists and not force_rebuild:
            self.collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=None
            )
            existing_count = self.collection.count()
            self.timer.log_step(f"Found existing collection with {existing_count} patterns")
            
            if existing_count == len(patterns):
                self.timer.end("Checking existing collection")
                self.timer.log_step("✓ Using cached embeddings from ChromaDB")
                return time.time() - total_start
        
        self.timer.end("Checking existing collection")
        
        self.timer.start("Creating new collection")
        if collection_exists:
            self.client.delete_collection(self.collection_name)
            self.timer.log_step("Deleted old collection")
        
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.timer.end("Creating new collection")
        
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
        
        self.timer.start("Computing embeddings")
        embeddings = self.embedding_model.encode(pattern_texts, show_progress_bar=False)
        self.timer.log_step(f"Computed embeddings shape: {embeddings.shape}")
        self.timer.end("Computing embeddings")
        
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
    
    def query_similar(self, incident_text: str, k: int = Config.TOP_K_RESULTS) -> Tuple[List[int], List[float], float]:
        """Query ChromaDB for similar patterns"""
        query_start = time.time()
        
        self.timer.start("Computing incident embedding")
        incident_embedding = self.embedding_model.encode([incident_text], show_progress_bar=False)
        self.timer.end("Computing incident embedding")
        
        self.timer.start("Querying ChromaDB")
        results = self.collection.query(
            query_embeddings=incident_embedding.tolist(),
            n_results=k,
            include=["metadatas", "distances", "documents"]
        )
        self.timer.end("Querying ChromaDB")
        
        indices = [m["index"] for m in results["metadatas"][0]]
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
        """Initialize database schema"""
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
                    similarity_score REAL,
                    vector_query_time REAL,
                    total_time REAL
                )
            """)
            conn.commit()
    
    def save_match(self, incident: Dict, result: Dict, times: Dict):
        """Save match result to database"""
        with sqlite3.connect(self.db_file) as conn:
            conn.execute("""
                INSERT INTO incident_matches
                (timestamp, incident_title, incident_description, 
                 incident_category, incident_sub_category,
                 matched_pattern, confidence, similarity_score,
                 vector_query_time, total_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                incident.get('title', ''),
                incident.get('description', ''),
                incident.get('category', ''),
                incident.get('sub_category', ''),
                result.get('matched_pattern', ''),
                result.get('confidence', ''),
                result.get('similarity_score', 0.0),
                times.get('vector_time', 0),
                times.get('total_time', 0)
            ))
            conn.commit()


# =====================================================
# VECTOR-ONLY PATTERN MATCHER
# =====================================================

class VectorOnlyMatcher:
    """Ultra-fast pattern matching using ONLY vector similarity (no LLM)"""
    
    def __init__(self, pattern_file: str = Config.PATTERN_FILE,
                 enable_db: bool = Config.ENABLE_DB,
                 force_rebuild: bool = False):
        self.pattern_file = pattern_file
        self.patterns = None
        self.db = None
        self.chroma_store = None
        self.timer = TimingLogger("VectorMatcher")
        
        print("\n" + "="*60)
        print("🚀 INITIALIZING VECTOR-ONLY MATCHER (NO LLM)")
        print("="*60)
        
        init_start = time.time()
        
        self._load_patterns()
        self._init_chroma(force_rebuild)
        
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
    
    def _get_confidence(self, similarity_score: float) -> str:
        """Determine confidence level based on similarity score"""
        if similarity_score >= Config.HIGH_CONFIDENCE_THRESHOLD:
            return ConfidenceLevel.HIGH
        elif similarity_score >= Config.MEDIUM_CONFIDENCE_THRESHOLD:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW
    
    def match(self, incident: Dict, return_top_k: int = Config.TOP_K_RESULTS) -> Dict:
        """
        Match incident using ONLY vector similarity (no LLM).
        Returns top K matches instantly.
        
        Args:
            incident: Dictionary with title, description, category, sub_category
            return_top_k: Number of top matches to return (default 5)
            
        Returns:
            Dictionary with top matches, timing, and confidence
        """
        total_start = time.time()
        timing = {}
        
        print("\n" + "-"*60)
        print(f"🔍 MATCHING INCIDENT: {incident['title'][:50]}...")
        print("-"*60)
        
        # Step 1: Create incident text
        self.timer.start("Preparing incident text")
        incident_text = f"{incident['title']} {incident['description']} {incident['category']} {incident['sub_category']}"
        self.timer.end("Preparing incident text")
        
        # Step 2: Query ChromaDB for similar patterns
        self.timer.start("Vector similarity search (ChromaDB)")
        indices, similarities, vector_time = self.chroma_store.query_similar(
            incident_text, 
            k=return_top_k
        )
        timing['vector_time'] = vector_time
        self.timer.end("Vector similarity search (ChromaDB)")
        
        # Step 3: Build results
        self.timer.start("Building results")
        top_matches = []
        
        for i, (idx, sim) in enumerate(zip(indices, similarities)):
            pattern = self.patterns[idx]
            title = pattern.get('pattern_title', pattern.get('pattern_name', f'Pattern {idx}'))
            incident_id = pattern.get('incident_ID', idx)  # Use incident_ID from pattern, fallback to index
            
            top_matches.append({
                "rank": i + 1,
                "incident_ID": incident_id,
                "pattern_title": title,
                "similarity_score": round(sim, 4),
                "confidence": self._get_confidence(sim),
                "pattern_details": pattern
            })
        
        self.timer.end("Building results")
        
        # Step 4: Display results
        print("\n📊 TOP MATCHING PATTERNS (by cosine similarity):")
        print("-"*60)
        for match in top_matches:
            score_bar = "█" * int(match['similarity_score'] * 20)
            conf_emoji = "🟢" if match['confidence'] == ConfidenceLevel.HIGH else "🟡" if match['confidence'] == ConfidenceLevel.MEDIUM else "🔴"
            print(f"  {match['rank']}. [{match['similarity_score']:.4f}] {score_bar} {conf_emoji}")
            print(f"     {match['pattern_title'][:55]}")
        print("-"*60)
        
        # Step 5: Calculate total time
        total_time = time.time() - total_start
        timing['total_time'] = total_time
        
        # Build final result
        result = {
            "matched_pattern": top_matches[0]['pattern_title'],
            "confidence": top_matches[0]['confidence'],
            "similarity_score": top_matches[0]['similarity_score'],
            "top_matches": top_matches,
            "timing": {
                'vector_search_ms': round(timing['vector_time'] * 1000, 2),
                'total_ms': round(total_time * 1000, 2)
            }
        }
        
        # Step 6: Save to database
        if self.db:
            self.timer.start("Saving to database")
            self.db.save_match(incident, result, timing)
            self.timer.end("Saving to database")
        
        # Print timing summary
        print(self.timer.summary())
        
        return result


# =====================================================
# INCIDENT-BASED MATCHER (Using Train Embeddings)
# =====================================================

class IncidentMatcher:
    """Match incidents against train set using pre-computed embeddings"""
    
    def __init__(self, train_file: str = Config.TRAIN_EMBEDDINGS_FILE,
                 embedding_model: str = Config.EMBEDDING_MODEL):
        self.train_file = train_file
        self.train_incidents = None
        self.train_embeddings = None
        self.embedding_model = None
        self.embedding_model_name = embedding_model
        self.timer = TimingLogger("IncidentMatcher")
        
        print("\n" + "="*60)
        print("🚀 INITIALIZING INCIDENT-BASED MATCHER")
        print("="*60)
        
        self._load_train_data()
        self._load_embedding_model()
        
        print("✅ Incident Matcher ready!")
        print("="*60 + "\n")
    
    def _load_train_data(self):
        """Load train incidents with pre-computed embeddings"""
        self.timer.start("Loading train data with embeddings")
        
        if not Path(self.train_file).exists():
            raise FileNotFoundError(f"Train file not found: {self.train_file}")
        
        with open(self.train_file, "r", encoding="utf-8") as f:
            self.train_incidents = json.load(f)
        
        # Extract embeddings as numpy array for fast computation
        self.train_embeddings = np.array([inc['embedding'] for inc in self.train_incidents])
        
        self.timer.log_step(f"Loaded {len(self.train_incidents)} train incidents")
        self.timer.log_step(f"Embeddings shape: {self.train_embeddings.shape}")
        self.timer.end("Loading train data with embeddings")
    
    def _load_embedding_model(self):
        """Load SentenceTransformer model for encoding new incidents"""
        self.timer.start("Loading embedding model")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        self.timer.end("Loading embedding model")
    
    def _cosine_similarity(self, query_embedding: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and all train embeddings"""
        # Normalize vectors
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        train_norms = self.train_embeddings / np.linalg.norm(self.train_embeddings, axis=1, keepdims=True)
        
        # Compute cosine similarity
        similarities = np.dot(train_norms, query_norm)
        return similarities
    
    def _get_confidence(self, similarity_score: float) -> str:
        """Determine confidence level based on similarity score"""
        if similarity_score >= Config.EXACT_MATCH_THRESHOLD:
            return ConfidenceLevel.HIGH
        elif similarity_score >= Config.HIGH_CONFIDENCE_THRESHOLD:
            return ConfidenceLevel.HIGH
        elif similarity_score >= Config.MEDIUM_CONFIDENCE_THRESHOLD:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW
    
    def match(self, incident: Dict, return_top_k: int = Config.TOP_K_RESULTS) -> Dict:
        """
        Match incident against train set using embedding similarity.
        
        Args:
            incident: Dictionary with short_description, description, etc.
            return_top_k: Number of top matches to return
            
        Returns:
            Dictionary with top matches, timing, and confidence
        """
        total_start = time.time()
        timing = {}
        
        # Create context from incident
        short_desc = incident.get('short_description', incident.get('title', ''))
        desc = incident.get('description', '')
        context = f"{short_desc} {desc}".strip()
        
        print("\n" + "-"*60)
        print(f"🔍 MATCHING INCIDENT: {short_desc[:50]}...")
        print("-"*60)
        
        # Generate embedding for query incident
        self.timer.start("Generating query embedding")
        query_embedding = self.embedding_model.encode([context], show_progress_bar=False)[0]
        embed_time = self.timer.end("Generating query embedding")
        timing['embedding_time'] = embed_time
        
        # Compute similarities
        self.timer.start("Computing cosine similarities")
        similarities = self._cosine_similarity(query_embedding)
        sim_time = self.timer.end("Computing cosine similarities")
        timing['similarity_time'] = sim_time
        
        # Get top K matches
        self.timer.start("Ranking results")
        top_indices = np.argsort(similarities)[::-1][:return_top_k]
        
        top_matches = []
        for rank, idx in enumerate(top_indices, 1):
            train_inc = self.train_incidents[idx]
            sim_score = float(similarities[idx])
            
            top_matches.append({
                "rank": rank,
                "number": train_inc.get('number', ''),
                "short_description": train_inc.get('short_description', ''),
                "problem_id": train_inc.get('problem_id', ''),
                "category": train_inc.get('category', ''),
                "subcategory": train_inc.get('subcategory', ''),
                "similarity_score": round(sim_score, 4),
                "confidence": self._get_confidence(sim_score),
                "is_exact_match": sim_score >= Config.EXACT_MATCH_THRESHOLD
            })
        
        self.timer.end("Ranking results")
        
        # Display results
        print("\n📊 TOP MATCHING INCIDENTS:")
        print("-"*60)
        for match in top_matches:
            score_bar = "█" * int(match['similarity_score'] * 20)
            conf_emoji = "🟢" if match['confidence'] == ConfidenceLevel.HIGH else "🟡" if match['confidence'] == ConfidenceLevel.MEDIUM else "🔴"
            exact = " ⭐EXACT" if match['is_exact_match'] else ""
            print(f"  {match['rank']}. [{match['similarity_score']:.4f}] {score_bar} {conf_emoji}{exact}")
            print(f"     {match['number']}: {match['short_description'][:50]}")
            print(f"     Problem ID: {match['problem_id']}")
        print("-"*60)
        
        # Calculate total time
        total_time = time.time() - total_start
        timing['total_time'] = total_time
        
        result = {
            "matched_incident": top_matches[0] if top_matches else None,
            "confidence": top_matches[0]['confidence'] if top_matches else ConfidenceLevel.LOW,
            "similarity_score": top_matches[0]['similarity_score'] if top_matches else 0.0,
            "is_exact_match": top_matches[0]['is_exact_match'] if top_matches else False,
            "top_matches": top_matches,
            "timing": {
                'embedding_ms': round(timing.get('embedding_time', 0) * 1000, 2),
                'similarity_ms': round(timing.get('similarity_time', 0) * 1000, 2),
                'total_ms': round(total_time * 1000, 2)
            }
        }
        
        print(self.timer.summary())
        
        return result


# =====================================================
# INTERACTIVE MODE
# =====================================================

def interactive_mode():
    """Interactive mode with detailed logging"""
    print("\n" + "="*60)
    print("⚡ VECTOR-ONLY MATCHER - No LLM, Instant Results!")
    print("="*60)
    
    matcher = VectorOnlyMatcher()
    
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
            
            if not all(incident.values()):
                print("❌ All fields are required!")
                continue
            
            # Match
            result = matcher.match(incident, return_top_k=5)
            
            # Display result
            print("\n" + "="*60)
            print("🎯 TOP 5 MATCHES")
            print("="*60)
            for match in result['top_matches']:
                print(f"{match['rank']}. {match['pattern_title']}")
                print(f"   Similarity: {match['similarity_score']:.4f} | Confidence: {match['confidence']}")
            print("-"*60)
            print(f"⏱️  Total Time: {result['timing']['total_ms']:.2f}ms")
            print("="*60)
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user.")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            logger.exception("Error in interactive mode")
    
    return 0


def incident_match_mode():
    """Interactive mode for incident-based matching"""
    print("\n" + "="*60)
    print("⚡ INCIDENT MATCHER - Match against Train Set!")
    print("="*60)
    
    matcher = IncidentMatcher()
    
    while True:
        print("\n" + "="*60)
        print("Enter Incident Details (or 'quit' to exit)")
        print("="*60)
        
        try:
            short_desc = input("📝 Short Description: ").strip()
            if short_desc.lower() == 'quit':
                print("\n👋 Exiting...")
                break
            
            incident = {
                "short_description": short_desc,
                "description": input("📄 Description: ").strip(),
            }
            
            if not incident['short_description']:
                print("❌ Short description is required!")
                continue
            
            # Match
            result = matcher.match(incident, return_top_k=5)
            
            # Display result
            print("\n" + "="*60)
            print("🎯 TOP 5 MATCHING INCIDENTS")
            print("="*60)
            for match in result['top_matches']:
                exact = " ⭐EXACT MATCH" if match['is_exact_match'] else ""
                print(f"{match['rank']}. {match['number']}: {match['short_description'][:50]}{exact}")
                print(f"   Problem ID: {match['problem_id']} | Similarity: {match['similarity_score']:.4f}")
            print("-"*60)
            print(f"⏱️  Total Time: {result['timing']['total_ms']:.2f}ms")
            print("="*60)
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user.")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            logger.exception("Error in incident match mode")
    
    return 0


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "rebuild":
            print("🔄 Rebuilding ChromaDB index...")
            matcher = VectorOnlyMatcher(force_rebuild=True)
            print("✅ ChromaDB index rebuilt successfully!")
            exit(0)
        elif sys.argv[1] == "incident":
            exit(incident_match_mode())
    
    exit(interactive_mode())
