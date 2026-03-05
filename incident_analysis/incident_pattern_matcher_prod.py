"""
Incident Pattern Matcher - Production Version
============================================
Classifies incidents against predefined patterns using LLM.

Features:
- Error handling and retry logic
- Logging and audit trails
- Configuration management
- Database persistence
- Batch processing support
- CLI and programmatic interfaces
"""

import json
import re
import csv
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from enum import Enum
import sqlite3
import numpy as np

try:
    from langchain_ollama import ChatOllama
except ImportError:
    raise ImportError("Please install langchain_ollama: pip install langchain-ollama")

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    raise ImportError("Please install: pip install sentence-transformers scikit-learn")

try:
    import click
except ImportError:
    click = None


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
    LOG_FILE: str = "incident_matcher.log"
    LOG_LEVEL: str = logging.INFO
    DB_FILE: str = "incident_matches.db"
    ENABLE_DB: bool = True
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    TOP_K_PATTERNS: int = 15
    ENABLE_SIMILARITY_FILTER: bool = True


class ConfidenceLevel(str, Enum):
    """Confidence levels for pattern matches"""
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"
    NO_MATCH = "No Match"


# =====================================================
# LOGGING SETUP
# =====================================================

def setup_logging(log_file: str = Config.LOG_FILE, log_level = Config.LOG_LEVEL):
    """Configure logging to file and console"""
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


logger = setup_logging()


# =====================================================
# DATABASE SETUP
# =====================================================

class IncidentDatabase:
    """Manages persistence of incident match results"""
    
    def __init__(self, db_file: str = Config.DB_FILE):
        self.db_file = db_file
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema"""
        try:
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
                        llm_response_time REAL
                    )
                """)
                conn.commit()
            logger.info(f"Database initialized: {self.db_file}")
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    def save_match(self, incident: Dict, result: Dict, response_time: float):
        """Save match result to database"""
        try:
            with sqlite3.connect(self.db_file) as conn:
                conn.execute("""
                    INSERT INTO incident_matches
                    (timestamp, incident_title, incident_description, 
                     incident_category, incident_sub_category,
                     matched_pattern, confidence, reason, llm_response_time)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    datetime.now().isoformat(),
                    incident.get('title', ''),
                    incident.get('description', ''),
                    incident.get('category', ''),
                    incident.get('sub_category', ''),
                    result.get('matched_pattern', ''),
                    result.get('confidence', ''),
                    result.get('reason', ''),
                    response_time
                ))
                conn.commit()
            logger.debug(f"Match saved to database: {incident['title']}")
        except Exception as e:
            logger.error(f"Failed to save match to database: {e}")


# =====================================================
# INCIDENT PATTERN MATCHER
# =====================================================

class PatternMatcher:
    """Main incident pattern matching engine"""
    
    def __init__(self, pattern_file: str = Config.PATTERN_FILE, 
                 enable_db: bool = Config.ENABLE_DB):
        """
        Initialize the matcher.
        
        Args:
            pattern_file: Path to JSON file with patterns
            enable_db: Whether to persist results to database
        """
        self.pattern_file = pattern_file
        self.patterns = None
        self.llm = None
        self.db = None
        self.embedding_model = None
        self.pattern_embeddings = None
        self.pattern_texts = None
        
        self._load_patterns()
        self._init_llm()
        
        if Config.ENABLE_SIMILARITY_FILTER:
            self._init_embeddings()
        
        if enable_db:
            self.db = IncidentDatabase()
    
    def _load_patterns(self):
        """Load patterns from JSON file"""
        try:
            if not Path(self.pattern_file).exists():
                raise FileNotFoundError(f"Pattern file not found: {self.pattern_file}")
            
            with open(self.pattern_file, "r", encoding="utf-8") as f:
                self.patterns = json.load(f)
            
            if not isinstance(self.patterns, list):
                raise ValueError("Patterns must be a JSON array")
            
            logger.info(f"Loaded {len(self.patterns)} patterns from {self.pattern_file}")
        except Exception as e:
            logger.error(f"Failed to load patterns: {e}")
            raise
    
    def _init_llm(self):
        """Initialize LLM connection"""
        try:
            self.llm = ChatOllama(
                model=Config.MODEL_NAME,
                temperature=Config.LLM_TEMPERATURE,
                base_url=Config.LLM_BASE_URL,
                timeout=Config.REQUEST_TIMEOUT
            )
            logger.info(f"LLM initialized: {Config.MODEL_NAME}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise
    
    def _init_embeddings(self):
        """Initialize embedding model and pre-compute pattern embeddings"""
        try:
            logger.info(f"Loading embedding model: {Config.EMBEDDING_MODEL}")
            self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
            
            # Pre-compute embeddings for all patterns
            self.pattern_texts = []
            for pattern in self.patterns:
                # Combine pattern fields for better semantic matching
                pattern_text = f"{pattern.get('pattern_title', '')} {pattern.get('pattern_name', '')} "
                pattern_text += " ".join(pattern.get('belongs_when', []))
                self.pattern_texts.append(pattern_text)
            
            logger.info(f"Computing embeddings for {len(self.pattern_texts)} patterns...")
            self.pattern_embeddings = self.embedding_model.encode(self.pattern_texts, show_progress_bar=False)
            logger.info(f"Embeddings initialized. Shape: {self.pattern_embeddings.shape}")
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            logger.warning("Falling back to using all patterns without filtering")
            self.embedding_model = None
    
    def _validate_incident(self, incident: Dict) -> Tuple[bool, Optional[str]]:
        """
        Validate incident input.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        required_fields = ['title', 'description', 'category', 'sub_category']
        
        for field in required_fields:
            if field not in incident:
                return False, f"Missing required field: {field}"
            
            if not isinstance(incident[field], str):
                return False, f"Field '{field}' must be a string"
            
            if not incident[field].strip():
                return False, f"Field '{field}' cannot be empty"
        
        return True, None
    
    def _extract_json(self, text: str) -> Dict:
        """
        Robustly extract JSON from LLM response.
        
        Args:
            text: LLM response text
            
        Returns:
            Parsed JSON object
            
        Raises:
            ValueError: If valid JSON cannot be extracted
        """
        # Remove markdown code blocks
        text = re.sub(r'```json|```', '', text, flags=re.IGNORECASE).strip()
        
        # Fix common escape issues
        text = text.replace('\\n', '\n')
        text = text.replace('\\t', '\t')
        text = re.sub(r'\\(?!["\\/bfnrtu])', '', text)  # Remove invalid backslashes
        
        # Try to find JSON object
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            json_str = match.group()
            try:
                # Clean any problematic characters
                json_str = json_str.encode().decode('utf-8', errors='ignore')
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse error: {e}")
                # Last resort: try to extract just the fields we need
                try:
                    pattern_match = re.search(r'"matched_pattern"\s*:\s*"([^"]*)"', json_str)
                    confidence_match = re.search(r'"confidence"\s*:\s*"([^"]*)"', json_str)
                    reason_match = re.search(r'"reason"\s*:\s*"([^"]*)"', json_str)
                    
                    if pattern_match and confidence_match and reason_match:
                        return {
                            "matched_pattern": pattern_match.group(1),
                            "confidence": confidence_match.group(1),
                            "reason": reason_match.group(1)
                        }
                except Exception:
                    pass
        
        raise ValueError("Could not extract valid JSON from LLM response")
    
    def _get_top_k_patterns(self, incident: Dict, k: int = Config.TOP_K_PATTERNS) -> List[Dict]:
        """
        Filter patterns using cosine similarity to get top K most relevant patterns.
        
        Args:
            incident: Incident details
            k: Number of top patterns to return
            
        Returns:
            List of top K most similar patterns
        """
        if self.embedding_model is None or self.pattern_embeddings is None:
            logger.debug("Embedding model not available, using all patterns")
            return self.patterns
        
        try:
            # Create incident text for embedding
            incident_text = f"{incident['title']} {incident['description']} {incident['category']} {incident['sub_category']}"
            
            # Compute incident embedding
            incident_embedding = self.embedding_model.encode([incident_text], show_progress_bar=False)
            
            # Calculate cosine similarities
            similarities = cosine_similarity(incident_embedding, self.pattern_embeddings)[0]
            
            # Get top K indices
            top_k_indices = np.argsort(similarities)[-k:][::-1]
            
            # Get top K patterns with their similarity scores
            top_patterns = [self.patterns[i] for i in top_k_indices]
            top_scores = [similarities[i] for i in top_k_indices]
            
            logger.info(f"Filtered to top {k} patterns. Similarity scores: {[f'{s:.3f}' for s in top_scores[:5]]}...")
            
            return top_patterns
            
        except Exception as e:
            logger.error(f"Error in pattern filtering: {e}")
            logger.warning("Falling back to using all patterns")
            return self.patterns
    
    def match(self, incident: Dict) -> Dict:
        """
        Match an incident against patterns.
        
        Args:
            incident: Dictionary with title, description, category, sub_category
            
        Returns:
            Dictionary with matched_pattern, confidence, reason
        """
        # Validate input
        is_valid, error_msg = self._validate_incident(incident)
        if not is_valid:
            logger.error(f"Incident validation failed: {error_msg}")
            return {
                "matched_pattern": "No Match",
                "confidence": ConfidenceLevel.NO_MATCH,
                "reason": f"Validation error: {error_msg}"
            }
        
        try:
            # Attempt matching with retry logic
            for attempt in range(Config.RETRY_LIMIT):
                try:
                    import time
                    start_time = time.time()
                    
                    result = self._call_llm(incident)
                    
                    response_time = time.time() - start_time
                    logger.info(
                        f"Match successful for '{incident['title']}' "
                        f"→ {result['matched_pattern']} ({response_time:.2f}s)"
                    )
                    
                    # Save to database
                    if self.db:
                        self.db.save_match(incident, result, response_time)
                    
                    return result
                
                except ValueError as e:
                    if attempt < Config.RETRY_LIMIT - 1:
                        logger.warning(
                            f"JSON extraction failed (attempt {attempt + 1}/"
                            f"{Config.RETRY_LIMIT}): {e}"
                        )
                        continue
                    else:
                        raise
            
        except Exception as e:
            logger.error(f"Pattern matching failed: {e}")
            return {
                "matched_pattern": "No Match",
                "confidence": ConfidenceLevel.NO_MATCH,
                "reason": f"Matching error: {str(e)}"
            }
    
    def _call_llm(self, incident: Dict) -> Dict:
        """
        Call LLM to find matching pattern.
        
        Args:
            incident: Incident details
            
        Returns:
            Match result with confidence and reason
        """
        # Filter patterns using cosine similarity if enabled
        if Config.ENABLE_SIMILARITY_FILTER:
            filtered_patterns = self._get_top_k_patterns(incident)
            logger.debug(f"Using {len(filtered_patterns)} filtered patterns instead of {len(self.patterns)}")
        else:
            filtered_patterns = self.patterns
        
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
        result = self._extract_json(response.content)
        
        # Validate result fields
        required_fields = ['matched_pattern', 'confidence', 'reason']
        if not all(field in result for field in required_fields):
            raise ValueError("LLM response missing required fields")
        
        return result


# =====================================================
# CLI INTERFACE
# =====================================================

def interactive_mode():
    """Interactive mode - prompt user for incident details"""
    print("\n" + "="*60)
    print("INCIDENT PATTERN MATCHER - Interactive Mode")
    print("="*60)
    
    try:
        matcher = PatternMatcher()
        
        while True:
            print("\nEnter Incident Details (or 'quit' to exit)")
            print("-" * 40)
            
            try:
                incident = {
                    "title": input("Title: ").strip(),
                    "description": input("Description: ").strip(),
                    "category": input("Category: ").strip(),
                    "sub_category": input("Sub Category: ").strip()
                }
                
                if incident['title'].lower() == 'quit':
                    print("Exiting...")
                    break
                
                print("\n⏳ Matching incident with LLM...")
                result = matcher.match(incident)
                
                print("\n" + "="*60)
                print("MATCH RESULT")
                print("="*60)
                print(json.dumps(result, indent=2))
                print("="*60)
                
            except KeyboardInterrupt:
                print("\n\nInterrupted by user.")
                break
    
    except Exception as e:
        logger.error(f"Interactive mode failed: {e}")
        print(f"\n❌ Error: {e}")
        return 1
    
    return 0


def batch_mode(input_file: str, output_file: Optional[str] = None):
    """Batch mode - process incidents from JSON file"""
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            incidents = json.load(f)
        
        if not isinstance(incidents, list):
            raise ValueError("Input file must contain a JSON array of incidents")
        
        logger.info(f"Processing {len(incidents)} incidents from {input_file}")
        
        matcher = PatternMatcher()
        results = []
        
        for i, incident in enumerate(incidents, 1):
            print(f"Processing incident {i}/{len(incidents)}...", end='\r')
            result = matcher.match(incident)
            results.append({
                "incident": incident,
                "match": result
            })
        
        # Save results
        output_path = output_file or f"matches_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {output_path}")
        print(f"\n✅ Processed {len(incidents)} incidents. Results saved to {output_path}")
        
        return 0
    
    except Exception as e:
        logger.error(f"Batch mode failed: {e}")
        print(f"\n❌ Error: {e}")
        return 1


# =====================================================
# CLI SETUP (with Click)
# =====================================================

if click:
    @click.group()
    def cli():
        """Incident Pattern Matcher CLI"""
        pass
    
    @cli.command()
    def interactive():
        """Run in interactive mode"""
        exit(interactive_mode())
    
    @cli.command()
    @click.option('--input', '-i', required=True, help='Input JSON file with incidents')
    @click.option('--output', '-o', help='Output file for results')
    def batch(input, output):
        """Run in batch mode"""
        exit(batch_mode(input, output))
    
    if __name__ == "__main__":
        cli()

else:
    # Fallback without Click
    if __name__ == "__main__":
        import sys
        
        if len(sys.argv) > 1 and sys.argv[1] == "batch":
            if len(sys.argv) < 3:
                print("Usage: python script.py batch <input_file> [output_file]")
                exit(1)
            exit(batch_mode(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None))
        else:
            exit(interactive_mode())
