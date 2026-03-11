"""
Embedding Generation Script
=============================
Generate embeddings for train and test incidents using SentenceTransformer
"""

import json
from pathlib import Path
from typing import List
import time

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("Please install: pip install sentence-transformers")


class EmbeddingGenerator:
    """Generate embeddings for incident context using SentenceTransformer"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"⏳ Loading embedding model: {model_name}")
        start = time.time()
        self.model = SentenceTransformer(model_name)
        print(f"✅ Model loaded in {time.time() - start:.2f}s")
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        print(f"⏳ Generating embeddings for {len(texts)} texts...")
        start = time.time()
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"✅ Embeddings generated in {time.time() - start:.2f}s")
        return embeddings.tolist()


def main():
    print("=" * 60)
    print("🔢 EMBEDDING GENERATION SCRIPT")
    print("=" * 60)
    
    output_dir = Path(__file__).parent
    
    # Load train and test incidents
    train_json = output_dir / 'train_incidents.json'
    test_json = output_dir / 'test_incidents.json'
    
    if not train_json.exists():
        print("❌ train_incidents.json not found! Run prepare_data.py first.")
        return
    
    with open(train_json, 'r', encoding='utf-8') as f:
        train_incidents = json.load(f)
    
    with open(test_json, 'r', encoding='utf-8') as f:
        test_incidents = json.load(f)
    
    print(f"\n📂 Loaded {len(train_incidents)} train incidents")
    print(f"📂 Loaded {len(test_incidents)} test incidents")
    
    # Initialize embedding generator
    generator = EmbeddingGenerator()
    
    # Generate embeddings for train set
    print("\n📊 Processing TRAIN set...")
    train_contexts = [inc['context'] for inc in train_incidents]
    train_embeddings = generator.generate_embeddings(train_contexts)
    
    for i, inc in enumerate(train_incidents):
        inc['embedding'] = train_embeddings[i]
    
    # Generate embeddings for test set
    print("\n📊 Processing TEST set...")
    test_contexts = [inc['context'] for inc in test_incidents]
    test_embeddings = generator.generate_embeddings(test_contexts)
    
    for i, inc in enumerate(test_incidents):
        inc['embedding'] = test_embeddings[i]
    
    # Save with embeddings
    train_emb_json = output_dir / 'train_with_embeddings.json'
    test_emb_json = output_dir / 'test_with_embeddings.json'
    
    with open(train_emb_json, 'w', encoding='utf-8') as f:
        json.dump(train_incidents, f, indent=2)
    
    with open(test_emb_json, 'w', encoding='utf-8') as f:
        json.dump(test_incidents, f, indent=2)
    
    print(f"\n✅ Saved: {train_emb_json}")
    print(f"✅ Saved: {test_emb_json}")
    
    print("\n" + "=" * 60)
    print("✅ EMBEDDING GENERATION COMPLETE")
    print("=" * 60)
    print(f"\nEmbedding dimension: {len(train_embeddings[0])}")


if __name__ == "__main__":
    main()
