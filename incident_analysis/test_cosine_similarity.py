"""
Test script to verify cosine similarity filtering is working
"""
import json
from incident_pattern_matcher_prod import PatternMatcher

if __name__ == "__main__":
    print("=" * 60)
    print("TESTING COSINE SIMILARITY PATTERN FILTERING")
    print("=" * 60)
    
    # Initialize matcher (this will load embeddings)
    print("\nInitializing PatternMatcher...")
    matcher = PatternMatcher()
    
    # Test incident
    test_incident = {
        "title": "User unable to login",
        "description": "Users are reporting they cannot access the banking system",
        "category": "Banking",
        "sub_category": "Authentication"
    }
    
    print(f"\nTest Incident:")
    print(json.dumps(test_incident, indent=2))
    
    # Get top patterns using cosine similarity
    print("\nFiltering patterns using cosine similarity...")
    top_patterns = matcher._get_top_k_patterns(test_incident, k=15)
    
    print(f"\nFiltered from {len(matcher.patterns)} patterns to {len(top_patterns)} patterns")
    print("\nTop 5 most similar patterns:")
    for i, pattern in enumerate(top_patterns[:5], 1):
        print(f"{i}. {pattern.get('pattern_title', pattern.get('pattern_name', 'Unknown'))}")
    
    # Now test full matching
    print("\n" + "=" * 60)
    print("TESTING FULL MATCH WITH LLM")
    print("=" * 60)
    
    result = matcher.match(test_incident)
    
    print("\nMatch Result:")
    print(json.dumps(result, indent=2))
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
