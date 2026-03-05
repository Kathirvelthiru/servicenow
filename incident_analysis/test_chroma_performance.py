"""
Test script to demonstrate ChromaDB performance improvements
============================================================
Compares old cosine similarity approach vs new ChromaDB approach
"""

import time
import json
from incident_matcher_chroma import ChromaPatternMatcher

def test_single_incident():
    """Test single incident classification with detailed output"""
    print("\n" + "="*70)
    print("🧪 TEST: SINGLE INCIDENT CLASSIFICATION WITH CHROMADB")
    print("="*70)
    
    # Initialize matcher
    matcher = ChromaPatternMatcher()
    
    # Test incident
    test_incident = {
        "title": "User unable to login to banking portal",
        "description": "Multiple users reporting login failures with error code AUTH-500",
        "category": "Banking",
        "sub_category": "Authentication"
    }
    
    print("\n📋 TEST INCIDENT:")
    print(json.dumps(test_incident, indent=2))
    
    # Run classification
    result = matcher.match(test_incident, show_similar=True)
    
    # Display final result
    print("\n" + "="*70)
    print("🎯 FINAL CLASSIFICATION RESULT")
    print("="*70)
    print(json.dumps(result, indent=2))
    
    return result


def test_multiple_incidents():
    """Test multiple incidents to show consistent performance"""
    print("\n" + "="*70)
    print("🧪 TEST: MULTIPLE INCIDENTS PERFORMANCE")
    print("="*70)
    
    test_incidents = [
        {
            "title": "Database connection timeout",
            "description": "Application unable to connect to database during peak hours",
            "category": "Database",
            "sub_category": "Connectivity"
        },
        {
            "title": "Permission denied accessing reports",
            "description": "User cannot access financial reports, getting access denied error",
            "category": "Banking",
            "sub_category": "Authorization"
        },
        {
            "title": "API error in payment gateway",
            "description": "Payment transactions failing with API timeout errors",
            "category": "Banking",
            "sub_category": "Payments"
        },
        {
            "title": "Server not responding",
            "description": "Production server showing high latency and timeouts",
            "category": "Infrastructure",
            "sub_category": "Server"
        },
        {
            "title": "Network connectivity issue",
            "description": "Intermittent network drops affecting multiple users",
            "category": "Network",
            "sub_category": "Connectivity"
        }
    ]
    
    # Initialize matcher (one-time cost)
    print("\n⏱️  Initializing matcher (one-time cost)...")
    init_start = time.time()
    matcher = ChromaPatternMatcher()
    init_time = time.time() - init_start
    print(f"✅ Initialization completed in {init_time:.2f}s\n")
    
    # Process each incident
    results = []
    times = []
    
    for i, incident in enumerate(test_incidents, 1):
        print(f"\n{'='*70}")
        print(f"📝 INCIDENT {i}/{len(test_incidents)}: {incident['title']}")
        print("="*70)
        
        result = matcher.match(incident, show_similar=True)
        results.append(result)
        times.append(result['timing']['total_ms'])
        
        print(f"\n✅ Result: {result['matched_pattern']} ({result['confidence']})")
    
    # Summary
    print("\n" + "="*70)
    print("📊 PERFORMANCE SUMMARY")
    print("="*70)
    print(f"  Incidents Processed:  {len(test_incidents)}")
    print(f"  Initialization Time:  {init_time*1000:.2f}ms")
    print(f"  Average Match Time:   {sum(times)/len(times):.2f}ms")
    print(f"  Fastest Match:        {min(times):.2f}ms")
    print(f"  Slowest Match:        {max(times):.2f}ms")
    print("-"*70)
    print("  BREAKDOWN (Average):")
    avg_vector = sum(r['timing']['vector_search_ms'] for r in results) / len(results)
    avg_llm = sum(r['timing']['llm_inference_ms'] for r in results) / len(results)
    print(f"    Vector Search:      {avg_vector:.2f}ms")
    print(f"    LLM Inference:      {avg_llm:.2f}ms")
    print("="*70)
    
    return results


def compare_with_without_chroma():
    """Compare performance with and without ChromaDB caching"""
    print("\n" + "="*70)
    print("🧪 TEST: CHROMADB CACHING BENEFIT")
    print("="*70)
    
    test_incident = {
        "title": "Transaction failed during checkout",
        "description": "Customer payment not processing, showing error on checkout page",
        "category": "Banking",
        "sub_category": "Transactions"
    }
    
    # First run - may need to build index
    print("\n🔄 First run (may include index building)...")
    matcher1 = ChromaPatternMatcher(force_rebuild=True)
    result1 = matcher1.match(test_incident, show_similar=False)
    time1 = result1['timing']['total_ms']
    
    # Second run - uses cached index
    print("\n🚀 Second run (using cached ChromaDB index)...")
    matcher2 = ChromaPatternMatcher(force_rebuild=False)
    result2 = matcher2.match(test_incident, show_similar=False)
    time2 = result2['timing']['total_ms']
    
    print("\n" + "="*70)
    print("📊 CACHING COMPARISON")
    print("="*70)
    print(f"  First Run (with index build):   {time1:.2f}ms")
    print(f"  Second Run (cached index):      {time2:.2f}ms")
    print(f"  Speedup:                        {time1/time2:.2f}x faster")
    print("="*70)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "single":
            test_single_incident()
        elif sys.argv[1] == "multi":
            test_multiple_incidents()
        elif sys.argv[1] == "compare":
            compare_with_without_chroma()
        else:
            print("Usage: python test_chroma_performance.py [single|multi|compare]")
    else:
        # Run all tests
        print("\n" + "#"*70)
        print("# CHROMADB INCIDENT MATCHER - PERFORMANCE TEST SUITE")
        print("#"*70)
        
        test_single_incident()
        
        print("\n\n" + "#"*70)
        print("# TEST COMPLETE")
        print("#"*70)
