"""
Performance Comparison: LLM vs Vector-Only Matching
====================================================
Compare speed and accuracy between LLM-based and vector-only approaches
"""

import time
import json
from incident_matcher_chroma import ChromaPatternMatcher
from incident_matcher_vector_only import VectorOnlyMatcher

def compare_single_incident():
    """Compare performance on a single incident"""
    print("\n" + "="*70)
    print("🧪 PERFORMANCE COMPARISON: LLM vs VECTOR-ONLY")
    print("="*70)
    
    test_incident = {
        "title": "User unable to login to banking portal",
        "description": "Multiple users reporting login failures with error code AUTH-500",
        "category": "Banking",
        "sub_category": "Authentication"
    }
    
    print("\n📋 TEST INCIDENT:")
    print(json.dumps(test_incident, indent=2))
    
    # Test 1: Vector-Only (No LLM)
    print("\n" + "-"*70)
    print("⚡ TEST 1: VECTOR-ONLY MATCHER (NO LLM)")
    print("-"*70)
    
    vector_start = time.time()
    vector_matcher = VectorOnlyMatcher()
    vector_init_time = time.time() - vector_start
    
    vector_match_start = time.time()
    vector_result = vector_matcher.match(test_incident, return_top_k=5)
    vector_match_time = time.time() - vector_match_start
    
    print(f"\n✅ Vector-Only Results:")
    print(f"   Top Match: {vector_result['matched_pattern']}")
    print(f"   Confidence: {vector_result['confidence']}")
    print(f"   Similarity: {vector_result['similarity_score']:.4f}")
    print(f"   Match Time: {vector_match_time*1000:.1f}ms")
    
    # Test 2: ChromaDB + LLM
    print("\n" + "-"*70)
    print("🤖 TEST 2: CHROMADB + LLM MATCHER")
    print("-"*70)
    
    llm_start = time.time()
    llm_matcher = ChromaPatternMatcher()
    llm_init_time = time.time() - llm_start
    
    llm_match_start = time.time()
    llm_result = llm_matcher.match(test_incident, show_similar=False)
    llm_match_time = time.time() - llm_match_start
    
    print(f"\n✅ LLM Results:")
    print(f"   Matched Pattern: {llm_result['matched_pattern']}")
    print(f"   Confidence: {llm_result['confidence']}")
    print(f"   Match Time: {llm_match_time*1000:.1f}ms")
    
    # Comparison
    print("\n" + "="*70)
    print("📊 PERFORMANCE COMPARISON")
    print("="*70)
    
    print(f"\n{'Metric':<30} {'Vector-Only':<20} {'ChromaDB+LLM':<20} {'Speedup':<15}")
    print("-"*85)
    print(f"{'Initialization Time':<30} {vector_init_time*1000:>8.1f}ms {llm_init_time*1000:>18.1f}ms {llm_init_time/vector_init_time:>14.1f}x")
    print(f"{'Match Time':<30} {vector_match_time*1000:>8.1f}ms {llm_match_time*1000:>18.1f}ms {llm_match_time/vector_match_time:>14.1f}x")
    print(f"{'Vector Search':<30} {vector_result['timing']['vector_search_ms']:>8.1f}ms {llm_result['timing']['vector_search_ms']:>18.1f}ms {'~same':<15}")
    print(f"{'LLM Inference':<30} {'N/A':<20} {llm_result['timing']['llm_inference_ms']:>18.1f}ms {'N/A':<15}")
    print("-"*85)
    print(f"{'TOTAL SPEEDUP':<30} {'':<20} {'':<20} {llm_match_time/vector_match_time:>14.1f}x FASTER")
    print("="*70)
    
    print(f"\n💡 KEY INSIGHTS:")
    print(f"   • Vector-Only is {llm_match_time/vector_match_time:.0f}x faster than LLM approach")
    print(f"   • Vector search time is similar (~{vector_result['timing']['vector_search_ms']:.1f}ms)")
    print(f"   • LLM inference adds ~{llm_result['timing']['llm_inference_ms']:.0f}ms overhead")
    print(f"   • Vector-Only response: {vector_match_time*1000:.1f}ms vs LLM: {llm_match_time*1000:.1f}ms")
    
    print(f"\n🎯 ACCURACY COMPARISON:")
    print(f"   Vector-Only Top Match: {vector_result['matched_pattern']}")
    print(f"   LLM Match:             {llm_result['matched_pattern']}")
    
    if vector_result['matched_pattern'] == llm_result['matched_pattern']:
        print(f"   ✅ SAME RESULT - Vector-only is accurate AND faster!")
    else:
        print(f"   ⚠️  Different results - LLM may provide better semantic understanding")
        print(f"\n   Vector-Only Top 5:")
        for match in vector_result['top_matches']:
            print(f"      {match['rank']}. {match['pattern_title']} ({match['similarity_score']:.4f})")


def compare_multiple_incidents():
    """Compare performance across multiple incidents"""
    print("\n" + "="*70)
    print("🧪 BATCH PERFORMANCE COMPARISON")
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
        }
    ]
    
    print(f"\n📋 Testing with {len(test_incidents)} incidents...")
    
    # Initialize matchers
    print("\n⚡ Initializing Vector-Only Matcher...")
    vector_matcher = VectorOnlyMatcher()
    
    print("\n🤖 Initializing ChromaDB+LLM Matcher...")
    llm_matcher = ChromaPatternMatcher()
    
    # Test Vector-Only
    print("\n" + "-"*70)
    print("⚡ TESTING VECTOR-ONLY MATCHER")
    print("-"*70)
    
    vector_times = []
    for i, incident in enumerate(test_incidents, 1):
        print(f"\n[{i}/{len(test_incidents)}] {incident['title'][:40]}...")
        start = time.time()
        result = vector_matcher.match(incident, return_top_k=5)
        elapsed = time.time() - start
        vector_times.append(elapsed * 1000)
        print(f"   ✅ Matched: {result['matched_pattern']} ({elapsed*1000:.1f}ms)")
    
    # Test LLM
    print("\n" + "-"*70)
    print("🤖 TESTING CHROMADB+LLM MATCHER")
    print("-"*70)
    
    llm_times = []
    for i, incident in enumerate(test_incidents, 1):
        print(f"\n[{i}/{len(test_incidents)}] {incident['title'][:40]}...")
        start = time.time()
        result = llm_matcher.match(incident, show_similar=False)
        elapsed = time.time() - start
        llm_times.append(elapsed * 1000)
        print(f"   ✅ Matched: {result['matched_pattern']} ({elapsed*1000:.1f}ms)")
    
    # Summary
    print("\n" + "="*70)
    print("📊 BATCH PERFORMANCE SUMMARY")
    print("="*70)
    
    avg_vector = sum(vector_times) / len(vector_times)
    avg_llm = sum(llm_times) / len(llm_times)
    total_vector = sum(vector_times)
    total_llm = sum(llm_times)
    
    print(f"\n{'Metric':<30} {'Vector-Only':<20} {'ChromaDB+LLM':<20}")
    print("-"*70)
    print(f"{'Incidents Processed':<30} {len(test_incidents):<20} {len(test_incidents):<20}")
    print(f"{'Average Time':<30} {avg_vector:>8.1f}ms {avg_llm:>18.1f}ms")
    print(f"{'Min Time':<30} {min(vector_times):>8.1f}ms {min(llm_times):>18.1f}ms")
    print(f"{'Max Time':<30} {max(vector_times):>8.1f}ms {max(llm_times):>18.1f}ms")
    print(f"{'Total Time':<30} {total_vector:>8.1f}ms {total_llm:>18.1f}ms")
    print("-"*70)
    print(f"{'SPEEDUP':<30} {'':<20} {avg_llm/avg_vector:>18.1f}x FASTER")
    print("="*70)
    
    print(f"\n💡 CONCLUSION:")
    print(f"   • Vector-Only is {avg_llm/avg_vector:.0f}x faster on average")
    print(f"   • Total time saved: {(total_llm - total_vector)/1000:.2f} seconds for {len(test_incidents)} incidents")
    print(f"   • For 100 incidents: Vector-Only ~{avg_vector*100/1000:.1f}s vs LLM ~{avg_llm*100/1000:.1f}s")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "single":
            compare_single_incident()
        elif sys.argv[1] == "batch":
            compare_multiple_incidents()
        else:
            print("Usage: python compare_performance.py [single|batch]")
    else:
        # Run both
        compare_single_incident()
        
        print("\n\n" + "#"*70)
        input("\nPress Enter to continue with batch comparison...")
        
        compare_multiple_incidents()
        
        print("\n\n" + "#"*70)
        print("# COMPARISON COMPLETE")
        print("#"*70)
