import requests
import sys

print("Checking Ollama service...")
print("="*60)

try:
    response = requests.get("http://localhost:11434/api/tags", timeout=5)
    print("✓ Ollama is running!")
    print(f"Status Code: {response.status_code}")
    
    data = response.json()
    models = data.get('models', [])
    
    if models:
        print(f"\n✓ Found {len(models)} model(s):")
        for model in models:
            print(f"  - {model.get('name', 'Unknown')}")
        
        if any('llama3.1' in str(m.get('name', '')).lower() for m in models):
            print("\n✓ llama3.1 model is available!")
        else:
            print("\n⚠ llama3.1 model not found. Available models:")
            for model in models:
                print(f"  - {model}")
    else:
        print("\n⚠ No models found. Please download a model first.")
        print("  Run: ollama pull llama3.1")
        
except requests.exceptions.ConnectionError:
    print("✗ Cannot connect to Ollama!")
    print("  Make sure Ollama is running on http://localhost:11434")
    print("  Start Ollama with: ollama serve")
except Exception as e:
    print(f"✗ Error: {str(e)}")
    sys.exit(1)

print("\n" + "="*60)
