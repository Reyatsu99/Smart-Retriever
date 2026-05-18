import os
import json
from pathlib import Path
from smart_retriever_v2.indexer import build_index
from smart_retriever_v2.auditor import DocumentAuditor
from smart_retriever_v2.search import SearchEngine
from smart_retriever_v2 import settings

def setup_test_data():
    data_dir = Path("v3_test_data")
    data_dir.mkdir(exist_ok=True)
    
    # 1. A Genuine File
    genuine_path = data_dir / "genuine_offer_alex.txt"
    genuine_path.write_text(
        "Offer Letter for Alex. Position: Data Scientist. "
        "Salary: $120,000 per year. Start date: June 1st. "
        "Please sign and return the document to HR.",
        encoding="utf-8"
    )
    
    # 2. A Bogus (Spoofed) File
    # This is designed to have a high vector similarity for the word 'offer' 
    # but it's clearly not a real document.
    bogus_path = data_dir / "bogus_offer_spam.txt"
    bogus_path.write_text(
        "offer offer offer offer offer offer offer offer offer offer "
        "salary salary salary salary salary salary salary salary salary "
        "alex alex alex alex alex alex alex alex alex alex alex alex",
        encoding="utf-8"
    )
    
    return data_dir

def run_v3_test():
    data_dir = setup_test_data()
    index_dir = Path("v3_test_index")
    
    print("--- Building V3 Test Index ---")
    # Build a fresh index for this test
    build_index(data_dir, index_dir, force=True)
    
    engine = SearchEngine(index_dir=index_dir)
    auditor = DocumentAuditor(engine)
    
    query = "Find the offer letter for Alex with salary details"
    
    print(f"\n--- Running Autonomous Audit for: '{query}' ---")
    print("Note: This requires Ollama and the Phi-3 model to be running.")
    
    try:
        reports = auditor.auto_verify(query)
        
        print("\nFINAL TEST RESULTS:")
        print("="*50)
        for r in reports:
            audit = r['audit']
            is_authentic = audit.get("is_authentic", True)
            status = "PASSED" if is_authentic else "FAILED (BOGUS DETECTED)"
            
            print(f"FILE: {r['file_name']}")
            print(f"AUTHENTICITY: {status}")
            if not is_authentic:
                print(f"REASON: {audit.get('authenticity_reason')}")
            
            print(f"SUMMARY: {audit.get('overall_summary')}")
            print("-" * 30)
        print("="*50)
    except Exception as e:
        print(f"\nAudit failed: {e}")
        print("Tip: Ensure 'ollama serve' is running and 'ollama pull phi3' has been executed.")

if __name__ == "__main__":
    run_v3_test()
