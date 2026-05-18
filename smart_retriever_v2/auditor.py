import logging
from typing import Any, List, Dict
from pathlib import Path

from smart_retriever_v2.search import SearchEngine
from smart_retriever_v2.llm import LocalLLM

LOGGER = logging.getLogger(__name__)

AUDIT_SYSTEM_PROMPT = """You are a meticulous Document Auditor and Security Analyst. 
Your task is to verify if specific requirements are met AND check for document authenticity.

AUTHENTICITY CHECK:
Identify if the document is 'BOGUS'. A document is bogus if:
- It uses 'Keyword Stuffing' (repeating search terms without context).
- It contains gibberish or non-coherent text.
- It is a 'decoy' (text that has high similarity but zero actual useful information).

VERIFICATION TASK:
For each requirement, you must return:
1. Status: 'MET', 'NOT_MET', or 'NOT_FOUND'.
2. Reason: A brief explanation.
3. Evidence: A direct quote.

You MUST return your answer as a valid JSON object:
{
  "is_authentic": boolean,
  "authenticity_reason": "Explanation if bogus, or 'Verified' if genuine",
  "requirements": [
    {"name": "Requirement Name", "status": "MET", "reason": "...", "evidence": "..."}
  ],
  "overall_summary": "..."
}
"""

class DocumentAuditor:
    def __init__(self, search_engine: SearchEngine, llm_model: str = "phi3"):
        self.search_engine = search_engine
        self.llm = LocalLLM(model=llm_model)

    def auto_verify(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Autonomously infer requirements from the query and verify documents.
        """
        print(f"Inferring verification criteria for: '{query}'...")
        
        extraction_prompt = f"""Given the search query: '{query}'
What are the 3 specific factual criteria a document must meet to be considered a 'perfect match'?
Provide the criteria as a simple comma-separated list of short phrases.
Example for 'signed budget 2023': Document is a budget, Mentions 2023, Contains a signature.
Criteria:"""

        raw_reqs = self.llm.generate(extraction_prompt)
        # Clean up the list
        requirements = [r.strip() for r in raw_reqs.split(",") if r.strip()]
        print(f"Auto-extracted requirements: {requirements}")
        
        return self.audit(query, requirements, top_k=top_k)

    def audit(self, query: str, requirements: List[str], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Search for relevant documents and audit them against requirements.
        """
        # 1. Search for relevant chunks
        results = self.search_engine.search(query, top_k=top_k)
        
        audit_reports = []
        
        for res in results:
            context = res['text']
            file_name = res['file_name']
            
            # 2. Build the audit prompt
            req_list_str = "\n".join([f"- {req}" for req in requirements])
            prompt = f"""AUDIT REQUEST for Document: {file_name}
            
DOCUMENT SEGMENT:
---
{context}
---

REQUIREMENTS TO VERIFY:
{req_list_str}

Please perform the audit and return the JSON report."""

            # 3. Call the LLM
            print(f"Auditing '{file_name}'...")
            report = self.llm.generate_json(prompt, system_prompt=AUDIT_SYSTEM_PROMPT)
            
            # 4. Attach metadata
            full_report = {
                "file_name": file_name,
                "relative_path": res['relative_path'],
                "audit": report
            }
            audit_reports.append(full_report)
            
        return audit_reports
