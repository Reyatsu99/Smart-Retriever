import logging
from typing import Any, List, Dict
from pathlib import Path

from smart_retriever.search import SearchEngine
from smart_retriever.llm import LocalLLM

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

    def _get_full_document_text(self, relative_path: str) -> str:
        """Fetch and reconstruct all chunks for a document from LanceDB in order."""
        try:
            table_name = "document_chunks"
            if table_name not in self.search_engine.db.table_names():
                return ""
            table = self.search_engine.db.open_table(table_name)
            # Escape single quotes in path if present
            safe_path = relative_path.replace("'", "''")
            chunks = table.search().where(f"relative_path = '{safe_path}'").to_arrow().to_pylist()
            if not chunks:
                return ""
            chunks = sorted(chunks, key=lambda x: x.get("chunk_id", 0))
            return "\n\n".join(c["text"] for c in chunks if "text" in c)
        except Exception as exc:
            LOGGER.warning(f"Failed to fetch full document text for {relative_path}: {exc}")
            return ""

    def audit(self, query: str, requirements: List[str], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Search for relevant documents and audit full document text against requirements.
        """
        # 1. Search for relevant documents
        results = self.search_engine.search(query, top_k=top_k)
        
        audit_reports = []
        
        for res in results:
            full_text = self._get_full_document_text(res['relative_path'])
            context = full_text if full_text else res['text']
            file_name = res['file_name']
            
            # 2. Build the audit prompt
            req_list_str = "\n".join([f"- {req}" for req in requirements])
            prompt = f"""AUDIT REQUEST for Document: {file_name}
            
DOCUMENT CONTENT:
---
{context}
---

REQUIREMENTS TO VERIFY:
{req_list_str}

Please perform the audit and return the JSON report."""

            # 3. Call the LLM
            print(f"Auditing '{file_name}' (full context)...")
            report = self.llm.generate_json(prompt, system_prompt=AUDIT_SYSTEM_PROMPT)
            
            # 4. Attach metadata
            full_report = {
                "file_name": file_name,
                "relative_path": res['relative_path'],
                "audit": report
            }
            audit_reports.append(full_report)
            
        return audit_reports
