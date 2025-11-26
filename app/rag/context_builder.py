from typing import List, Dict, Any

class ContextBuilder:
    """
    Formats the list of final documents into a single, structured text string
    suitable for insertion into the LLM prompt.
    """
    
    def build_context(self, docs: List[Dict[str, Any]]) -> str:
        """
        Formats the documents into a readable context string with Source IDs.
        
        Format:
        ---
        [Source S1] Type: law | Source: Criminal Code, Sec 105
        Document: The maximum penalty for a Class A misdemeanor...
        ---
        [Source S2] Type: case | Case: State v. Williams (2023)
        Document: In State v. Williams (2023), the court upheld...
        ---
        ...
        """
        
        if not docs:
            return "No relevant legal documents were retrieved to form a context."

        context_parts = []
        
        # 1. Prepare and assign unique Source IDs (S1, S2, S3, ...)
        for i, doc in enumerate(docs):
            source_id = f"S{i+1}"
            doc['source_id'] = source_id # Attach the ID for reference in the final output

            # 2. Extract and format metadata for the header (no category/type)
            # 2. Extract and format metadata for the header (no category/type)
            src_db = doc.get('source_db') or doc['metadata'].get('source_file') or doc['metadata'].get('source') or 'unknown'
            doc['source_db'] = src_db  # Ensure source_db is set on the doc
            # Optional nice fields
            title = doc['metadata'].get('case_title') or doc['metadata'].get('orig_question') or doc['metadata'].get('original_question')
            year = doc['metadata'].get('year')

            if title:
                header = f"Source: {src_db} | Title: {title}"
            elif year:
                header = f"Source: {src_db} | Year: {year}"
            else:
                header = f"Source: {src_db}"
                
            
            # 3. Construct the full context entry
            context_entry = (
                f"[Source {source_id}] {header}\n"
                f"Document: {doc['text']}\n"
                f"Score: {float(doc.get('score', doc.get('rerank_score', 0.0))):.4f}"
            )
            context_parts.append(context_entry)
            
        return "\n---\n".join(context_parts)