import json
import os
import re
import time
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

from excel_ingestion import load_structured_payloads

# Formatting helper
def _format_multi_sheet_answers(candidates: List[Dict[str, Any]]) -> Optional[str]:
    if not candidates:
        return None
    
    candidates.sort(key=lambda x: x["score"], reverse=True)
    top_score = candidates[0]["score"]
    
    # Keep candidates within a reasonable margin of the top score
    best_candidates = [c for c in candidates if c["score"] >= top_score - 1.5]
    
    seen = set()
    deduped = []
    for c in best_candidates:
        val_str = str(c["value"]).strip()
        if not val_str: continue
        key = (c["sheet_name"], val_str)
        if key not in seen:
            seen.add(key)
            deduped.append(c)
            
    if not deduped:
        return None
        
    if len(deduped) == 1:
        return str(deduped[0]["value"])
        
    # Check if all values are identical across sheets
    all_values = {str(c["value"]).strip().lower() for c in deduped}
    if len(all_values) == 1:
        return str(deduped[0]["value"])
        
    # Group by sheet
    sheet_groups = {}
    for c in deduped:
        sheet_groups.setdefault(c["sheet_name"], []).append(str(c["value"]))
        
    lines = []
    for sheet, vals in sheet_groups.items():
        unique_vals = list(dict.fromkeys(vals))
        lines.append(f"[{sheet}] " + " | ".join(unique_vals))
        
    return "\n".join(lines)
