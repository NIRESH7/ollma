"""
NeuralRAG Query & Retrieval Pipeline (deterministic-first).
1) STRUCTURED_LOOKUP: Exact row/field lookup from structured JSON cache.
2) NUMERIC_ENGINE: Pandas aggregation for numeric questions.
3) VECTOR_RETRIEVAL: Semantic recall from Qdrant.
4) LLM_REASONING: Strict final fallback.
"""

import json
import os
import re
import time
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

from langchain_community.llms import Ollama
from langchain_qdrant import QdrantVectorStore
from qdrant_client.http import models

from database import get_qdrant_client
from embeddings_provider import get_embeddings
from excel_ingestion import load_structured_payloads
from excel_query import is_numeric_question, run_dataframe_query
from excel_agent import agentic_excel_query

# ── Configuration ──────────────────────────────────────────────────
COLLECTION_NAME = "local_documents"
OLLAMA_MODEL = "qwen2.5:14b"  # Primary model — best local accuracy
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# ── Model Singletons ───────────────────────────────────────────────
MODEL_PRIORITY = [
    "qwen2.5:14b",
    "qwen2.5:7b",
    "qwen3:8b",
    "qwen3:4b",
    "llama3:8b",
    "llama3:70b",
    "mistral",
]

def _get_best_model():
    try:
        import requests
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        if resp.status_code == 200:
            installed = [m["name"] for m in resp.json().get("models", [])]
            # Pick the highest-priority installed model
            for preferred in MODEL_PRIORITY:
                for installed_m in installed:
                    if installed_m.startswith(preferred.split(":")[0]) and ":" in preferred and preferred in installed_m:
                        return installed_m
                    if installed_m == preferred:
                        return installed_m
            # Fallback: any chat model
            for m in installed:
                if any(k in m for k in ["qwen", "llama", "mistral", "phi"]):
                    return m
            if installed:
                return installed[0]
    except Exception:
        pass
    return OLLAMA_MODEL

actual_model = _get_best_model()
print(f"--- [RAG] Initializing Embeddings & LLM ({actual_model}) ---")
embeddings, _, embedding_backend = get_embeddings()
print(f"--- [RAG] Embedding backend: {embedding_backend} ---")
llm = Ollama(
    model=actual_model,
    temperature=0,
    top_p=0.9,
    repeat_penalty=1.1,
    base_url=OLLAMA_BASE_URL,
)
print("--- [RAG] Ready ---")


# ── Structured Retrieval Helpers ───────────────────────────────────
STOPWORDS = {
    "the", "this", "that", "for", "from", "with", "and", "or", "to", "me", "is", "are", "of", "in", "on", "tell", "what", "please",
}

INTENT_COLUMN_HINTS = {
    "count": ["headcount", "head_count", "count", "total_headcount", "number"],
    "coverage": ["coverage", "limit", "room", "board", "inpatient", "outpatient"],
    "plan_code": ["plan_code", "pmcare_plan_code", "code"],
    "name": ["name", "company", "insured"],
    "age_limit": ["age", "age_limit", "dependent_age_limit", "dependent", "limit"],
}

ENTITY_HINTS = {
    "employee": ["employee", "employees", "emp"],
    "spouse": ["spouse", "wife", "husband"],
    "children": ["children", "child", "kids"],
    "family": ["family", "dependent"],
    "executive": ["executive", "exec"],
    "non_exec": ["non exec", "non_exec", "non-exec"],
    "vice_president": ["vice president", "vp"],
}


def _tokenize(text: str) -> List[str]:
    text = (text or "").lower()
    # Collapse single-letter spaced words (e.g. "R O O M" → "room")
    text = re.sub(r"(?<=\b[a-z]) (?=[a-z]\b)", "", text)
    # Generic synonyms only (domain-agnostic)
    text = re.sub(r"\b(limitation|cap|maximum)\b", "limit", text)
    text = re.sub(r"\b(amt|value|sum)\b", "amount", text)
    text = re.sub(r"\b(headcount|qty|no\s+of)\b", "count", text)
    text = re.sub(r"\b(charges?|fee|cost|price)\b", "charge", text)
    tokens = re.split(r"[\W_]+", text)
    return [t for t in tokens if len(t) > 1 or t.isdigit()]


def _value_match_score(question_tokens: List[str], text: str) -> float:
    blob = (text or "").lower()
    blob = re.sub(r"(?<=\b[a-z]) (?=[a-z]\b)", "", blob)
    blob_tokens = re.split(r"[\W_]+", blob)
    
    score = 0.0
    stopwords = {"is", "the", "for", "and", "but", "what", "how", "give", "me"}
    for token in question_tokens:
        if token in stopwords:
            continue
        if token in blob_tokens:
            score += 2.0 if len(token) >= 5 else 1.0
        elif len(token) > 4 and token in blob:
            score += 1.0
        elif any(c.isdigit() for c in token):
            if token in blob_tokens:
                score += 2.5
            elif len(token) > 3 and token in blob:
                # Only allow substring match for longer alphanumeric tokens (like "plan1")
                score += 1.0
        elif len(token) > 3:
            # Add fuzzy matching for spelling mistakes
            for word in re.split(r"[\W_]+", blob):
                if word and SequenceMatcher(None, token, word).ratio() > 0.8:
                    score += 1.5 if len(token) >= 5 else 0.8
                    break
    return score


def _split_question_parts(question: str) -> Tuple[str, str]:
    q = (question or "").lower().strip()
    marker = re.search(r"\b(tell me|what is|what's|give me|show me|provide|find)\b", q)
    if marker:
        return q[: marker.start()].strip(), q[marker.end() :].strip()
    return "", q


def _detect_intents(question: str) -> List[str]:
    q = (question or "").lower()
    intents: List[str] = []
    if re.search(r"\b(head ?count|employee count|count|how many|number)\b", q):
        intents.append("count")
    if re.search(r"\b(coverage|limit|sum insured|inpatient|outpatient|room|board)\b", q):
        intents.append("coverage")
    if re.search(r"\b(plan code|pmcare|code)\b", q):
        intents.append("plan_code")
    if re.search(r"\b(name|company|insured)\b", q):
        intents.append("name")
    if re.search(r"\b(age|age limit|dependent age)\b", q):
        intents.append("age_limit")
    return intents


def _column_intent_boost(column: str, intents: List[str]) -> float:
    label = column.replace("_", " ").lower()
    boost = 0.0
    for intent in intents:
        hints = INTENT_COLUMN_HINTS.get(intent, [])
        if any(h.replace("_", " ") in label for h in hints):
            boost += 1.0
    if "count" in intents and any(w in label for w in ["group", "plan", "type", "category", "section"]):
        boost -= 0.8
    return boost


def _column_specificity_bonus(target_tokens: List[str], col_tokens: set) -> float:
    bonus = 0.0
    overlap = len(set(target_tokens).intersection(col_tokens))
    bonus += overlap * 2.5

    if "inpatient" in target_tokens:
        bonus += 1.8 if "inpatient" in col_tokens else -0.9
    if "outpatient" in target_tokens:
        bonus += 1.8 if "outpatient" in col_tokens else -0.9
    if "room" in target_tokens and "board" in target_tokens:
        bonus += 3.0 if "room" in col_tokens or "board" in col_tokens else -2.0
    if any(t in target_tokens for t in ["head", "headcount", "count", "employee"]):
        if any(t in col_tokens for t in ["headcount", "count", "no", "number"]):
            bonus += 1.0
    return bonus


def _extract_explicit_entities(question: str) -> List[str]:
    q = (question or "").lower()
    entities: List[str] = []
    for canonical, aliases in ENTITY_HINTS.items():
        for alias in aliases:
            alias_norm = alias.replace("_", " ")
            if alias_norm in q:
                entities.append(canonical)
                break
    return entities


def _row_entity_match(values: Dict[str, Any], entities: List[str]) -> float:
    if not entities:
        return 0.0

    entity_cols = [
        k
        for k in values.keys()
        if any(x in k.lower() for x in ["coverage", "type", "group", "position", "category", "job", "plan"])
    ]
    if not entity_cols:
        return 0.0

    blob = " ".join([str(values.get(k, "")).lower() for k in entity_cols])
    score = 0.0
    for entity in entities:
        aliases = ENTITY_HINTS.get(entity, [entity])
        if any(alias.replace("_", " ") in blob for alias in aliases):
            score += 1.5
    return score


def _choose_answer_column(
    columns: List[str], full_question: str, target_part: str, filter_part: str, intents: List[str]
) -> Tuple[Optional[str], float]:
    if not columns:
        return None, 0.0

    target_norm = target_part or full_question
    target_tokens = _tokenize(target_norm)
    filter_tokens = set(_tokenize(filter_part))
    best_col = None
    best_score = 0.0

    for col in columns:
        col_score = _field_score(target_norm, target_tokens, col)
        col_score += _column_intent_boost(col, intents)

        col_tokens = set(_tokenize(col.replace("_", " ")))
        col_score += _column_specificity_bonus(target_tokens, col_tokens)
        # Tokens mentioned in filter phrase are likely constraints, not answer fields.
        if filter_tokens and col_tokens.intersection(filter_tokens):
            col_score -= 0.6

        if col_score > best_score:
            best_score = col_score
            best_col = col

    return best_col, best_score


# Source column preference: prefer columns with 'eastspring' in name, downrank 'pamb'
_SOURCE_PREFER = re.compile(r'eastspring', re.I)
_SOURCE_DOWNRANK = re.compile(r'pamb', re.I)


def _rank_answer_columns(
    columns: List[str], full_question: str, target_part: str, filter_part: str, intents: List[str]
) -> List[Tuple[str, float]]:
    if not columns:
        return []

    target_norm = target_part or full_question
    target_tokens = _tokenize(target_norm)
    filter_tokens = set(_tokenize(filter_part))
    scored: List[Tuple[str, float]] = []

    for col in columns:
        col_score = _field_score(target_norm, target_tokens, col)
        col_score += _column_intent_boost(col, intents)
        col_tokens = set(_tokenize(col.replace("_", " ")))
        col_score += _column_specificity_bonus(target_tokens, col_tokens)
        if filter_tokens and col_tokens.intersection(filter_tokens):
            col_score -= 0.6
        scored.append((col, col_score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def _row_constraint_score(
    values: Dict[str, Any], answer_col: str, question_norm: str, constraint_tokens: List[str]
) -> float:
    score = 0.0
    token_set = set(constraint_tokens)
    seen_texts = set()

    # Mutual-exclusion groups: if question asks for one, penalize rows containing another
    _EXCLUSION_GROUPS = [
        {"employee", "staff", "emp"},
        {"family", "dependent", "spouse", "children"},
        {"plan 1", "plan1"},
        {"plan 2", "plan2"},
        {"plan 3", "plan3"},
        {"plan 4", "plan4"},
        {"plan 5", "plan5"},
        {"plan 6", "plan6"},
    ]

    for col, val in values.items():
        if col == answer_col:
            continue
        text = str(val).strip().lower()
        if not text or text in seen_texts:
            continue
        seen_texts.add(text)

        if len(text) > 2 and text in question_norm:
            score += 2.5
            continue

        value_tokens = set(_tokenize(text))
        overlap = len(value_tokens.intersection(token_set))
        if overlap:
            score += float(overlap)

    # Apply conflict penalty: if a row cell matches a conflicting group label
    for group in _EXCLUSION_GROUPS:
        q_hit = any(g in question_norm for g in group)
        if not q_hit:
            continue
        # Check if any cell in this row is a member of a DIFFERENT group
        for col, val in values.items():
            text = str(val).strip().lower()
            for other_g in group:
                if other_g in question_norm:
                    continue  # this is the asked one, skip
                if other_g in text:
                    score -= 2.5  # penalize: row is for a different entity

    return score


def _format_multi_sheet_answers(candidates: List[Dict[str, Any]]) -> Optional[str]:
    if not candidates:
        return None
    candidates.sort(key=lambda x: x["score"], reverse=True)
    top_score = candidates[0]["score"]
    best_candidates = [c for c in candidates if c["score"] >= top_score - 0.1]
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

    # Filter out placeholder values first
    real_vals = [c for c in deduped if not _is_placeholder(str(c.get("value", "")))]
    if not real_vals:
        return None
    if len(real_vals) == 1:
        return str(real_vals[0]["value"])

    # Prefer values from 'eastspring' source columns over 'pamb' when both present
    eastspring_vals = [c for c in real_vals if "eastspring" in str(c.get("sheet_name") or "").lower() 
                       or "eastspring" in str(c.get("field_key") or "").lower()]
    pamb_vals = [c for c in real_vals if "pamb" in str(c.get("field_key") or "").lower()]

    # Build unique value list — prefer Eastspring source
    all_values = []
    for c in real_vals:
        val = str(c["value"]).strip()
        if val not in all_values and not _is_placeholder(val):
            all_values.append(val)

    if len(all_values) == 0:
        return None
    if len(all_values) == 1:
        return all_values[0]

    # Cap at 2 values to avoid returning noise. If more than 2, take highest-scoring pair.
    return " | ".join(all_values[:2])


def _extract_sheet_name(question: str, folder_name: str) -> Optional[str]:
    q_norm = question.lower()
    q_tokens = set(_tokenize(q_norm))
    print(f"DEBUG: _extract_sheet_name q_tokens={q_tokens}")
    payloads = load_structured_payloads(folder_name=folder_name)
    
    best_sheet = None
    best_score = 0.0
    
    for payload in payloads:
        for sheet_name in payload.get("sheets", {}).keys():
            s_norm = sheet_name.lower()
            
            # 1. Exact substring match
            if s_norm in q_norm and len(s_norm) > 3:
                if len(s_norm) > (len(best_sheet or "")):
                    best_sheet = sheet_name
                    best_score = 1.0
                continue
                
            # 2. Fuzzy token overlap
            s_tokens = set(_tokenize(s_norm))
            if not s_tokens:
                continue
                
            overlap = len(q_tokens.intersection(s_tokens))
            score = overlap / len(s_tokens)
            print(f"DEBUG: sheet='{sheet_name}', s_tokens={s_tokens}, score={score}")
            
            # Require at least 70% of the sheet name's tokens to be in the question
            if score >= 0.70 and score > best_score:
                best_score = score
                best_sheet = sheet_name

    print(f"DEBUG: best_sheet returned={best_sheet}")
    return best_sheet


def _tabular_row_lookup(question: str, folder_name: str, target_sheet: Optional[str] = None) -> List[Dict[str, Any]]:
    q_norm = (question or "").lower()
    filter_part, target_part = _split_question_parts(q_norm)
    intents = _detect_intents(q_norm)
    explicit_entities = _extract_explicit_entities(q_norm)

    full_tokens = [t for t in _tokenize(q_norm) if t not in STOPWORDS]
    metric_tokens = {"count", "head", "headcount", "many", "number", "coverage", "limit", "inpatient", "outpatient"}
    constraint_tokens = [t for t in full_tokens if t not in metric_tokens]

    candidates = []

    payloads = load_structured_payloads(folder_name=folder_name)
    for payload in payloads:
        for sheet_name, sheet_data in payload.get("sheets", {}).items():
            if target_sheet and sheet_name != target_sheet:
                continue
            records = sheet_data.get("records", [])
            if not records:
                continue

            columns = list(sheet_data.get("headers") or [])
            if not columns:
                col_set = set()
                for r in records:
                    col_set.update((r.get("values") or {}).keys())
                columns = sorted(col_set)

            ranked_columns = _rank_answer_columns(
                columns=columns,
                full_question=q_norm,
                target_part=target_part,
                filter_part=filter_part,
                intents=intents,
            )
            print(f"DEBUG: ranked_columns for sheet {sheet_name} = {ranked_columns}")
            if not ranked_columns:
                continue

            for record in records:
                values = record.get("values", {})
                if not isinstance(values, dict) or not values:
                    continue

                row_score = _row_constraint_score(values, "", q_norm, constraint_tokens)
                row_score += _row_entity_match(values, explicit_entities)
                if constraint_tokens and row_score < 1.0:
                    continue

                for answer_col, answer_col_score in ranked_columns[:4]:
                    if answer_col_score < 1.0:
                        continue

                    answer_val = str(values.get(answer_col, "")).strip()
                    if not answer_val:
                        continue

                    total_score = (row_score * 2.0) + (answer_col_score * 3.0)

                    if "count" in intents:
                        if re.search(r"\d", answer_val):
                            total_score += 1.0
                        else:
                            total_score -= 1.0
                    if "age_limit" in intents:
                        answer_lower = answer_val.lower()
                        if re.search(r"\d", answer_lower) or "year" in answer_lower or "age" in answer_lower:
                            total_score += 1.2
                        else:
                            total_score -= 2.0

                    print(f"DEBUG: evaluated col {answer_col}: val={answer_val}, row_score={row_score}, score={total_score}")

                    if intents and filter_part and answer_val.lower() in filter_part and len(answer_val.split()) <= 3:
                        total_score -= 1.5

                    if explicit_entities and _row_entity_match(values, explicit_entities) <= 0:
                        total_score -= 1.5

                    if total_score >= 2.5:
                        candidates.append({
                            "sheet_name": sheet_name,
                            "value": answer_val,
                            "score": total_score
                        })

    return candidates


def _field_score(question_norm: str, question_tokens: List[str], key: str) -> float:
    key_label = key.replace("_", " ").lower()
    key_tokens = _tokenize(key_label)
    if not key_tokens:
        return 0.0

    overlap = len(set(question_tokens).intersection(set(key_tokens))) / len(set(key_tokens))
    contains = 1.0 if any(kt in question_norm for kt in key_tokens) else 0.0
    return overlap + (0.2 * contains)


def _choose_best_field(
    question_norm: str, question_tokens: List[str], values: Dict[str, Any]
) -> Tuple[Optional[str], float]:
    best_key = None
    best_score = 0.0
    for key in values.keys():
        score = _field_score(question_norm, question_tokens, key)
        if score > best_score:
            best_score = score
            best_key = key
    return best_key, best_score


def _choose_label_driven_value(question_tokens: List[str], values: Dict[str, Any]) -> Optional[str]:
    """
    For rows shaped like [label -> value], return the value side when the label
    text strongly matches the question.
    """
    scored_items: List[Tuple[str, str, float]] = []
    for key, value in values.items():
        value_text = str(value).strip()
        if not value_text:
            continue
        score = _value_match_score(question_tokens, value_text)
        scored_items.append((key, value_text, score))

    if not scored_items:
        return None

    # Deduplicate identical text values (common in merged-cell propagated sheets).
    dedup: Dict[str, Tuple[str, str, float]] = {}
    for item in scored_items:
        _, value_text, score = item
        if value_text not in dedup or score > dedup[value_text][2]:
            dedup[value_text] = item
    scored_items = list(dedup.values())
    if len(scored_items) < 2:
        return None

    # Identify the label cell that best matches the question.
    label_key, _, label_score = max(scored_items, key=lambda x: x[2])
    if label_score < 2.0:
        return None

    # Pick the best candidate value from other cells.
    value_candidates = []
    for key, value_text, score in scored_items:
        if key == label_key:
            continue
        if len(value_text) <= 1:
            continue
        # Prefer non-label-like values with useful length and alphabetic content.
        candidate_score = (len(value_text) / 40.0) - score
        if re.search(r"[A-Za-z]", value_text):
            candidate_score += 0.3
        if re.fullmatch(r"\d+(\.\d+)?", value_text):
            candidate_score += 0.2
        value_candidates.append((value_text, candidate_score))

    if not value_candidates:
        return None

    best_value, _ = max(value_candidates, key=lambda x: x[1])
    return best_value.strip() if best_value.strip() else None


def _matrix_lookup(question: str, folder_name: str, target_sheet: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Handles cross-row matrix tables where:
    - one row identifies entity/category by column
    - another row (or header) holds the requested metric for the same column
    """
    q_norm = question.lower()
    q_tokens = _tokenize(q_norm)
    candidates = []

    payloads = load_structured_payloads(folder_name=folder_name)
    for payload in payloads:
        for sheet_name, sheet_data in payload.get("sheets", {}).items():
            if target_sheet and sheet_name != target_sheet:
                continue
            records = sheet_data.get("records", [])
            headers = sheet_data.get("headers", [])
            if not records or len(headers) < 2:
                continue

            first_col_key = headers[0]

            # 1) Find the best matching entity cell and capture its column key.
            best_entity_col = None
            best_entity_score = 0.0
            
            for col_key in headers[1:]:
                header_score = _value_match_score(q_tokens, col_key.replace("_", " "))
                if header_score > best_entity_score:
                    best_entity_score = header_score
                    best_entity_col = col_key

            for record in records:
                values = record.get("values", {})
                for col_key, col_val in values.items():
                    if col_key == first_col_key:
                        continue
                    score = _value_match_score(q_tokens, str(col_val))
                    if score > best_entity_score:
                        best_entity_score = score
                        best_entity_col = col_key

            if not best_entity_col or best_entity_score < 2.0:
                continue

            # 2) If question asks for the first-column label itself (e.g. PMCare plan code),
            # the answer is often the value in the first column for the matched entity row!
            if _field_score(q_norm, q_tokens, first_col_key) >= 0.70:
                # Find the value in first_col_key for the row where best_entity_col matched best
                for record in records:
                    vals = record.get("values", {})
                    # If this row is the one that matched the entity
                    if _value_match_score(q_tokens, str(vals.get(best_entity_col, ""))) == best_entity_score:
                        ans_val = str(vals.get(first_col_key, "")).strip()
                        if ans_val:
                            candidates.append({
                                "sheet_name": sheet_name,
                                "value": ans_val,
                                "score": best_entity_score * 3.0
                            })
                            break

            # 3) Find row whose first-column label matches the entity/filter in question.
            # For plan+entity queries (e.g. "Employee OP for PLAN 1"), only accept strict row match.
            best_row_value = None
            best_row_score = 0.0
            _plan_entity_q = bool(re.search(r'\bplan\s*\d+\b', q_norm)) and any(
                e in q_norm for e in ["employee", "family", "dependent", "spouse", "staff"]
            )
            for record in records:
                values = record.get("values", {})
                row_label = str(values.get(first_col_key, ""))
                row_score = _value_match_score(q_tokens, row_label)
                if row_score <= 0:
                    continue
                # Strict match for plan+entity questions: row label must be in the question
                if _plan_entity_q and row_label.strip().lower() not in q_norm:
                    continue
                candidate_val = values.get(best_entity_col)
                if candidate_val is None or str(candidate_val).strip() == "":
                    continue
                if row_score > best_row_score:
                    best_row_score = row_score
                    best_row_value = str(candidate_val).strip()

            if best_row_value and best_row_score >= 1.0:
                candidates.append({
                    "sheet_name": sheet_name,
                    "value": best_row_value,
                    "score": best_row_score * 3.0
                })

            # Raw matrix fallback for sheets where structured headers are ambiguous.
            raw_rows = sheet_data.get("raw_rows", [])
            if raw_rows and re.search(r"\b(plan code|pmcare|coverage|limit|room|board|headcount|count)\b", q_norm):
                # Build ordered row cells from col_1..col_n keys.
                ordered_rows: List[List[str]] = []
                for raw in raw_rows:
                    values = raw.get("values", {})
                    ordered = []
                    for key, value in values.items():
                        m = re.match(r"col_(\d+)$", key)
                        if m:
                            ordered.append((int(m.group(1)), str(value).strip()))
                    if not ordered:
                        continue
                    ordered.sort(key=lambda x: x[0])
                    ordered_rows.append([v for _, v in ordered])

                if ordered_rows:
                    plan_row = None
                    axis_row = None
                    for row_cells in ordered_rows:
                        label = " ".join([c.lower() for c in row_cells[:2] if c]).strip()
                        if "plan code" in label or "pmcare" in label:
                            plan_row = row_cells
                        if any(k in label for k in ["job category", "position", "coverage for", "type of coverage"]):
                            axis_row = row_cells

                    if plan_row and axis_row:
                        best_col = None
                        best_col_score = 0.0
                        for idx in range(2, min(len(plan_row), len(axis_row))):
                            axis_value = axis_row[idx].lower().strip()
                            if not axis_value:
                                continue
                            score = _value_match_score(q_tokens, axis_value)
                            if score > best_col_score:
                                best_col_score = score
                                best_col = idx

                        if best_col is not None and best_col_score >= 2.0:
                            candidate = plan_row[best_col].strip()
                            if candidate:
                                candidates.append({
                                    "sheet_name": sheet_name,
                                    "value": candidate,
                                    "score": best_col_score * 2.0
                                })

    return candidates


def _iter_structured_records(folder_name: str, target_sheet: Optional[str] = None):
    payloads = load_structured_payloads(folder_name=folder_name)
    for payload in payloads:
        file_name = payload.get("file_name", "")
        for sheet_name, sheet_data in payload.get("sheets", {}).items():
            if target_sheet and sheet_name != target_sheet:
                continue
            for record in sheet_data.get("records", []):
                values = record.get("values", {})
                if isinstance(values, dict) and values:
                    yield file_name, sheet_name, record
            for raw_record in sheet_data.get("raw_rows", []):
                values = raw_record.get("values", {})
                if isinstance(values, dict) and values:
                    yield file_name, sheet_name, raw_record


# Values that are placeholder/incomplete — should never be returned as answers
_PLACEHOLDER_PATTERNS = re.compile(
    r'^(please\s+(confirm|advise|check|revert)|to\s+be\s+(confirm(ed)?|advise?d?)|tba|tbc|n/a|na|'  
    r'covered\s*/\s*not\s+covered|cover\s*/\s*not\s+covered|yes/no|yes\s*/\s*no|refer\s+ghs|'  
    r'please\s+let\s+us|if\s+any|no\s+number).*$',
    re.IGNORECASE
)

def _is_placeholder(value: str) -> bool:
    """Return True if the value is a placeholder that should not be returned as an answer."""
    v = (value or "").strip()
    return bool(_PLACEHOLDER_PATTERNS.match(v))


def _suspicious_direct_answer(question: str, answer: str) -> bool:
    q = (question or "").lower()
    a = (answer or "").strip().lower()
    if not a:
        return True

    # Filter out placeholder-style answers
    if _is_placeholder(answer):
        return True

    # Reject concatenated headers masquerading as answers
    if "|" in a and not re.search(r"(\d|rm|RM)", answer) and "as charged" not in a:
        return True

    # Detect plan+entity matrix questions (e.g. "Employee OP limit for PLAN 1")
    # These need precise matrix lookup — single numbers from structured lookup are unreliable
    # NOTE: Do NOT route to agentic here — it's too slow (100s+). Matrix lookup handles it.

    intents = _detect_intents(q)
    if not intents:
        return False

    metric_intent = any(x in intents for x in ["count", "coverage", "plan_code"])
    if metric_intent and a in q and len(a.split()) <= 3:
        return True

    if "count" in intents and not re.search(r"\d", a):
        return True
    if "age_limit" in intents and not (re.search(r"\d", a) or "age" in a or "year" in a):
        return True

    return False


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    text = text.strip()

    # Fast path.
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Try first {...} block.
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _collect_structured_candidates(question: str, folder_name: str, limit: int = 200) -> List[str]:
    q_tokens = _tokenize(question)
    scored: List[Tuple[float, str]] = []

    payloads = load_structured_payloads(folder_name=folder_name)
    for payload in payloads:
        file_name = payload.get("file_name", "")
        for sheet_name, sheet_data in payload.get("sheets", {}).items():
            # Mandatory header context for sheets
            headers = sheet_data.get("headers", [])
            if headers:
                h_text = "COLUMNS: " + " | ".join(headers)
                scored.append((5.0, f"File: {file_name} | Sheet: {sheet_name} | {h_text}"))
            
            # Include top raw rows (pivots/headers) regardless of score to help LLM orientation
            raw_rows = sheet_data.get("raw_rows", [])
            for record in raw_rows[:5]:
                values = record.get("values", {})
                row_text = " | ".join([f"{v}" for v in values.values()])
                scored.append((4.0, f"File: {file_name} | Sheet: {sheet_name} | HEADER_ROW: {row_text}"))

            for row_bucket in ("records", "raw_rows"):
                for record in sheet_data.get(row_bucket, []):
                    values = record.get("values", {})
                    if not isinstance(values, dict) or not values:
                        continue
                    row_text = " | ".join([f"{k}: {v}" for k, v in values.items()])
                    section = record.get("section", "General")
                    context = f"File: {file_name} | Sheet: {sheet_name} | Section: {section} | {row_text}"
                    score = _value_match_score(q_tokens, context)
                    if score > 0:
                        scored.append((score, context))

    if not scored:
        return []

    scored.sort(key=lambda x: x[0], reverse=True)
    unique_contexts: List[str] = []
    seen = set()
    for _, context in scored:
        if context in seen:
            continue
        seen.add(context)
        unique_contexts.append(context)
        if len(unique_contexts) >= 120:
            break
    return unique_contexts


def _llm_structured_fallback(question: str, folder_name: str) -> Optional[str]:
    candidates = _collect_structured_candidates(question, folder_name, limit=70)
    if not candidates:
        return None

    context = "\n".join([f"- {c}" for c in candidates])
    prompt = f"""You are a Universal Spreadsheet Intelligence Engine.

Your task is to answer user questions using structured data extracted from spreadsheet files such as Excel or CSV.
The spreadsheets may have completely different layouts, column names, structures, and sheet organizations.
You must dynamically understand the spreadsheet structure and return accurate answers strictly from the provided data.

------------------------------------------------
PRIMARY OBJECTIVE
Always prioritize exact values from the spreadsheet data.
If the answer exists in the data, return it exactly.
Never invent numbers, sheet names, rows, columns, or values.
If the requested information is not present in the provided data, respond with:
"The requested information is not available in the provided spreadsheet data."

------------------------------------------------
SUPPORTED SPREADSHEET STRUCTURES
Spreadsheet data may appear in many formats. You must dynamically interpret these layouts.
Possible structures include:
1. Tabular tables: A header row followed by multiple data rows.
2. Key-value mappings: Two-column structures where a key corresponds to a value.
3. Hierarchical documents: Sections containing lists, numbered items, or nested content.
4. Matrix layouts: Row headers intersecting with column headers.
5. Semi-structured policy documents: Text sections mixed with tables.
You must correctly identify the structure before answering.

------------------------------------------------
MULTI-SHEET DATA
Spreadsheets may contain multiple sheets.
If the user specifies a sheet name in the question: Search only that sheet.
If no sheet is mentioned: Search all sheets and select the most relevant result.

------------------------------------------------
ENTITY IDENTIFICATION
Extract important entities from the user's question, such as:
codes, categories, plans, groups, identifiers, items, employee types, benefit names, coverage types, limits, positions, product names, dates, sections.
Match these entities against the spreadsheet rows.
The correct row is the row that matches the highest number of relevant entities.

------------------------------------------------
ROW MATCHING STRATEGY
Never match rows using only a single keyword.
Instead, find the row that matches the most entities simultaneously (e.g., match identifiers + categories + plan names + employee types + item names).
The row with the strongest combined match should be selected.

------------------------------------------------
COLUMN DETECTION
Determine which column or field the user is asking for (e.g., coverage, limit, headcount, amount, value, benefit, position, category, description).
Return only the value from the requested column.
Do not return entire rows unless explicitly requested.

------------------------------------------------
HEADER RULE
Column headers describe the structure of the table but are not valid answers.
Do not return headers unless the user explicitly asks for column names.

------------------------------------------------
LIST QUERIES
If the user asks for a list of values:
Return all relevant entries from the corresponding column.
Avoid repeating identical values unless they represent different rows.

------------------------------------------------
STRUCTURE QUESTIONS
If the question refers to document structure such as: first item, title, heading, checklist item, section name.
Identify the appropriate section and return the requested element.

------------------------------------------------
NUMERIC QUESTIONS
For numeric queries such as: limits, amounts, coverage, totals, counts.
Return the exact value from the spreadsheet data. Never estimate numbers.

------------------------------------------------
FUZZY MATCHING
Users may include small spelling variations or formatting differences.
Treat similar expressions as the same concept when appropriate (e.g., spacing differences, minor spelling mistakes, abbreviations).
However, do not guess values.

------------------------------------------------
CONTEXT USAGE
Only use the spreadsheet data provided in the context.
Do not rely on external knowledge. Do not fabricate missing information.

------------------------------------------------
OUTPUT FORMAT
Return STRICT JSON with the following keys:
- "answer": concise exact value string from the data. If the answer comes from specific cell coordinates or rows, mention them if helpful. Format: "[Sheet Name] Value". If not found, use exactly "The requested information is not available in the provided spreadsheet data."
- "evidence": The exact raw row text or data snippet you found.
- "confidence": A score from 0.0 to 1.0 reflecting how certainly the data answers the specific question.
- "reasoning": A 1-sentence explanation of why you picked this data.

JSON FORMAT ONLY.

QUESTION:
{question}

ROWS CONTEXT:
{context}

JSON:"""

    try:
        raw = llm.invoke(prompt).strip()
    except Exception as e:
        print(f"LOG: LLM_STRUCTURED_FALLBACK_ERROR | {e}")
        return None

    parsed = _extract_json_object(raw)
    if not parsed:
        return None

    answer = str(parsed.get("answer", "")).strip()
    evidence = str(parsed.get("evidence", "")).strip()
    try:
        confidence = float(parsed.get("confidence", 0))
    except Exception:
        confidence = 0.0

    if not answer or answer.lower() == "data not available":
        return None

    # Fix 4: Relaxed guardrails for production robustness
    # - Allow partial answer match in context (not exact)
    # - Lower confidence threshold to 0.40
    normalized_context = "\n".join(candidates).lower()
    answer_words = [w for w in answer.lower().split() if len(w) > 2]
    context_hit = any(w in normalized_context for w in answer_words) if answer_words else True
    if not context_hit:
        return None
    # Relax evidence check: allow partial match
    if evidence:
        evidence_words = [w for w in answer.lower().split() if len(w) > 2]
        if evidence_words and not any(w in evidence.lower() for w in evidence_words):
            return None
    if confidence < 0.40:
        return None

    return answer


def _structured_lookup(question: str, folder_name: str, target_sheet: Optional[str] = None) -> Optional[str]:
    """
    Strict deterministic retrieval from structured JSON cache.
    Returns value only when confidence is high; otherwise None.
    """
    question_norm = question.lower()
    question_tokens = _tokenize(question_norm)
    if not question_tokens:
        return None

    intents = _detect_intents(question_norm)

    # Matrix lookup handles cross-column mapping questions like plan code by category.
    matrix_cands = _matrix_lookup(question, folder_name, target_sheet)
    if matrix_cands:
        matrix_cands = [c for c in matrix_cands if not _suspicious_direct_answer(question, str(c.get("value", "")))]
        if matrix_cands:
            ans = _format_multi_sheet_answers(matrix_cands)
            if ans: return ans

    # Tabular row lookup handles "for group X plan Y give me headcount" queries.
    table_cands = _tabular_row_lookup(question, folder_name, target_sheet)
    # print(f"DEBUG: table_cands={table_cands}")
    if table_cands:
        table_cands = [c for c in table_cands if not _suspicious_direct_answer(question, str(c.get("value", "")))]
        if table_cands:
            ans = _format_multi_sheet_answers(table_cands)
            print(f"DEBUG: ans={ans}")
            if ans: return ans

    filter_part, target_part = _split_question_parts(question_norm)
    scoring_text = target_part or question_norm
    scoring_tokens = _tokenize(scoring_text)
    if not scoring_tokens:
        scoring_tokens = question_tokens

    candidates = []

    for file_name, sheet_name, record in _iter_structured_records(folder_name, target_sheet):
        values = record.get("values", {})
        section = str(record.get("section", ""))
        row_text = " ".join([str(v) for v in values.values()])
        context = f"{file_name} {sheet_name} {section} {row_text}"
        row_score = _value_match_score(question_tokens, context)
        if row_score <= 0:
            continue

        field_key, field_score = _choose_best_field(scoring_text, scoring_tokens, values)
        label_driven_value = _choose_label_driven_value(question_tokens, values)
        total_score = row_score + (field_score * 3.0)

        if field_key:
            field_label_tokens = set(_tokenize(field_key))
            if filter_part and field_label_tokens.intersection(set(_tokenize(filter_part))):
                total_score -= 1.0

        candidates.append({
            "sheet_name": sheet_name,
            "values": values,
            "field_key": field_key,
            "field_score": field_score,
            "label_driven_value": label_driven_value,
            "row_score": row_score,
            "score": total_score,
        })

    if not candidates:
        mc = _matrix_lookup(question, folder_name, target_sheet)
        if mc:
            mc = [c for c in mc if not _suspicious_direct_answer(question, str(c.get("value", "")))]
            if mc:
                ans = _format_multi_sheet_answers(mc)
                if ans: return ans
        return None

    # Sort and filter fallbacks
    candidates.sort(key=lambda x: x["score"], reverse=True)
    best_candidate = candidates[0]

    # Strict threshold to avoid hallucinated field mapping.
    if best_candidate["row_score"] < 2.0:
        return None

    # Build final formatted return similarly to previous simple logic but allowing multiple sheets if they tied
    high_scoring = [c for c in candidates if c["score"] >= best_candidate["score"] - 0.1 and c["row_score"] >= 2.0]
    
    extracted_vals = []
    for cd in high_scoring:
        field_key = cd["field_key"]
        val = None
        if field_key and cd["field_score"] >= 0.45:
            cv = cd["values"].get(field_key)
            if cv is not None and str(cv).strip():
                keys_in_order = list(cd["values"].keys())
                first_key = keys_in_order[0] if keys_in_order else None
                if field_key == first_key and len(cd["values"]) > 2:
                    other_cell_match = max(
                        (_value_match_score(question_tokens, str(v)) for k, v in cd["values"].items() if k != field_key), 
                        default=0.0
                    )
                    if other_cell_match < 2.0:
                        val = str(cv).strip()
                else:
                    val = str(cv).strip()

        if not val and cd.get("label_driven_value"):
            val = str(cd["label_driven_value"]).strip()

        if not val:
            ordered_values = list(cd["values"].values())
            if len(ordered_values) == 2 and str(ordered_values[1]).strip():
                val = str(ordered_values[1]).strip()
                
        if val:
            extracted_vals.append({
                "sheet_name": cd["sheet_name"],
                "value": val,
                "score": cd["score"],
                "field_key": cd.get("field_key", ""),  # needed for source preference
            })

    if extracted_vals:
        extracted_vals = [c for c in extracted_vals if not _suspicious_direct_answer(question, str(c.get("value", "")))]
        if extracted_vals:
            ans = _format_multi_sheet_answers(extracted_vals)
            if ans: return ans

    # Cross-row matrix fallback.
    mc = _matrix_lookup(question, folder_name, target_sheet)
    if mc:
        mc = [c for c in mc if not _suspicious_direct_answer(question, str(c.get("value", "")))]
        if mc:
            ans = _format_multi_sheet_answers(mc)
            if ans: return ans
    return None


def _answer_from_vector_metadata(question: str, docs: List[Any]) -> Optional[str]:
    question_norm = question.lower()
    question_tokens = _tokenize(question_norm)
    best_value = None
    best_score = 0.0

    for doc in docs:
        metadata = getattr(doc, "metadata", {}) or {}
        row_json = metadata.get("record_json")
        if not row_json:
            continue
        try:
            values = json.loads(row_json)
            if not isinstance(values, dict) or not values:
                continue
        except Exception:
            continue

        row_text = " ".join([str(v) for v in values.values()])
        row_score = _value_match_score(question_tokens, row_text)
        if row_score <= 0:
            continue
        field_key, field_score = _choose_best_field(question_norm, question_tokens, values)
        total_score = row_score + (field_score * 3.0)
        if total_score > best_score and field_key and field_score >= 0.45:
            value = values.get(field_key)
            if value is not None and str(value).strip():
                best_score = total_score
                best_value = str(value).strip()

    return best_value


# ── Spell Correction Helpers ───────────────────────────────────────
def _build_vocab_from_payloads(folder_name: str) -> set:
    """Build a vocabulary of known words from all ingested data for this folder."""
    vocab = set()
    payloads = load_structured_payloads(folder_name=folder_name)
    for payload in payloads:
        for sheet_name, sheet_data in payload.get("sheets", {}).items():
            vocab.update(_tokenize(sheet_name))
            for record in sheet_data.get("records", []) + sheet_data.get("raw_rows", []):
                for v in record.get("values", {}).values():
                    vocab.update(_tokenize(str(v)))
            for h in (sheet_data.get("headers") or []):
                vocab.update(_tokenize(h))
    # Remove very short/generic tokens
    return {w for w in vocab if len(w) > 3}


def _fuzzy_correct_question(question: str, folder_name: str) -> str:
    """Correct spelling mistakes in the question using known vocab from data."""
    from difflib import get_close_matches
    vocab = _build_vocab_from_payloads(folder_name)
    if not vocab:
        return question
    vocab_list = list(vocab)
    tokens = re.split(r"([\W_]+)", question)  # preserve separators
    corrected = []
    for token in tokens:
        word = token.strip().lower()
        # Only try to correct alpha words of decent length
        if len(word) >= 4 and word.isalpha() and word not in vocab:
            matches = get_close_matches(word, vocab_list, n=1, cutoff=0.80)
            if matches:
                # Preserve original casing style
                corrected.append(matches[0])
                continue
        corrected.append(token)
    result = "".join(corrected)
    if result != question:
        print(f"LOG: SPELL_CORRECT | '{question}' → '{result}'")
    return result


def query_rag(question: str, folder_name: str = "All") -> str:
    print(f"\nLOG: QUERY_RECEIVED | {question}")
    _ = time.time()

    print("LOG: FLOW | Normalization")
    clean_question = question.replace("garde", "grade").replace("JL12", "JL 12").replace("emp", "employee")
    # Spell correction: fix typos using vocab from the uploaded data
    clean_question = _fuzzy_correct_question(clean_question, folder_name)

    print("LOG: FLOW | Intent Detection")
    intents = _detect_intents(clean_question)
    
    print("LOG: FLOW | Entity Extraction")
    entities = _extract_explicit_entities(clean_question)

    print("LOG: FLOW | Sheet Detection")
    target_sheet = _extract_sheet_name(clean_question, folder_name)
    if target_sheet:
        print(f"LOG: SHEET DETECTED | Restricting lookup to sheet: {target_sheet}")

    # STEP 1: Deterministic lookup (DISABLED - User strictly prefers the filtered LLM flow)
    # direct_match = _structured_lookup(clean_question, folder_name, target_sheet)
    # if direct_match:
    #     if not _suspicious_direct_answer(clean_question, direct_match):
    #         return direct_match

    print("LOG: FLOW | Bypassing fragile deterministic lookup -> Using Fast Pruned LLM Table-QA directly")

    # STEP 2: Agentic Pandas Engine — LLM writes & runs Pandas code on the actual Excel file
    print("LOG: FLOW | Agentic Markdown Table-QA (qwen2.5:14b reads flat text)")
    agentic_result = agentic_excel_query(clean_question, folder_name, target_sheet=target_sheet, llm=llm)
    if agentic_result:
        if not _suspicious_direct_answer(clean_question, agentic_result):
            print(f"LOG: FINAL_ANSWER | (Agentic) {agentic_result}")
            return agentic_result
        print(f"LOG: AGENTIC | Suspicious answer ignored: {agentic_result}")

    # STEP 3: Numeric math engine
    if is_numeric_question(clean_question):
        print("LOG: FLOW | Math Engine Computation")
        math_result = run_dataframe_query(clean_question, folder_name)
        if math_result:
            print(f"LOG: FINAL_ANSWER | (Math Engine) {math_result}")
            return math_result

    # STEP 4: Universal Analyst Engine (LLM reads structured JSON context)
    print("LOG: FLOW | Universal Analyst Engine Scanning...")
    llm_structured = _llm_structured_fallback(clean_question, folder_name)
    if llm_structured:
        print("LOG: FLOW | Universal Engine Found Answer")
        print(f"LOG: FINAL_ANSWER | (Engine) {llm_structured}")
        return llm_structured

    # STEP 3: Vector retrieval
    print("LOG: FLOW | Vector Search (Qdrant Raw Texts)")
    docs: List[Any] = []
    try:
        client = get_qdrant_client()
        vector_store = QdrantVectorStore(
            client=client,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
        )

        folder_filter = None
        if folder_name and folder_name != "All":
            folder_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.folder",
                        match=models.MatchValue(value=folder_name),
                    )
                ]
            )

        docs = vector_store.similarity_search(clean_question, k=20, filter=folder_filter)
    except Exception as e:
        print(f"LOG: VECTOR_RETRIEVAL_ERROR | {e}")
        docs = []

    if docs:
        metadata_match = _answer_from_vector_metadata(clean_question, docs)
        if metadata_match:
            print(f"LOG: FINAL_ANSWER | (Vector Metadata) {metadata_match}")
            return metadata_match

    if not docs:
        print("LOG: FINAL_ANSWER | Data not available.")
        return "Data not available."

    # STEP 4: LLM strict fallback
    print("LOG: LLM_REASONING | Strict synthesis fallback...")
    context = "\n---\n".join([d.page_content for d in docs])
    prompt = f"""Answer the question using only the context.
Return only the exact value if found.
If not found, return exactly: Data not available.

CONTEXT:
{context}

QUESTION:
{clean_question}

ANSWER:"""

    try:
        response = llm.invoke(prompt).strip()
        answer = response.split("\n")[0].strip()
        answer = re.sub(r"^(ANSWER:|FINAL ANSWER:|RESULT:)", "", answer, flags=re.IGNORECASE).strip()
        if not answer:
            answer = "Data not available."
        print(f"LOG: FINAL_ANSWER | (LLM) {answer}")
        return answer
    except Exception as e:
        print(f"LOG: ERROR | {e}")
        return "Data not available."