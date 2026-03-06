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

# ── Configuration ──────────────────────────────────────────────────
COLLECTION_NAME = "local_documents"
OLLAMA_MODEL = "qwen2:1.5b"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# ── Model Singletons ───────────────────────────────────────────────
def _get_best_model():
    try:
        import requests
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        if resp.status_code == 200:
            models = [m["name"] for m in resp.json().get("models", [])]
            if OLLAMA_MODEL in models:
                return OLLAMA_MODEL
            # Find any suitable fallback model
            for m in models:
                if "qwen" in m or "llama3" in m or "llama2" in m or "mistral" in m:
                    return m
            if models:
                return models[0]
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
    tokens = re.split(r"[\W_]+", (text or "").lower())
    return [t for t in tokens if len(t) > 1 or t.isdigit()]


def _value_match_score(question_tokens: List[str], text: str) -> float:
    blob = (text or "").lower()
    score = 0.0
    for token in question_tokens:
        if token in blob:
            score += 2.0 if len(token) >= 5 else 1.0
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
    bonus += overlap * 0.7

    if "inpatient" in target_tokens:
        bonus += 1.8 if "inpatient" in col_tokens else -0.9
    if "outpatient" in target_tokens:
        bonus += 1.8 if "outpatient" in col_tokens else -0.9
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

    for col, val in values.items():
        if col == answer_col:
            continue
        text = str(val).strip().lower()
        if not text:
            continue

        if len(text) > 2 and text in question_norm:
            score += 2.5
            continue

        value_tokens = set(_tokenize(text))
        overlap = len(value_tokens.intersection(token_set))
        if overlap:
            score += float(overlap)

    return score


def _format_multi_sheet_answers(candidates: List[Dict[str, Any]]) -> Optional[str]:
    if not candidates:
        return None
    candidates.sort(key=lambda x: x["score"], reverse=True)
    top_score = candidates[0]["score"]
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
    all_values = {str(c["value"]).strip().lower() for c in deduped}
    if len(all_values) == 1:
        return str(deduped[0]["value"])
    sheet_groups = {}
    for c in deduped:
        sheet_groups.setdefault(c["sheet_name"], []).append(str(c["value"]))
    lines = []
    for sheet, vals in sheet_groups.items():
        unique_vals = list(dict.fromkeys(vals))
        lines.append(f"[{sheet}] " + " | ".join(unique_vals))
    return "\n".join(lines)


def _extract_sheet_name(question: str, folder_name: str) -> Optional[str]:
    q_norm = question.lower()
    payloads = load_structured_payloads(folder_name=folder_name)
    best_sheet = None
    best_len = 0
    for payload in payloads:
        for sheet_name in payload.get("sheets", {}).keys():
            s_norm = sheet_name.lower()
            if s_norm in q_norm and len(s_norm) > best_len and len(s_norm) > 3:
                best_len = len(s_norm)
                best_sheet = sheet_name
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

                for answer_col, answer_col_score in ranked_columns[:6]:
                    if answer_col_score < 0.45:
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

    overlap = len(set(question_tokens) & set(key_tokens)) / len(set(key_tokens))
    fuzzy = SequenceMatcher(None, question_norm, key_label).ratio()
    contains = 1.0 if any(kt in question_norm for kt in key_tokens) else 0.0
    return max(fuzzy, overlap + (0.2 * contains))


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
            if _field_score(q_norm, q_tokens, first_col_key) >= 0.45:
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

            # 3) Otherwise, find row whose first-column label matches requested field.
            best_row_value = None
            best_row_score = 0.0
            for record in records:
                values = record.get("values", {})
                row_label = str(values.get(first_col_key, ""))
                row_score = _value_match_score(q_tokens, row_label)
                if row_score <= 0:
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
            if raw_rows and re.search(r"\b(plan code|pmcare)\b", q_norm):
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


def _suspicious_direct_answer(question: str, answer: str) -> bool:
    q = (question or "").lower()
    a = (answer or "").strip().lower()
    if not a:
        return True

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


def _collect_structured_candidates(question: str, folder_name: str, limit: int = 60) -> List[str]:
    q_tokens = _tokenize(question)
    scored: List[Tuple[float, str]] = []

    payloads = load_structured_payloads(folder_name=folder_name)
    for payload in payloads:
        file_name = payload.get("file_name", "")
        for sheet_name, sheet_data in payload.get("sheets", {}).items():
            for row_bucket in ("records", "raw_rows"):
                for record in sheet_data.get(row_bucket, []):
                    values = record.get("values", {})
                    if not isinstance(values, dict) or not values:
                        continue
                    row_text = " | ".join([f"{k}: {v}" for k, v in values.items()])
                    context = f"File: {file_name} | Sheet: {sheet_name} | {row_text}"
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
        if len(unique_contexts) >= limit:
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
- "answer": concise exact value string. If the answer comes from a specific sheet, include the sheet name in brackets (e.g., "[Sheet Name] value"). If multiple results exist, list them clearly separated by newlines. If not found, use exactly "The requested information is not available in the provided spreadsheet data."
- "evidence": exact row text you used.
- "confidence": number from 0 to 1.

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

    # Guardrails: answer should appear in evidence/context and confidence should be decent.
    normalized_context = "\n".join(candidates).lower()
    if answer.lower() not in normalized_context:
        return None
    if evidence and answer.lower() not in evidence.lower():
        return None
    if confidence < 0.55:
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
    if "plan_code" in intents:
        matrix_cands = _matrix_lookup(question, folder_name, target_sheet)
        if matrix_cands:
            ans = _format_multi_sheet_answers(matrix_cands)
            if ans: return ans

    # Tabular row lookup handles "for group X plan Y give me headcount" queries.
    table_cands = _tabular_row_lookup(question, folder_name, target_sheet)
    if table_cands:
        ans = _format_multi_sheet_answers(table_cands)
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
        return _format_multi_sheet_answers(mc) if mc else None

    # Sort and filter fallbacks
    candidates.sort(key=lambda x: x["score"], reverse=True)
    best_candidate = candidates[0]

    # Strict threshold to avoid hallucinated field mapping.
    if best_candidate["row_score"] < 2.0:
        return None

    # Build final formatted return similarly to previous simple logic but allowing multiple sheets if they tied
    high_scoring = [c for c in candidates if c["score"] >= best_candidate["score"] - 1.0 and c["row_score"] >= 2.0]
    
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
                "score": cd["score"]
            })

    if extracted_vals:
        return _format_multi_sheet_answers(extracted_vals)

    # Cross-row matrix fallback.
    mc = _matrix_lookup(question, folder_name, target_sheet)
    return _format_multi_sheet_answers(mc) if mc else None


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


# ── Query Entry ────────────────────────────────────────────────────
def query_rag(question: str, folder_name: str = "All") -> str:
    print(f"\nLOG: QUERY_RECEIVED | {question}")
    _ = time.time()

    print("LOG: FLOW | Normalization")
    clean_question = question.replace("garde", "grade").replace("JL12", "JL 12").replace("emp", "employee")

    print("LOG: FLOW | Intent Detection")
    intents = _detect_intents(clean_question)
    
    print("LOG: FLOW | Entity Extraction")
    entities = _extract_explicit_entities(clean_question)

    print("LOG: FLOW | Sheet Detection")
    target_sheet = _extract_sheet_name(clean_question, folder_name)
    if target_sheet:
        print(f"LOG: SHEET DETECTED | Restricting lookup to sheet: {target_sheet}")

    # STEP 1: Structured deterministic lookup
    print("LOG: FLOW | JSON Lookup")
    direct_match = _structured_lookup(clean_question, folder_name, target_sheet)
    if direct_match:
        if not _suspicious_direct_answer(clean_question, direct_match):
            print("LOG: FLOW | Found (Return)")
            print(f"LOG: FINAL_ANSWER | (Structured) {direct_match}")
            return direct_match
        print(f"LOG: STRUCTURED_LOOKUP | Suspicious direct answer ignored: {direct_match}")

    print("LOG: FLOW | Not Found -> Vector Search / AI")

    # STEP 2: Numeric engine
    if is_numeric_question(clean_question):
        print("LOG: FLOW | Math Engine Computation")
        math_result = run_dataframe_query(clean_question, folder_name)
        if math_result:
            print(f"LOG: FINAL_ANSWER | (Math Engine) {math_result}")
            return math_result

    # STEP 2.5: Structured-context LLM fallback (offline, no vector DB required)
    print("LOG: FLOW | Vector Search (Structured Candidates)")
    llm_structured = _llm_structured_fallback(clean_question, folder_name)
    if llm_structured:
        print("LOG: FLOW | LLM -> Final Answer")
        print(f"LOG: FINAL_ANSWER | (LLM Structured) {llm_structured}")
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