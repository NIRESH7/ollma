from rag import _tabular_row_lookup, load_structured_payloads
candidates = _tabular_row_lookup("What is the coverage for INSO UEMS PLAN A?", "All")
print(candidates)
