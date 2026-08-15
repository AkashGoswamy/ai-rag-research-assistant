from src.ingestion import DocumentStore
from src.retrieval import retrieve

store = DocumentStore()
store.add_pdf("test_data/attention_is_all_you_need.pdf", doc_id="attention")
store.add_pdf("test_data/bert_paper.pdf", doc_id="bert")

test_queries = [
    ("What is the Transformer architecture based on?", None),
    ("How does BERT handle bidirectional context?", None),
    ("What is self-attention?", None),
    ("What is the capital of France?", None),  # out-of-scope
    ("How do you bake a chocolate cake?", None),  # out-of-scope
    ("What is masked language modeling?", ["bert"]),  # doc-filtered
]

for query, doc_filter in test_queries:
    print("=" * 80)
    print(f"QUERY: {query}")
    if doc_filter:
        print(f"doc_filter: {doc_filter}")
    print("=" * 80)

    results = retrieve(query, store, doc_filter=doc_filter, k=4)

    print(f"Results returned: {len(results)}")
    for r in results:
        print(f"  [{r.doc_name} p.{r.page_number}] {r.text[:80]!r}")
    print()
