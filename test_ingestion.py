from src.ingestion import ingest_pdf

test_files = [
    ("test_data/attention_is_all_you_need.pdf", "attention"),
    ("test_data/bert_paper.pdf", "bert"),
    ("test_data/rag_original_paper.pdf", "rag_original"),
]

for pdf_path, doc_id in test_files:
    print("=" * 80)
    print(f"INGESTING: {pdf_path}")
    print("=" * 80)

    chunks = ingest_pdf(pdf_path, doc_id=doc_id)

    print(f"Total chunks: {len(chunks)}")
    print()

    # Print first 3 chunks in full to eyeball boundaries and overlap
    for c in chunks[:3]:
        print(f"--- chunk_index={c.chunk_index} | page={c.page_number} | doc_id={c.doc_id} | len={len(c.text)} ---")
        print(c.text)
        print()

    # Print page numbers seen, to sanity check page-tracking
    pages_seen = sorted(set(c.page_number for c in chunks))
    print(f"Pages with chunks: {pages_seen}")
    print()
