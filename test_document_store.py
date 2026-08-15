from src.ingestion import DocumentStore

store = DocumentStore()

added1 = store.add_pdf("test_data/attention_is_all_you_need.pdf", doc_id="attention")
print(f"Added from attention paper: {added1} chunks")

added2 = store.add_pdf("test_data/bert_paper.pdf", doc_id="bert")
print(f"Added from bert paper: {added2} chunks")

print()
print(f"Total chunks in store: {store.total_chunks()}")
print(f"FAISS index total vectors: {store.index.ntotal}")
print()
print("Chunks per doc:")
for doc_name, count in store.chunks_per_doc().items():
    print(f"  {doc_name}: {count}")

print()
print("Sample metadata check (first chunk of each doc):")
seen_docs = set()
for c in store.metadata:
    if c.doc_id not in seen_docs:
        seen_docs.add(c.doc_id)
        print(f"  doc_id={c.doc_id} | doc_name={c.doc_name} | page={c.page_number} | text_preview={c.text[:60]!r}")
