# test_rag.py
from utils.rag_engine import build_vector_store, search_capabilities

print("🔨 Building vector store from Capability Library...")
build_vector_store()

print("\n🔍 Test Query 1: 'cybersecurity compliance and ISO certification'")
results = search_capabilities("cybersecurity compliance and ISO certification", top_k=3)
for r in results:
    print(f"  [{r['cap_id']}] {r['domain']} | Score: {r['score']:.3f} | {r['certification']}")

print("\n🔍 Test Query 2: 'hospital IT system and medical software'")
results = search_capabilities("hospital IT system and medical software", top_k=3)
for r in results:
    print(f"  [{r['cap_id']}] {r['domain']} | Score: {r['score']:.3f}")

print("\n🎉 Step 2 Complete! RAG engine is working.")