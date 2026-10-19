import chromadb
from sentence_transformers import SentenceTransformer


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")


# ============================================================
# CONNECT TO CHROMADB
# ============================================================

client = chromadb.PersistentClient(
    path="models/chroma_db"
)

collection = client.get_collection(
    name="medical_knowledge"
)


# ============================================================
# USER INPUT
# ============================================================

subject = input("Enter subject: ").strip()
question = input("Enter your question: ").strip()


# ============================================================
# CREATE QUESTION EMBEDDING
# ============================================================

query_embedding = model.encode(
    question
).tolist()


# ============================================================
# RETRIEVE FROM SELECTED SUBJECT
# ============================================================

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5,
    where={"subject": subject}
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 60)
print("RAG RETRIEVAL RESULTS")
print("=" * 60)

print("\nSubject:", subject)
print("Question:", question)


documents = results["documents"][0]
metadatas = results["metadatas"][0]


for i, document in enumerate(documents):

    print("\n" + "-" * 60)
    print(f"RESULT {i + 1}")
    print("-" * 60)

    print("Content:")
    print(document)

    print("\nMetadata:")
    print(metadatas[i])


print("\n" + "=" * 60)
print("RETRIEVAL COMPLETED")
print("=" * 60)