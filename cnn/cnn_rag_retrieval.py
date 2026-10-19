# ============================================================
# CNN -> RAG RETRIEVAL
# Medical AI Project
# ============================================================

import os
import chromadb

from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

CHROMA_PATH = "models/chroma_db"

COLLECTION_NAME = "medical_knowledge"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

TOP_K = 5


# ============================================================
# CNN PREDICTION
# ============================================================

DATASET_NAME = "organsmnist"

PREDICTED_CLASS = 5

PREDICTED_LABEL = "kidney-right"

CONFIDENCE = 35.50


# ============================================================
# CREATE RAG QUERY
# ============================================================

def create_rag_query():

    query = (
        f"Explain the anatomy of the {PREDICTED_LABEL}. "
        f"Describe its location, structure, function, "
        f"blood supply, and important clinical information."
    )

    return query


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print()
print("=" * 65)
print("              CNN -> RAG RETRIEVAL")
print("=" * 65)

print()
print("Loading embedding model...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)

print("Embedding model loaded.")


# ============================================================
# CONNECT TO CHROMADB
# ============================================================

print()
print("Connecting to ChromaDB...")

client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

collection = client.get_collection(
    name=COLLECTION_NAME
)

print("ChromaDB connected.")

print(
    "Total records:",
    collection.count()
)


# ============================================================
# CREATE QUERY
# ============================================================

query = create_rag_query()

print()
print("=" * 65)
print("CNN RESULT")
print("=" * 65)

print("Dataset    :", DATASET_NAME)
print("Class ID   :", PREDICTED_CLASS)
print("Prediction :", PREDICTED_LABEL)
print("Confidence :", f"{CONFIDENCE:.2f}%")


print()
print("=" * 65)
print("RAG QUERY")
print("=" * 65)

print(query)


# ============================================================
# CREATE EMBEDDING
# ============================================================

print()
print("Creating query embedding...")

query_embedding = embedding_model.encode(
    query
).tolist()

print(
    "Embedding created."
)

print(
    "Embedding dimensions:",
    len(query_embedding)
)


# ============================================================
# SEARCH CHROMADB
# ============================================================

print()
print("=" * 65)
print("SEARCHING CHROMADB")
print("=" * 65)

results = collection.query(
    query_embeddings=[
        query_embedding
    ],

    n_results=TOP_K,

    where={
        "subject": "Anatomy"
    }
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

documents = results.get(
    "documents",
    [[]]
)[0]

metadatas = results.get(
    "metadatas",
    [[]]
)[0]

distances = results.get(
    "distances",
    [[]]
)[0]


print()
print("=" * 65)
print("RETRIEVED MEDICAL INFORMATION")
print("=" * 65)


if len(documents) == 0:

    print()
    print("No documents found.")

else:

    for i, document in enumerate(documents):

        print()
        print(
            f"RESULT {i + 1}"
        )

        print("-" * 65)

        print(
            "Document:"
        )

        print(document)

        print()

        if i < len(metadatas):

            metadata = metadatas[i]

            print(
                "Source:",
                metadata.get(
                    "source",
                    "Unknown"
                )
            )

            print(
                "Topic:",
                metadata.get(
                    "topic",
                    "Unknown"
                )
            )

            print(
                "Subtopic:",
                metadata.get(
                    "subtopic",
                    "Unknown"
                )
            )

            print(
                "Verification:",
                metadata.get(
                    "verification_status",
                    "Unknown"
                )
            )

            print(
                "Source URL:",
                metadata.get(
                    "source_url",
                    "Not available"
                )
            )

        if i < len(distances):

            print(
                "Distance:",
                distances[i]
            )


# ============================================================
# COMBINE CONTEXT FOR GEMINI
# ============================================================

print()
print("=" * 65)
print("RAG CONTEXT FOR GEMINI")
print("=" * 65)


context_parts = []


for i, document in enumerate(documents):

    metadata = {}

    if i < len(metadatas):

        metadata = metadatas[i]

    source = metadata.get(
        "source",
        "Unknown source"
    )

    topic = metadata.get(
        "topic",
        "Unknown topic"
    )

    context_parts.append(
        f"""
SOURCE {i + 1}
Source: {source}
Topic: {topic}

Medical information:
{document}
"""
    )


context = "\n".join(
    context_parts
)


print(context)


# ============================================================
# FINAL STATUS
# ============================================================

print()
print("=" * 65)
print("CNN -> RAG RETRIEVAL COMPLETED")
print("=" * 65)

print()
print("CNN prediction:")
print(
    f"{PREDICTED_LABEL} ({CONFIDENCE:.2f}%)"
)

print()
print(
    "Retrieved documents:",
    len(documents)
)

print()
print("Next stage:")
print("RAG Context -> Gemini -> Final Explanation")

print()