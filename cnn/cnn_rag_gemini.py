# ============================================================
# CNN -> RAG -> GEMINI
# Medical AI Project
# ============================================================

import os
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
from google import genai
from dotenv import load_dotenv


# ============================================================
# 1. PROJECT PATHS
# ============================================================

# This file is:
# Medical_AI/cnn/cnn_rag_gemini.py
#
# Therefore:
# parent       = Medical_AI/cnn
# parent.parent = Medical_AI

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BACKEND_ENV = PROJECT_ROOT / "backend" / ".env"

CHROMA_PATH = PROJECT_ROOT / "models" / "chroma_db"


# ============================================================
# 2. LOAD .ENV
# ============================================================

print()
print("=" * 70)
print("              CNN -> RAG -> GEMINI")
print("=" * 70)

print()
print("Project folder:")
print(PROJECT_ROOT)

print()
print("Looking for .env:")
print(BACKEND_ENV)


if not BACKEND_ENV.exists():

    raise FileNotFoundError(
        f"\n.env file not found here:\n{BACKEND_ENV}\n\n"
        "Make sure your Gemini API key is inside:\n"
        "Medical_AI/backend/.env"
    )


load_dotenv(
    dotenv_path=BACKEND_ENV
)


# ============================================================
# 3. GET GEMINI API KEY
# ============================================================

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY"
)


if not GEMINI_API_KEY:

    raise ValueError(
        "\nGEMINI_API_KEY was not found in:\n"
        f"{BACKEND_ENV}\n\n"
        "Your backend/.env should contain:\n"
        "GEMINI_API_KEY=your_key_here"
    )


print()
print("Gemini API key loaded successfully.")


# ============================================================
# 4. GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=GEMINI_API_KEY
)

print("Gemini client created successfully.")


# ============================================================
# 5. CONFIGURATION
# ============================================================

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

COLLECTION_NAME = "medical_knowledge"

TOP_K = 5

GEMINI_MODEL = "gemini-3.5-flash"


# ============================================================
# 6. CNN PREDICTION
# ============================================================

# This is the result from your OrganSMNIST prediction.

DATASET_NAME = "organsmnist"

PREDICTED_CLASS = 5

PREDICTED_LABEL = "kidney-right"

CONFIDENCE = 35.50

RAG_SUBJECT = "Anatomy"


# ============================================================
# 7. CREATE RAG QUERY
# ============================================================

def create_rag_query():

    query = (
        f"Explain the anatomy of the {PREDICTED_LABEL}. "
        f"Describe its location, structure, function, "
        f"blood supply, and important clinical information."
    )

    return query


# ============================================================
# 8. LOAD EMBEDDING MODEL
# ============================================================

print()
print("=" * 70)
print("LOADING EMBEDDING MODEL")
print("=" * 70)

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)

print()
print("Embedding model loaded successfully.")


# ============================================================
# 9. CONNECT TO CHROMADB
# ============================================================

print()
print("=" * 70)
print("CONNECTING TO CHROMADB")
print("=" * 70)

print()
print("ChromaDB path:")
print(CHROMA_PATH)


if not CHROMA_PATH.exists():

    raise FileNotFoundError(
        f"\nChromaDB folder not found:\n{CHROMA_PATH}"
    )


chroma_client = chromadb.PersistentClient(
    path=str(CHROMA_PATH)
)


try:

    collection = chroma_client.get_collection(
        name=COLLECTION_NAME
    )

except Exception as error:

    raise RuntimeError(
        "\nCould not find ChromaDB collection "
        f"'{COLLECTION_NAME}'.\n\n"
        f"Original error:\n{error}"
    )


print()
print("ChromaDB connected successfully.")

print(
    "Total records:",
    collection.count()
)


# ============================================================
# 10. CREATE RAG QUERY
# ============================================================

query = create_rag_query()


print()
print("=" * 70)
print("CNN RESULT")
print("=" * 70)

print(
    "Dataset    :",
    DATASET_NAME
)

print(
    "Class ID   :",
    PREDICTED_CLASS
)

print(
    "Prediction :",
    PREDICTED_LABEL
)

print(
    "Confidence :",
    f"{CONFIDENCE:.2f}%"
)


print()
print("=" * 70)
print("RAG QUERY")
print("=" * 70)

print(query)


# ============================================================
# 11. CREATE QUERY EMBEDDING
# ============================================================

print()
print("=" * 70)
print("CREATING QUERY EMBEDDING")
print("=" * 70)

query_embedding = embedding_model.encode(
    query
).tolist()

print()
print("Embedding created successfully.")

print(
    "Embedding dimensions:",
    len(query_embedding)
)


# ============================================================
# 12. SEARCH CHROMADB
# ============================================================

print()
print("=" * 70)
print("SEARCHING CHROMADB")
print("=" * 70)

try:

    results = collection.query(

        query_embeddings=[
            query_embedding
        ],

        n_results=TOP_K,

        where={
            "subject": RAG_SUBJECT
        }
    )

except Exception as error:

    raise RuntimeError(
        "\nChromaDB search failed.\n\n"
        f"Original error:\n{error}"
    )


# ============================================================
# 13. GET RESULTS
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


if not documents:

    raise RuntimeError(
        "\nNo relevant medical documents were retrieved."
    )


print()
print(
    "Retrieved documents:",
    len(documents)
)


# ============================================================
# 14. DISPLAY RETRIEVED INFORMATION
# ============================================================

print()
print("=" * 70)
print("RETRIEVED MEDICAL INFORMATION")
print("=" * 70)


for i, document in enumerate(documents):

    print()
    print(
        f"RESULT {i + 1}"
    )

    print(
        "-" * 70
    )

    print(
        "Document:"
    )

    print(document)


    if i < len(metadatas):

        metadata = metadatas[i]

        print()

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
# 15. BUILD RAG CONTEXT
# ============================================================

context_parts = []


for i, document in enumerate(documents):

    metadata = {}

    if i < len(metadatas):

        metadata = metadatas[i]


    source = metadata.get(
        "source",
        "Unknown"
    )

    topic = metadata.get(
        "topic",
        "Unknown"
    )

    subtopic = metadata.get(
        "subtopic",
        "Unknown"
    )

    verification = metadata.get(
        "verification_status",
        "Unknown"
    )

    source_url = metadata.get(
        "source_url",
        "Not available"
    )


    context_parts.append(
        f"""
SOURCE {i + 1}

Source: {source}

Topic: {topic}

Subtopic: {subtopic}

Verification: {verification}

Source URL: {source_url}

Medical information:
{document}
"""
    )


context = "\n".join(
    context_parts
)


# ============================================================
# 16. CREATE GEMINI PROMPT
# ============================================================

prompt = f"""
You are a medical education AI assistant.

Your purpose is to provide a clear educational
explanation for medical students.

The system used a CNN image-classification model.

CNN prediction:

Dataset:
{DATASET_NAME}

Predicted class:
{PREDICTED_CLASS}

Predicted label:
{PREDICTED_LABEL}

CNN confidence:
{CONFIDENCE:.2f}%

IMPORTANT:
The CNN prediction is NOT a confirmed medical diagnosis.

The confidence is only {CONFIDENCE:.2f}%, so clearly
state that the image classification is uncertain.

Use the retrieved medical information below as the
factual foundation for your answer.

Do not invent facts that are not supported by the
retrieved information.

If the retrieved information does not contain enough
information for a requested section, clearly say that
the retrieved sources do not provide enough information.

Create a student-friendly explanation using these sections:

## 1. CNN Prediction

Explain what the CNN predicted and include the
confidence.

## 2. Important Uncertainty

Explain that the CNN prediction is uncertain and
should not be treated as a confirmed diagnosis.

## 3. What Is the Organ?

Explain the organ using the retrieved information.

## 4. Location

Explain its anatomical location if supported by
the retrieved information.

## 5. Structure

Explain its structure if supported by the retrieved
information.

## 6. Function

Explain its function.

## 7. Clinical Importance

Explain important clinical information that is
supported by the retrieved sources.

## 8. Summary

Give a short student-friendly summary.

## 9. Sources

List the sources used from the retrieved context.

------------------------------------------------------------
RETRIEVED MEDICAL KNOWLEDGE
------------------------------------------------------------

{context}

------------------------------------------------------------
END OF RETRIEVED MEDICAL KNOWLEDGE
------------------------------------------------------------
"""


# ============================================================
# 17. SEND CONTEXT TO GEMINI
# ============================================================

print()
print("=" * 70)
print("SENDING RAG CONTEXT TO GEMINI")
print("=" * 70)

print()
print(
    "Gemini model:",
    GEMINI_MODEL
)


try:

    response = client.models.generate_content(

        model=GEMINI_MODEL,

        contents=prompt
    )

except Exception as error:

    print()
    print("=" * 70)
    print("GEMINI ERROR")
    print("=" * 70)

    print()
    print(error)

    raise


# ============================================================
# 18. GET GEMINI ANSWER
# ============================================================

answer = response.text


if not answer:

    raise RuntimeError(
        "\nGemini returned an empty response."
    )


# ============================================================
# 19. DISPLAY FINAL ANSWER
# ============================================================

print()
print("=" * 70)
print("              FINAL GEMINI EXPLANATION")
print("=" * 70)

print()

print(answer)


# ============================================================
# 20. DISPLAY SOURCES
# ============================================================

print()
print("=" * 70)
print("SOURCES USED BY RAG")
print("=" * 70)


for i, metadata in enumerate(
    metadatas
):

    print()

    print(
        f"Source {i + 1}"
    )

    print(
        "Name:",
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
        "URL:",
        metadata.get(
            "source_url",
            "Not available"
        )
    )


# ============================================================
# 21. COMPLETED
# ============================================================

print()
print("=" * 70)
print("          CNN -> RAG -> GEMINI COMPLETED")
print("=" * 70)

print()
print("Pipeline:")
print("Image")
print("  ↓")
print("CNN prediction")
print("  ↓")
print("RAG query")
print("  ↓")
print("Embedding")
print("  ↓")
print("ChromaDB")
print("  ↓")
print("Retrieved medical knowledge")
print("  ↓")
print("Gemini")
print("  ↓")
print("Educational explanation")
print()