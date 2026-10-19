import os
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
from google import genai
from dotenv import load_dotenv


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / "backend" / ".env"
DB_DIR = BASE_DIR / "models" / "chroma_db"


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv(ENV_FILE)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("ERROR: GEMINI_API_KEY not found.")
    print(f"Check your .env file: {ENV_FILE}")
    exit()


# ============================================================
# GEMINI CLIENT
# ============================================================

gemini_client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("=" * 60)
print("MEDICAL AI - RAG GENERATOR")
print("=" * 60)

print("\nLoading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded successfully.")


# ============================================================
# CONNECT TO CHROMADB
# ============================================================

print("\nConnecting to ChromaDB...")

client = chromadb.PersistentClient(
    path=str(DB_DIR)
)

collection = client.get_collection(
    name="medical_knowledge"
)

print(
    "ChromaDB records:",
    collection.count()
)


# ============================================================
# USER INPUT
# ============================================================

subject = input(
    "\nEnter subject: "
).strip()

question = input(
    "Enter your question: "
).strip()


if not subject:
    print("ERROR: Subject cannot be empty.")
    exit()


if not question:
    print("ERROR: Question cannot be empty.")
    exit()


# ============================================================
# CREATE QUESTION EMBEDDING
# ============================================================

print("\nSearching medical knowledge...")

query_embedding = embedding_model.encode(
    question
).tolist()


# ============================================================
# RAG RETRIEVAL
# ============================================================

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5,
    where={
        "subject": subject
    }
)


# ============================================================
# CHECK RETRIEVAL
# ============================================================

documents = results.get("documents", [[]])[0]
metadatas = results.get("metadatas", [[]])[0]


if not documents:

    print("\n" + "=" * 60)
    print("NO INFORMATION FOUND")
    print("=" * 60)

    print(
        f"\nNo medical information was found "
        f"for subject: {subject}"
    )

    print(
        "\nCheck that the subject name matches "
        "the dataset exactly."
    )

    exit()


print(
    f"Retrieved {len(documents)} relevant records."
)


# ============================================================
# BUILD RAG CONTEXT
# ============================================================

context_parts = []

for i, document in enumerate(documents):

    metadata = metadatas[i]

    context_parts.append(
        f"""
SOURCE {i + 1}

Subject:
{metadata.get("subject", "")}

Topic:
{metadata.get("topic", "")}

Subtopic:
{metadata.get("subtopic", "")}

Source:
{metadata.get("source", "")}

Source URL:
{metadata.get("source_url", "")}

Verification:
{metadata.get("verification_status", "")}

Medical Information:
{document}
"""
    )


context = "\n".join(context_parts)


# ============================================================
# GEMINI PROMPT
# ============================================================

prompt = f"""
You are a medical education AI assistant.

Your task is to answer a student's medical question
using the retrieved medical knowledge provided below.

IMPORTANT RULES:

1. Use the retrieved medical information as your primary
   knowledge source.

2. Do not invent medical facts that are not supported
   by the retrieved information.

3. If the retrieved information is insufficient to answer
   the question, clearly say that the available knowledge
   is insufficient.

4. Give a clear and easy-to-understand explanation.

5. Use medical terminology when appropriate, but explain
   difficult terms simply.

6. Organize the answer with headings or bullet points
   when useful.

7. This system is for medical education only.

8. Do not diagnose a real patient.

9. Do not provide personalized treatment instructions.

10. Do not claim certainty when the retrieved information
    does not support it.

Student's subject:
{subject}

Student's question:
{question}

Retrieved medical knowledge:
{context}

Now answer the student's question.
"""


# ============================================================
# SEND TO GEMINI
# ============================================================

print("\nGenerating answer with Gemini...")

try:

    response = gemini_client.models.generate_content(
        model="gemini-3.7-flash",
        contents=prompt
    )

except Exception as e:

    print("\nERROR while calling Gemini:")
    print(e)
    exit()


# ============================================================
# DISPLAY ANSWER
# ============================================================

print("\n")
print("=" * 60)
print("MEDICAL AI ANSWER")
print("=" * 60)

print("\n")
print(response.text)


# ============================================================
# DISPLAY SOURCES
# ============================================================

print("\n")
print("=" * 60)
print("SOURCES USED")
print("=" * 60)

for i, metadata in enumerate(metadatas):

    print(f"\nSource {i + 1}:")
    print(
        metadata.get(
            "source",
            "Unknown source"
        )
    )

    print(
        "Topic:",
        metadata.get(
            "topic",
            ""
        )
    )

    print(
        "URL:",
        metadata.get(
            "source_url",
            ""
        )
    )


# ============================================================
# FINISHED
# ============================================================

print("\n")
print("=" * 60)
print("RAG + GEMINI COMPLETED")
print("=" * 60)