import os
import time
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
from google import genai
from dotenv import load_dotenv

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


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
    raise ValueError(
        "GEMINI_API_KEY not found in backend/.env"
    )


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
print("MEDICAL AI BACKEND")
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

chroma_client = chromadb.PersistentClient(
    path=str(DB_DIR)
)

collection = chroma_client.get_collection(
    name="medical_knowledge"
)

print(
    f"ChromaDB records: {collection.count()}"
)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Medical AI API",
    description="RAG-based Medical Education AI",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST MODEL
# ============================================================

class QuestionRequest(BaseModel):

    subject: str
    question: str


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def home():

    return {
        "message": "Medical AI API is running",
        "status": "success"
    }


# ============================================================
# ASK MEDICAL QUESTION
# ============================================================

@app.post("/api/ask")
def ask_question(request: QuestionRequest):

    # --------------------------------------------------------
    # GET USER INPUT
    # --------------------------------------------------------

    subject = request.subject.strip()
    question = request.question.strip()


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if not subject:

        return {
            "success": False,
            "error": "Subject is required"
        }


    if not question:

        return {
            "success": False,
            "error": "Question is required"
        }


    print("\n" + "=" * 60)
    print("NEW MEDICAL QUESTION")
    print("=" * 60)

    print("Subject:", subject)
    print("Question:", question)


    # ========================================================
    # CREATE QUESTION EMBEDDING
    # ========================================================

    print("\nCreating question embedding...")

    query_embedding = embedding_model.encode(
        question
    ).tolist()


    # ========================================================
    # RAG RETRIEVAL
    # ========================================================

    print("Searching ChromaDB...")

    try:

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5,
            where={
                "subject": subject
            }
        )

    except Exception as e:

        return {
            "success": False,
            "error": f"ChromaDB retrieval error: {str(e)}"
        }


    # ========================================================
    # GET RETRIEVED DOCUMENTS
    # ========================================================

    documents = results.get(
        "documents",
        [[]]
    )[0]

    metadatas = results.get(
        "metadatas",
        [[]]
    )[0]


    print(
        f"Retrieved {len(documents)} records."
    )


    # ========================================================
    # NO RESULTS
    # ========================================================

    if not documents:

        return {
            "success": False,
            "error": (
                f"No medical information found "
                f"for subject: {subject}"
            )
        }


    # ========================================================
    # BUILD RAG CONTEXT
    # ========================================================

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

Verification Status:
{metadata.get("verification_status", "")}

Medical Information:
{document}
"""
        )


    context = "\n".join(
        context_parts
    )


    # ========================================================
    # GEMINI PROMPT
    # ========================================================

    prompt = f"""
You are a Medical Education AI Assistant.

Your job is to help medical students understand
medical subjects using the retrieved medical knowledge.

IMPORTANT RULES:

1. Use the retrieved medical information as your
   primary knowledge source.

2. Do not invent unsupported medical facts.

3. If the retrieved information is insufficient,
   clearly state that the available information
   is insufficient.

4. Give a clear educational explanation.

5. Explain difficult medical terminology in simple
   language when appropriate.

6. Use headings and bullet points when useful.

7. Keep the answer focused on the student's question.

8. This system is for medical education only.

9. Do not diagnose a real patient.

10. Do not provide personalized treatment instructions.

11. Do not claim certainty when the retrieved
    information does not support it.

12. Mention important clinical points only when
    supported by the retrieved information.

STUDENT SUBJECT:
{subject}

STUDENT QUESTION:
{question}

RETRIEVED MEDICAL KNOWLEDGE:
{context}

Now provide a clear and educational answer
for the medical student.
"""


    # ========================================================
    # GEMINI MODELS
    # ========================================================

    models = [
        "gemini-3.8-flash",
        "gemini-3.5-flash"
    ]


    answer = None
    last_error = None
    successful_model = None


    # ========================================================
    # GEMINI GENERATION WITH RETRY
    # ========================================================

    print("\nGenerating answer with Gemini...")


    for model_name in models:

        print(
            f"\nTrying model: {model_name}"
        )


        for attempt in range(3):

            try:

                print(
                    f"Attempt {attempt + 1}/3..."
                )


                response = (
                    gemini_client
                    .models
                    .generate_content(
                        model=model_name,
                        contents=prompt
                    )
                )


                # --------------------------------------------
                # CHECK RESPONSE
                # --------------------------------------------

                if response.text:

                    answer = response.text

                    successful_model = model_name

                    print(
                        f"SUCCESS: {model_name}"
                    )

                    break


                else:

                    last_error = (
                        "Gemini returned an empty response."
                    )


            except Exception as e:

                last_error = str(e)

                print(
                    f"Gemini error: {last_error}"
                )


            # --------------------------------------------
            # RETRY DELAY
            # --------------------------------------------

            if attempt < 2:

                delay = 2 ** attempt

                print(
                    f"Retrying in {delay} seconds..."
                )

                time.sleep(delay)


        # ----------------------------------------------------
        # STOP IF SUCCESSFUL
        # ----------------------------------------------------

        if answer:

            break


        print(
            f"{model_name} was unsuccessful."
        )


    # ========================================================
    # ALL GEMINI MODELS FAILED
    # ========================================================

    if not answer:

        print("\nAll Gemini attempts failed.")

        return {
            "success": False,
            "error": (
                "Gemini is temporarily unavailable. "
                "Please try again later."
            ),
            "details": last_error
        }


    # ========================================================
    # BUILD SOURCES
    # ========================================================

    sources = []


    for metadata in metadatas:

        sources.append(
            {
                "source": metadata.get(
                    "source",
                    ""
                ),

                "topic": metadata.get(
                    "topic",
                    ""
                ),

                "subtopic": metadata.get(
                    "subtopic",
                    ""
                ),

                "source_url": metadata.get(
                    "source_url",
                    ""
                ),

                "verification_status": metadata.get(
                    "verification_status",
                    ""
                )
            }
        )


    # ========================================================
    # FINAL RESPONSE
    # ========================================================

    print("\n" + "=" * 60)
    print("ANSWER GENERATED SUCCESSFULLY")
    print("=" * 60)

    print(
        "Model:",
        successful_model
    )


    return {

        "success": True,

        "subject": subject,

        "question": question,

        "answer": answer,

        "model": successful_model,

        "sources": sources

    }


# ============================================================
# END OF SERVER
# ============================================================