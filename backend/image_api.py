# ============================================================
# MEDICAL AI - COMPLETE BACKEND
# ============================================================
#
# Features:
#   1. Medical question answering
#   2. ChromaDB RAG retrieval
#   3. Gemini text generation
#   4. Gemini Vision image understanding
#   5. Automatic medical structure detection
#   6. Automatic Brain / Heart 3D routing
#   7. Source information
#
# IMPORTANT:
# This backend does NOT import pretrainedModel.
# It does NOT require a module named pretrainedModel.
#
# Educational use only.
# ============================================================

import os
import io
import json
import time
import re

from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from google import genai
from google.genai import types

from PIL import Image

from sentence_transformers import SentenceTransformer
import chromadb


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


if not GEMINI_API_KEY:

    raise RuntimeError(
        "GEMINI_API_KEY is missing from backend/.env"
    )


# ============================================================
# GEMINI CLIENT
# ============================================================

print("")
print("==============================================")
print("Initializing Gemini...")
print("==============================================")


client = genai.Client(
    api_key=GEMINI_API_KEY
)


print("Gemini client initialized.")


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="MedicalAI Backend",
    version="2.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(

    CORSMiddleware,

    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)


# ============================================================
# MEDICAL SUBJECTS
# ============================================================

MEDICAL_SUBJECTS = [

    "Anatomy",

    "Physiology",

    "Biochemistry",

    "Pathology",

    "Pharmacology",

    "Microbiology",

    "Forensic Medicine & Toxicology",

    "Community Medicine",

    "General Medicine",

    "General Surgery",

    "Pediatrics",

    "Obstetrics & Gynecology",

    "Orthopedics",

    "Dermatology",

    "Psychiatry",

    "Radiology",

    "Anesthesiology",

    "Ophthalmology",

    "Otorhinolaryngology (ENT)",

    "Dentistry / Oral Medicine",

    "Cardiology",

    "Neurology",

    "Pulmonology",

    "Gastroenterology",

    "Nephrology",

    "Endocrinology",

    "Rheumatology",

    "Oncology",

    "Urology",

    "Emergency Medicine",

]


# ============================================================
# GEMINI MODELS
# ============================================================
#
# We put the lightweight model first because it has worked
# reliably with your current API setup.
#
# If one model is temporarily unavailable, the backend tries
# the next model.
# ============================================================

TEXT_MODELS = [

    "gemini-flash-lite-latest",

    "gemini-2.5-flash-lite",

    "gemini-3.1-flash-lite",

    "gemini-3.5-flash-lite",

    "gemini-flash-latest",

]


VISION_MODELS = [

    "gemini-flash-lite-latest",

    "gemini-2.5-flash-lite",

    "gemini-3.1-flash-lite",

    "gemini-3.5-flash-lite",

    "gemini-flash-latest",

]


# ============================================================
# RETRY SETTINGS
# ============================================================

MAX_RETRIES = 2

RETRY_DELAYS = [
    2,
    5,
]


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("")
print("==============================================")
print("Loading embedding model...")
print("==============================================")


embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


print(
    "Embedding model loaded successfully."
)


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.abspath(

    os.path.join(

        os.path.dirname(__file__),

        ".."

    )

)


# ============================================================
# CHROMADB PATH
# ============================================================

CHROMA_PATH = os.path.join(

    PROJECT_ROOT,

    "models",

    "chroma_db"

)


print("")
print("==============================================")
print("ChromaDB")
print("==============================================")


print(
    "ChromaDB path:"
)

print(
    CHROMA_PATH
)


# ============================================================
# CHECK CHROMADB DIRECTORY
# ============================================================

if not os.path.exists(CHROMA_PATH):

    print("")
    print(
        "WARNING: ChromaDB directory does not exist:"
    )

    print(
        CHROMA_PATH
    )

    print(
        "RAG retrieval may not work until the database "
        "has been created using your ingest.py."
    )


# ============================================================
# CHROMADB CLIENT
# ============================================================

chroma_client = chromadb.PersistentClient(

    path=CHROMA_PATH

)


# ============================================================
# GET COLLECTION
# ============================================================

try:

    collection = chroma_client.get_collection(

        name="medical_knowledge"

    )

except Exception as error:

    print("")
    print(
        "ERROR: Could not load ChromaDB collection."
    )

    print(
        error
    )

    raise RuntimeError(

        "ChromaDB collection 'medical_knowledge' "
        "was not found. Run your RAG ingest.py first."

    )


print(
    "ChromaDB records:",
    collection.count()
)


# ============================================================
# RETRYABLE ERROR CHECK
# ============================================================

def is_retryable_error(error):

    text = str(error).lower()

    retry_words = [

        "503",

        "unavailable",

        "high demand",

        "temporarily",

        "internal server",

        "timeout",

        "429",

        "resource_exhausted",

        "service unavailable",

    ]

    return any(

        word in text

        for word in retry_words

    )


# ============================================================
# GEMINI TEXT GENERATION
# ============================================================

def generate_text(prompt):

    last_error = None


    for model_name in TEXT_MODELS:

        for attempt in range(MAX_RETRIES):

            try:

                print("")
                print("----------------------------------------------")
                print(
                    "Gemini text model:",
                    model_name
                )

                print(
                    "Attempt:",
                    attempt + 1
                )


                response = client.models.generate_content(

                    model=model_name,

                    contents=prompt,

                )


                if response is None:

                    raise RuntimeError(

                        "Gemini returned no response."

                    )


                response_text = response.text


                if not response_text:

                    raise RuntimeError(

                        "Gemini returned empty text."

                    )


                print(
                    "Gemini text generation successful."
                )


                return {

                    "text":
                        response_text,

                    "model":
                        model_name,

                }


            except Exception as error:

                last_error = error


                print("")
                print(
                    "Gemini text error:"
                )

                print(
                    repr(error)
                )


                if is_retryable_error(error):

                    if attempt < MAX_RETRIES - 1:

                        delay = RETRY_DELAYS[

                            min(

                                attempt,

                                len(RETRY_DELAYS) - 1

                            )

                        ]


                        print(

                            f"Retrying in {delay} seconds..."

                        )


                        time.sleep(delay)


                else:

                    break


    raise RuntimeError(

        "All Gemini text models failed. "
        f"Last error: {last_error}"

    )


# ============================================================
# CLEAN GEMINI JSON
# ============================================================

def clean_json(text):

    if not text:

        return {}


    text = text.strip()


    # --------------------------------------------------------
    # Remove markdown JSON fences
    # --------------------------------------------------------

    text = re.sub(

        r"^```json\s*",

        "",

        text,

        flags=re.IGNORECASE

    )


    text = re.sub(

        r"^```\s*",

        "",

        text

    )


    text = re.sub(

        r"\s*```$",

        "",

        text

    )


    # --------------------------------------------------------
    # Find JSON object
    # --------------------------------------------------------

    start = text.find("{")

    end = text.rfind("}")


    if (

        start != -1

        and end != -1

        and end > start

    ):

        text = text[

            start:end + 1

        ]


    # --------------------------------------------------------
    # Parse JSON
    # --------------------------------------------------------

    try:

        return json.loads(text)

    except Exception as error:

        print("")
        print(
            "JSON parsing failed:"
        )

        print(
            repr(error)
        )

        print(
            "Gemini returned:"
        )

        print(
            text
        )

        return {}


# ============================================================
# RAG RETRIEVAL
# ============================================================

def retrieve_medical_context(

    query,

    subject=None

):

    print("")
    print("==============================================")
    print("RAG RETRIEVAL")
    print("==============================================")


    print(
        "Query:"
    )

    print(
        query
    )


    print(
        "Subject:"
    )

    print(
        subject
    )


    # --------------------------------------------------------
    # Create embedding
    # --------------------------------------------------------

    query_embedding = (

        embedding_model

        .encode(query)

        .tolist()

    )


    # --------------------------------------------------------
    # Subject-filtered retrieval
    # --------------------------------------------------------

    if subject:

        try:

            results = collection.query(

                query_embeddings=[

                    query_embedding

                ],

                n_results=5,

                where={

                    "subject":
                        subject

                },

            )


            documents = (

                results

                .get(
                    "documents",
                    [[]]
                )[0]

            )


            metadatas = (

                results

                .get(
                    "metadatas",
                    [[]]
                )[0]

            )


            if documents:

                print(
                    "Subject-filtered documents:",
                    len(documents)
                )

                return (

                    documents,

                    metadatas

                )


        except Exception as error:

            print("")
            print(
                "Subject-filtered RAG failed:"
            )

            print(
                repr(error)
            )


    # --------------------------------------------------------
    # Global fallback
    # --------------------------------------------------------

    try:

        results = collection.query(

            query_embeddings=[

                query_embedding

            ],

            n_results=5,

        )


        documents = (

            results

            .get(
                "documents",
                [[]]
            )[0]

        )


        metadatas = (

            results

            .get(
                "metadatas",
                [[]]
            )[0]

        )


        print(
            "Global RAG documents:",
            len(documents)
        )


        return (

            documents,

            metadatas

        )


    except Exception as error:

        print("")
        print(
            "Global RAG failed:"
        )

        print(
            repr(error)
        )


        return [], []


# ============================================================
# BUILD RAG CONTEXT
# ============================================================

def build_context(

    documents,

    metadatas

):

    if not documents:

        return (

            "No retrieved medical reference "
            "was available."

        )


    context_parts = []


    for index, document in enumerate(

        documents

    ):

        metadata = {}


        if index < len(metadatas):

            metadata = (

                metadatas[index]

                or {}

            )


        source = metadata.get(

            "source",

            "Medical reference"

        )


        topic = metadata.get(

            "topic",

            ""

        )


        subtopic = metadata.get(

            "subtopic",

            ""

        )


        context_parts.append(

            f"""
SOURCE:
{source}

TOPIC:
{topic}

SUBTOPIC:
{subtopic}

CONTENT:
{document}
"""

        )


    return "\n\n".join(

        context_parts

    )


# ============================================================
# BUILD SOURCES
# ============================================================

def build_sources(

    metadatas

):

    sources = []


    for metadata in metadatas:

        if not metadata:

            continue


        sources.append({

            "source":

                metadata.get(

                    "source",

                    "Medical reference"

                ),

            "topic":

                metadata.get(

                    "topic",

                    ""

                ),

            "subtopic":

                metadata.get(

                    "subtopic",

                    ""

                ),

            "source_url":

                metadata.get(

                    "source_url",

                    ""

                ),

            "verification_status":

                metadata.get(

                    "verification_status",

                    ""

                ),

        })


    return sources


# ============================================================
# TEXT QUESTION ENDPOINT
# ============================================================

@app.post("/api/ask")
async def ask_medical_question(

    data: dict

):

    try:

        print("")
        print("==============================================")
        print("TEXT MEDICAL QUESTION")
        print("==============================================")


        # ----------------------------------------------------
        # Read request
        # ----------------------------------------------------

        subject = data.get(

            "subject",

            "Anatomy"

        )


        question = data.get(

            "question",

            ""

        )


        question = str(

            question

        ).strip()


        # ----------------------------------------------------
        # Validate question
        # ----------------------------------------------------

        if not question:

            return {

                "success":
                    False,

                "error":
                    "Please enter a medical question."

            }


        # ----------------------------------------------------
        # Validate subject
        # ----------------------------------------------------

        if subject not in MEDICAL_SUBJECTS:

            subject = "Anatomy"


        print(
            "Subject:",
            subject
        )


        print(
            "Question:",
            question
        )


        # ----------------------------------------------------
        # RAG
        # ----------------------------------------------------

        documents, metadatas = (

            retrieve_medical_context(

                query=question,

                subject=subject

            )

        )


        context = build_context(

            documents,

            metadatas

        )


        # ----------------------------------------------------
        # Prompt
        # ----------------------------------------------------

        prompt = f"""
You are MedicalAI, an educational medical
assistant for medical students.

Your task is to answer the user's medical
question using the retrieved medical
knowledge below.

MEDICAL SUBJECT:
{subject}

USER QUESTION:
{question}

RETRIEVED MEDICAL KNOWLEDGE:
{context}

Instructions:

1. Give a clear educational explanation.
2. Use the retrieved information as the primary
   factual reference.
3. Do not invent unsupported medical facts.
4. Explain difficult terminology simply.
5. Use headings and bullet points where useful.
6. Include important clinical relevance when
   appropriate.
7. Do not diagnose the user.
8. Do not provide personalized medical advice.
9. This system is for medical education.
10. If the retrieved context is insufficient,
    clearly state the limitation.

Give a useful answer suitable for a medical
student.
"""


        # ----------------------------------------------------
        # Gemini
        # ----------------------------------------------------

        result = generate_text(

            prompt

        )


        # ----------------------------------------------------
        # Sources
        # ----------------------------------------------------

        sources = build_sources(

            metadatas

        )


        print("")
        print(
            "TEXT ANSWER GENERATED SUCCESSFULLY"
        )

        print(
            "=============================================="
        )


        return {

            "success":
                True,

            "subject":
                subject,

            "question":
                question,

            "answer":
                result["text"],

            "model":
                result["model"],

            "sources":
                sources,

        }


    except Exception as error:

        print("")
        print("==============================================")
        print("TEXT QUESTION ERROR")
        print("==============================================")


        print(
            repr(error)
        )


        return {

            "success":
                False,

            "error":
                str(error),

        }


# ============================================================
# GEMINI VISION
# ============================================================

def analyze_image_with_vision(

    image_bytes,

    mime_type

):

    prompt = """
You are the vision component of an educational
medical learning application.

Analyze the uploaded image carefully.

This system is NOT a clinical diagnostic system.

The image may be:

- anatomical diagram
- medical illustration
- photograph
- X-ray
- CT
- MRI
- ultrasound
- pathology image
- histology image
- organ image
- medical educational image

Identify the MAIN visible anatomical structure
or medical concept.

Return ONLY valid JSON.

Use EXACTLY this structure:

{
  "image_type": "",
  "modality": "",
  "body_region": "",
  "identified_structure": "",
  "medical_subject": "",
  "confidence": 0,
  "is_clear": true,
  "visualization_type": "",
  "visualization_target": "",
  "description": ""
}

The medical_subject MUST be exactly one of:

Anatomy
Physiology
Biochemistry
Pathology
Pharmacology
Microbiology
Forensic Medicine & Toxicology
Community Medicine
General Medicine
General Surgery
Pediatrics
Obstetrics & Gynecology
Orthopedics
Dermatology
Psychiatry
Radiology
Anesthesiology
Ophthalmology
Otorhinolaryngology (ENT)
Dentistry / Oral Medicine
Cardiology
Neurology
Pulmonology
Gastroenterology
Nephrology
Endocrinology
Rheumatology
Oncology
Urology
Emergency Medicine

IMPORTANT 3D ROUTING:

If the main visible structure is the human brain,
use:

"visualization_type": "brain_3d"
"visualization_target": "Brain"

If the main visible structure is the human heart,
use:

"visualization_type": "heart_3d"
"visualization_target": "Heart"

For a structure without a current 3D model,
use:

"visualization_type": "none"
"visualization_target": ""

confidence must be between 0 and 100.

description should briefly explain what is
visibly present.

Do NOT diagnose disease.

Do NOT infer a disease merely from an image.

If the image is unclear, lower the confidence
and explain the uncertainty.
"""


    # --------------------------------------------------------
    # Create image part
    # --------------------------------------------------------

    image_part = types.Part.from_bytes(

        data=image_bytes,

        mime_type=mime_type

    )


    last_error = None


    # --------------------------------------------------------
    # Try vision models
    # --------------------------------------------------------

    for model_name in VISION_MODELS:

        for attempt in range(MAX_RETRIES):

            try:

                print("")
                print("----------------------------------------------")

                print(
                    "Gemini vision model:",
                    model_name
                )

                print(
                    "Attempt:",
                    attempt + 1
                )


                response = (

                    client.models.generate_content(

                        model=model_name,

                        contents=[

                            image_part,

                            prompt

                        ],

                        config=(

                            types.GenerateContentConfig(

                                temperature=0.1,

                                response_mime_type=
                                    "application/json",

                            )

                        ),

                    )

                )


                if response is None:

                    raise RuntimeError(

                        "Gemini Vision returned no response."

                    )


                response_text = response.text


                if not response_text:

                    raise RuntimeError(

                        "Gemini Vision returned empty response."

                    )


                result = clean_json(

                    response_text

                )


                if not result:

                    raise RuntimeError(

                        "Gemini Vision JSON parsing failed."

                    )


                print("")
                print(
                    "Vision analysis successful."
                )


                return (

                    result,

                    model_name

                )


            except Exception as error:

                last_error = error


                print("")
                print(
                    "Gemini Vision error:"
                )

                print(
                    repr(error)
                )


                if is_retryable_error(error):

                    if attempt < MAX_RETRIES - 1:

                        delay = RETRY_DELAYS[

                            min(

                                attempt,

                                len(RETRY_DELAYS) - 1

                            )

                        ]


                        print(

                            f"Retrying in {delay} seconds..."

                        )


                        time.sleep(delay)

                else:

                    break


    raise RuntimeError(

        "All Gemini vision models failed. "
        f"Last error: {last_error}"

    )


# ============================================================
# NORMALIZE VISION RESULT
# ============================================================

def normalize_vision_result(

    result

):

    if not isinstance(

        result,

        dict

    ):

        result = {}


    # --------------------------------------------------------
    # Basic information
    # --------------------------------------------------------

    image_type = str(

        result.get(

            "image_type",

            "Medical image"

        )

    )


    modality = str(

        result.get(

            "modality",

            "Unknown"

        )

    )


    body_region = str(

        result.get(

            "body_region",

            "Unknown"

        )

    )


    structure = str(

        result.get(

            "identified_structure",

            "Unknown structure"

        )

    )


    subject = str(

        result.get(

            "medical_subject",

            "Anatomy"

        )

    )


    if subject not in MEDICAL_SUBJECTS:

        subject = "Anatomy"


    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    try:

        confidence = float(

            result.get(

                "confidence",

                0

            )

        )

    except Exception:

        confidence = 0


    confidence = max(

        0,

        min(

            100,

            confidence

        )

    )


    # --------------------------------------------------------
    # Visualization
    # --------------------------------------------------------

    visualization_type = str(

        result.get(

            "visualization_type",

            "none"

        )

    ).lower()


    visualization_target = str(

        result.get(

            "visualization_target",

            ""

        )

    )


    description = str(

        result.get(

            "description",

            ""

        )

    )


    # --------------------------------------------------------
    # Combine text for routing
    # --------------------------------------------------------

    combined_text = (

        structure

        + " "

        + body_region

        + " "

        + description

    ).lower()


    # --------------------------------------------------------
    # HEART ROUTING
    # --------------------------------------------------------

    if (

        "heart" in combined_text

        or "cardiac" in combined_text

    ):

        visualization_type = "heart_3d"

        visualization_target = "Heart"


    # --------------------------------------------------------
    # BRAIN ROUTING
    # --------------------------------------------------------

    elif (

        "brain" in combined_text

        or "cerebr" in combined_text

        or "cerebell" in combined_text

        or "brainstem" in combined_text

    ):

        visualization_type = "brain_3d"

        visualization_target = "Brain"


    # --------------------------------------------------------
    # If unsupported visualization
    # --------------------------------------------------------

    elif visualization_type not in [

        "brain_3d",

        "heart_3d",

    ]:

        visualization_type = "none"

        visualization_target = ""


    # --------------------------------------------------------
    # Return normalized result
    # --------------------------------------------------------

    return {

        "image_type":
            image_type,

        "modality":
            modality,

        "body_region":
            body_region,

        "identified_structure":
            structure,

        "medical_subject":
            subject,

        "confidence":
            confidence,

        "is_clear":
            bool(

                result.get(

                    "is_clear",

                    True

                )

            ),

        "visualization_type":
            visualization_type,

        "visualization_target":
            visualization_target,

        "description":
            description,

    }


# ============================================================
# IMAGE EDUCATIONAL EXPLANATION
# ============================================================

def generate_image_explanation(

    vision,

    context

):

    prompt = f"""
You are MedicalAI, an educational medical
assistant for medical students.

The uploaded image has been analyzed by a
vision model.

This is NOT a clinical diagnostic system.

IMAGE INFORMATION:

Image type:
{vision["image_type"]}

Modality:
{vision["modality"]}

Body region:
{vision["body_region"]}

Identified structure:
{vision["identified_structure"]}

Medical subject:
{vision["medical_subject"]}

Vision confidence:
{vision["confidence"]}%

Image description:
{vision["description"]}

Available 3D visualization:
{vision["visualization_type"]}


RETRIEVED MEDICAL KNOWLEDGE:

{context}


Create a clear educational explanation for
a medical student.

Use these sections:

## Image Analysis

Explain what the AI identified from the image.

## What Is This?

Explain the structure or medical concept.

## Location

Explain where it is located.

## Structure

Explain important anatomical features.

## Function

Explain the major functions.

## Blood Supply

Explain major blood supply when relevant.

## Clinical Importance

Explain important educational clinical
connections.

Do NOT diagnose the uploaded image.

Do NOT claim that an image finding is a
confirmed disease.

## 3D Visualization

Explain what the available 3D model can help
the student understand.

## Important Uncertainty

Clearly explain that AI image interpretation
may be incorrect and is intended for education.

## Summary

Give a short revision summary.

Use the retrieved medical knowledge as the
primary factual reference.

Do not invent unsupported medical facts.
"""


    return generate_text(

        prompt

    )


# ============================================================
# IMAGE ANALYSIS ENDPOINT
# ============================================================

@app.post("/api/analyze-image")
async def analyze_image(

    image: UploadFile = File(...)

):

    try:

        print("")
        print("==============================================")
        print("IMAGE ANALYSIS REQUEST")
        print("==============================================")


        # ----------------------------------------------------
        # File information
        # ----------------------------------------------------

        print(
            "Filename:",
            image.filename
        )


        print(
            "Content type:",
            image.content_type
        )


        # ----------------------------------------------------
        # Read image
        # ----------------------------------------------------

        image_bytes = await image.read()


        if not image_bytes:

            return {

                "success":
                    False,

                "error":
                    "Uploaded image is empty."

            }


        print(
            "Image size:",
            len(image_bytes),
            "bytes"
        )


        # ----------------------------------------------------
        # Validate image
        # ----------------------------------------------------

        try:

            pil_image = Image.open(

                io.BytesIO(

                    image_bytes

                )

            )


            pil_image.verify()


        except Exception as error:

            print(
                "Invalid image:"
            )

            print(
                repr(error)
            )


            return {

                "success":
                    False,

                "error":
                    "The uploaded file is not a valid image."

            }


        # ----------------------------------------------------
        # MIME type
        # ----------------------------------------------------

        mime_type = (

            image.content_type

            or "image/jpeg"

        )


        # ----------------------------------------------------
        # Gemini Vision
        # ----------------------------------------------------

        print("")
        print(
            "Starting Gemini Vision..."
        )


        raw_vision, vision_model = (

            analyze_image_with_vision(

                image_bytes,

                mime_type

            )

        )


        # ----------------------------------------------------
        # Normalize
        # ----------------------------------------------------

        vision = normalize_vision_result(

            raw_vision

        )


        print("")
        print(
            "Identified structure:",
            vision["identified_structure"]
        )


        print(
            "Medical subject:",
            vision["medical_subject"]
        )


        print(
            "Confidence:",
            vision["confidence"]
        )


        print(
            "Visualization:",
            vision["visualization_type"]
        )


        # ----------------------------------------------------
        # RAG query
        # ----------------------------------------------------

        rag_query = f"""
Explain the following medical structure
for medical students.

Structure:
{vision["identified_structure"]}

Subject:
{vision["medical_subject"]}

Image description:
{vision["description"]}

Explain:

- location
- structure
- function
- blood supply
- important clinical relevance
"""


        # ----------------------------------------------------
        # Retrieve medical context
        # ----------------------------------------------------

        documents, metadatas = (

            retrieve_medical_context(

                query=rag_query,

                subject=vision["medical_subject"]

            )

        )


        context = build_context(

            documents,

            metadatas

        )


        # ----------------------------------------------------
        # Generate educational explanation
        # ----------------------------------------------------

        print("")
        print(
            "Generating educational explanation..."
        )


        explanation = (

            generate_image_explanation(

                vision,

                context

            )

        )


        # ----------------------------------------------------
        # Sources
        # ----------------------------------------------------

        sources = build_sources(

            metadatas

        )


        # ----------------------------------------------------
        # Final response
        # ----------------------------------------------------

        response = {

            "success":
                True,

            "automatic_analysis":
                True,

            "image_filename":
                image.filename,

            "image_type":
                vision["image_type"],

            "modality":
                vision["modality"],

            "body_region":
                vision["body_region"],

            "medical_subject":
                vision["medical_subject"],

            "identified_structure":
                vision["identified_structure"],

            "vision_confidence":
                vision["confidence"],

            "prediction": {

                "label":
                    vision["identified_structure"],

                "confidence":
                    vision["confidence"],

                "class_id":
                    None,

                "probabilities":
                    {},

            },

            "description":
                vision["description"],

            "rag": {

                "subject":
                    vision["medical_subject"],

                "query":
                    rag_query,

                "documents_retrieved":
                    len(documents),

            },

            "gemini": {

                "model":
                    explanation["model"],

                "vision_model":
                    vision_model,

                "answer":
                    explanation["text"],

            },

            "visualization": {

                "available":
                    vision["visualization_type"]
                    != "none",

                "type":
                    vision["visualization_type"],

                "target":
                    vision["visualization_target"],

            },

            "sources":
                sources,

        }


        print("")
        print("==============================================")
        print("IMAGE ANALYSIS COMPLETE")
        print("==============================================")


        return response


    except Exception as error:

        print("")
        print("==============================================")
        print("IMAGE ANALYSIS ERROR")
        print("==============================================")


        print(
            repr(error)
        )


        return {

            "success":
                False,

            "error":
                str(error),

        }

# ============================================================
# TOPIC-SPECIFIC AI QUIZ ENDPOINT
# ============================================================

@app.post("/api/quiz")
async def generate_topic_quiz(data: dict):
    try:
        # 1. Read and validate the request
        subject = str(data.get("subject", "")).strip()
        topic = str(data.get("topic", "")).strip()
        count = data.get("count", 20)

        if subject not in MEDICAL_SUBJECTS:
            return {
                "success": False,
                "error": "Please select a valid medical subject."
            }

        if not topic or len(topic) > 200:
            return {
                "success": False,
                "error": "Please select a valid topic."
            }

        if count != 20 or isinstance(count, bool):
            return {
                "success": False,
                "error": "Quiz must contain exactly 20 questions."
            }

        # 2. Retrieve knowledge relevant to this exact topic
        rag_query = (
            f"Medical subject: {subject}\n"
            f"Study topic: {topic}\n"
            f"Find factual educational information specifically "
            f"about {topic}, including its structure, functions, "
            f"important relationships, and relevant terminology."
        )

        documents, metadatas = retrieve_medical_context(
            query=rag_query,
            subject=subject
        )

        context = build_context(documents, metadatas)

        # 3. Ask Gemini to generate topic-specific questions
        prompt = f"""
You are an educational medical quiz generator.

SUBJECT: {subject}
EXACT STUDY TOPIC: {topic}

MEDICAL REFERENCE MATERIAL RETRIEVED FROM THE
KNOWLEDGE BASE:
{context}

TASK:
Generate exactly 20 distinct multiple-choice questions
about the EXACT STUDY TOPIC above.

STRICT RULES:
1. Every question must directly test the selected topic.
2. Do not generate general questions about unrelated
   parts of the subject.
3. Cover different concepts within the selected topic.
4. Give exactly four answer options per question.
5. There must be exactly one correct answer per question.
6. The answer must be an integer from 0 to 3, matching
   the option's zero-based index.
7. Include a clear explanation for every correct answer.
8. Use a mixture of easy, medium, and challenging
   medical-student-level questions.
9. Avoid duplicate questions and duplicate options.
10. Use the retrieved reference material as the primary
    source of facts. Do not claim that information came
    from the references unless it is actually supported.
11. If the reference material is insufficient, use only
    well-established medical knowledge and do not invent
    facts or unsupported details.
12. This quiz is for education, not patient diagnosis.

Return ONLY a valid JSON object with this structure:
{{
  "questions": [
    {{
      "question": "Question text",
      "options": [
        "Option A",
        "Option B",
        "Option C",
        "Option D"
      ],
      "answer": 0,
      "explanation": "Explanation of the correct answer"
    }}
  ]
}}

Return exactly 20 question objects.
Do not include Markdown fences or extra text.
"""

        result = generate_text(prompt)
        parsed = clean_json(result["text"])
        questions = parsed.get("questions", [])

        # 4. Validate the generated quiz
        if not isinstance(questions, list) or len(questions) != 20:
            raise ValueError(
                "Gemini did not return exactly 20 questions. "
                "Please try generating the quiz again."
            )

        validated_questions = []

        for index, item in enumerate(questions):
            if not isinstance(item, dict):
                raise ValueError(
                    f"Question {index + 1} has an invalid format."
                )

            question_text = item.get("question")
            options = item.get("options")
            answer = item.get("answer")
            explanation = item.get("explanation")

            if not isinstance(question_text, str) or not question_text.strip():
                raise ValueError(
                    f"Question {index + 1} has no question text."
                )

            if (
                not isinstance(options, list)
                or len(options) != 4
                or not all(
                    isinstance(option, str) and option.strip()
                    for option in options
                )
            ):
                raise ValueError(
                    f"Question {index + 1} must have four valid options."
                )

            if (
                not isinstance(answer, int)
                or isinstance(answer, bool)
                or answer not in range(4)
            ):
                raise ValueError(
                    f"Question {index + 1} has an invalid answer index."
                )

            if not isinstance(explanation, str) or not explanation.strip():
                raise ValueError(
                    f"Question {index + 1} has no explanation."
                )

            validated_questions.append({
                "question": question_text.strip(),
                "options": [option.strip() for option in options],
                "answer": answer,
                "explanation": explanation.strip()
            })

        # 5. Return the quiz to React
        return {
            "success": True,
            "subject": subject,
            "topic": topic,
            "count": len(validated_questions),
            "model": result["model"],
            "questions": validated_questions,
            "sources": build_sources(metadatas),
            "reference_documents_retrieved": len(documents)
        }

    except Exception as error:
        print("TOPIC QUIZ ERROR:", repr(error))

        from fastapi import HTTPException

        raise HTTPException(
            status_code=500,
            detail=str(error)
        ) from error
# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {

        "success":
            True,

        "message":
            "MedicalAI backend is running.",

        "endpoints": [

            "/api/ask",

            "/api/analyze-image",

            "/api/status",

        ],

    }


# ============================================================
# STATUS ENDPOINT
# ============================================================

@app.get("/api/status")
def status():

    return {

        "success":
            True,

        "chroma_records":
            collection.count(),

        "text_models":
            TEXT_MODELS,

        "vision_models":
            VISION_MODELS,

    }


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    import uvicorn


    print("")
    print("==============================================")
    print("STARTING MEDICAL AI BACKEND")
    print("==============================================")


    print(
        "Server:",
        "http://127.0.0.1:8000"
    )


    print(
        "Swagger:",
        "http://127.0.0.1:8000/docs"
    )


    print(
        "Ask API:",
        "http://127.0.0.1:8000/api/ask"
    )


    print(
        "Image API:",
        "http://127.0.0.1:8000/api/analyze-image"
    )


    print("")


    uvicorn.run(

        app,

        host="127.0.0.1",

        port=8000,

        reload=False,

    )