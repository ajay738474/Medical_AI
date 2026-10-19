# ============================================================
# MEDICAL AI
# AUTOMATIC CNN -> RAG -> GEMINI PIPELINE
# ============================================================

import os
import time
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google import genai

from image_predictor import predict_image


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BACKEND_ENV = PROJECT_ROOT / "backend" / ".env"

CHROMA_PATH = PROJECT_ROOT / "models" / "chroma_db"

COLLECTION_NAME = "medical_knowledge"


# ============================================================
# 2. CURRENT CNN DATASET
# ============================================================

# Change only these two values when testing another dataset.

DATASET_NAME = "organsmnist"

IMAGE_PATH = (
    PROJECT_ROOT
    / "cnn"
    / "organS_test_image.png"
)


# ============================================================
# 3. RAG SETTINGS
# ============================================================

TOP_K = 5


# ============================================================
# 4. DATASET -> MEDICAL SUBJECT
# ============================================================

DATASET_SUBJECTS = {

    "pathmnist": "Pathology",

    "chestmnist": "Radiology",

    "dermamnist": "Dermatology",

    "octmnist": "Ophthalmology",

    "pneumoniamnist": "Pulmonology",

    "retinamnist": "Ophthalmology",

    "breastmnist": "Oncology",

    "bloodmnist": "Pathology",

    "tissuemnist": "Anatomy",

    "organamnist": "Anatomy",

    "organcmnist": "Anatomy",

    "organsmnist": "Anatomy",
}


# ============================================================
# 5. NORMALIZE DATASET
# ============================================================

DATASET_NAME = DATASET_NAME.lower().strip()


if DATASET_NAME not in DATASET_SUBJECTS:

    raise ValueError(
        f"""
Invalid dataset:

{DATASET_NAME}

Available datasets:

{list(DATASET_SUBJECTS.keys())}
"""
    )


RAG_SUBJECT = DATASET_SUBJECTS[DATASET_NAME]


# ============================================================
# 6. START
# ============================================================

print()
print("=" * 70)
print("          AUTOMATIC CNN -> RAG -> GEMINI")
print("=" * 70)


# ============================================================
# 7. SHOW PROJECT PATHS
# ============================================================

print()
print("Project folder:")
print(PROJECT_ROOT)

print()
print("Environment file:")
print(BACKEND_ENV)


# ============================================================
# 8. CHECK .ENV
# ============================================================

if not BACKEND_ENV.exists():

    raise FileNotFoundError(
        f"""
backend/.env was not found.

Expected:

{BACKEND_ENV}
"""
    )


# ============================================================
# 9. LOAD ENVIRONMENT
# ============================================================

load_dotenv(
    dotenv_path=BACKEND_ENV
)


GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY"
)


if not GEMINI_API_KEY:

    raise ValueError(
        f"""
GEMINI_API_KEY was not found.

Please add:

GEMINI_API_KEY=YOUR_API_KEY

inside:

{BACKEND_ENV}
"""
    )


print()
print("Gemini API key loaded successfully.")


# ============================================================
# 10. CREATE GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=GEMINI_API_KEY
)


print()
print("Gemini client created successfully.")


# ============================================================
# 11. CHECK IMAGE
# ============================================================

print()
print("=" * 70)
print("CHECKING INPUT IMAGE")
print("=" * 70)

print()
print("Image:")
print(IMAGE_PATH)


if not IMAGE_PATH.exists():

    raise FileNotFoundError(
        f"""
Input image was not found:

{IMAGE_PATH}
"""
    )


print()
print("Image found successfully.")


# ============================================================
# 12. RUN CNN
# ============================================================

print()
print("=" * 70)
print("RUNNING CNN")
print("=" * 70)


cnn_result = predict_image(
    str(IMAGE_PATH),
    DATASET_NAME
)


# ============================================================
# 13. EXTRACT CNN RESULT
# ============================================================

predicted_class = cnn_result["class_id"]

predicted_label = cnn_result["label"]

confidence = cnn_result["confidence"]

probabilities = cnn_result["probabilities"]


print()
print("CNN prediction completed.")

print()
print("Dataset:")
print(DATASET_NAME)

print()
print("Predicted class:")
print(predicted_class)

print()
print("Predicted label:")
print(predicted_label)

print()
print("Confidence:")
print(f"{confidence:.2f}%")


# ============================================================
# 14. CREATE RAG QUERY
# ============================================================

print()
print("=" * 70)
print("CREATING RAG QUERY")
print("=" * 70)


rag_query = (
    f"Explain the medical anatomy and educational importance "
    f"of {predicted_label}. "
    f"Describe its location, structure, function, blood supply, "
    f"and important clinical information."
)


print()
print("RAG Subject:")
print(RAG_SUBJECT)

print()
print("RAG Query:")
print(rag_query)


# ============================================================
# 15. LOAD EMBEDDING MODEL
# ============================================================

print()
print("=" * 70)
print("LOADING EMBEDDING MODEL")
print("=" * 70)


embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


print()
print("Embedding model loaded successfully.")


# ============================================================
# 16. CONNECT TO CHROMADB
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
        f"""
ChromaDB was not found:

{CHROMA_PATH}
"""
    )


chroma_client = chromadb.PersistentClient(
    path=str(CHROMA_PATH)
)


# ============================================================
# 17. GET COLLECTION
# ============================================================

try:

    collection = chroma_client.get_collection(
        name=COLLECTION_NAME
    )

except Exception as error:

    raise RuntimeError(
        f"""
Could not open ChromaDB collection:

{COLLECTION_NAME}

Error:

{error}
"""
    )


print()
print("ChromaDB connected successfully.")

print()
print("Total records:")
print(collection.count())


# ============================================================
# 18. CREATE EMBEDDING
# ============================================================

print()
print("=" * 70)
print("CREATING QUERY EMBEDDING")
print("=" * 70)


query_embedding = embedding_model.encode(
    rag_query
).tolist()


print()
print("Embedding created.")

print()
print("Dimensions:")
print(len(query_embedding))


# ============================================================
# 19. RETRIEVE FROM CHROMADB
# ============================================================

print()
print("=" * 70)
print("RETRIEVING MEDICAL KNOWLEDGE")
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
        f"""
ChromaDB retrieval failed.

Error:

{error}
"""
    )


# ============================================================
# 20. EXTRACT RESULTS
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
        """
No medical documents were retrieved
from ChromaDB.
"""
    )


print()
print("Retrieved documents:")
print(len(documents))


# ============================================================
# 21. DISPLAY RETRIEVED DOCUMENTS
# ============================================================

print()
print("=" * 70)
print("RETRIEVED MEDICAL INFORMATION")
print("=" * 70)


for i, document in enumerate(documents):

    print()
    print(f"RESULT {i + 1}")
    print("-" * 70)

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
# 22. BUILD RAG CONTEXT
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
# 23. CREATE GEMINI PROMPT
# ============================================================

prompt = f"""
You are a medical education AI assistant.

Your job is to explain medical information clearly
for medical students.

The system analyzed an image using a CNN.

CNN DATASET:
{DATASET_NAME}

CNN PREDICTED CLASS:
{predicted_class}

CNN PREDICTED LABEL:
{predicted_label}

CNN CONFIDENCE:
{confidence:.2f}%

IMPORTANT SAFETY RULES:

The CNN prediction is NOT a confirmed diagnosis.

The prediction confidence is {confidence:.2f}%.

Therefore, clearly describe this as an AI prediction
and communicate the uncertainty.

Do not claim that the image definitely contains
the predicted structure.

Use the retrieved medical information below
as the factual foundation for the answer.

Do not invent facts that are not supported by
the retrieved context.

If the retrieved context does not contain enough
information for a requested detail, say that
the retrieved sources do not provide enough
information for that detail.

Explain the result using these sections:

1. CNN Prediction

2. Prediction Confidence

3. Important Uncertainty

4. What Is This Structure?

5. Location

6. Structure

7. Function

8. Blood Supply

9. Clinical Importance

10. Summary

11. Sources

Keep the explanation educational and easy
for a medical student to understand.

------------------------------------------------------------
RETRIEVED MEDICAL KNOWLEDGE
------------------------------------------------------------

{context}

------------------------------------------------------------
END OF RETRIEVED MEDICAL KNOWLEDGE
------------------------------------------------------------
"""


# ============================================================
# 24. GET AVAILABLE GEMINI MODELS
# ============================================================

def get_available_gemini_models():

    print()
    print("=" * 70)
    print("CHECKING AVAILABLE GEMINI MODELS")
    print("=" * 70)


    available_models = []


    try:

        models = client.models.list()


        for model in models:

            model_name = getattr(
                model,
                "name",
                None
            )


            if not model_name:

                continue


            # Remove models/ prefix
            if model_name.startswith(
                "models/"
            ):

                model_name = model_name[
                    len("models/"):
                ]


            print()
            print("Available model:")
            print(model_name)


            # We only want Flash models
            if "flash" in model_name.lower():

                available_models.append(
                    model_name
                )


    except Exception as error:

        print()
        print("Could not list Gemini models.")

        print()
        print("Error:")
        print(error)

        return []


    return available_models


# ============================================================
# 25. SELECT MODEL ORDER
# ============================================================

def build_model_order(
    available_models
):

    # Models here are only preferences.
    # The code uses them ONLY if the API
    # actually reports them as available.

    preferred_models = [

        "gemini-2.5-flash",

        "gemini-2.0-flash",

        "gemini-3.5-flash",

        "gemini-3.7-flash",

        "gemini-3.8-flash",
    ]


    models_to_try = []


    # First use preferred models
    for preferred in preferred_models:

        if preferred in available_models:

            if preferred not in models_to_try:

                models_to_try.append(
                    preferred
                )


    # Then use any remaining Flash models
    for model in available_models:

        if model not in models_to_try:

            models_to_try.append(
                model
            )


    return models_to_try


# ============================================================
# 26. GENERATE GEMINI ANSWER
# ============================================================

def generate_with_gemini(
    prompt_text
):

    available_models = (
        get_available_gemini_models()
    )


    if not available_models:

        raise RuntimeError(
            """
No Gemini Flash models were returned
for this API key.

The CNN and RAG stages were successful,
but no usable Gemini Flash model was found.
"""
        )


    print()
    print("=" * 70)
    print("AVAILABLE GEMINI FLASH MODELS")
    print("=" * 70)


    for model in available_models:

        print(
            "-",
            model
        )


    models_to_try = build_model_order(
        available_models
    )


    print()
    print("=" * 70)
    print("GEMINI MODEL TRY ORDER")
    print("=" * 70)


    for model in models_to_try:

        print(
            "-",
            model
        )


    # ========================================================
    # Try models
    # ========================================================

    for model_name in models_to_try:

        print()
        print("=" * 70)

        print(
            "Trying Gemini model:"
        )

        print(
            model_name
        )

        print("=" * 70)


        # Three attempts
        for attempt in range(
            1,
            4
        ):

            print()
            print(
                f"Attempt {attempt}/3"
            )


            try:

                # Create chat
                chat = client.chats.create(
                    model=model_name
                )


                # Send prompt
                response = chat.send_message(
                    prompt_text
                )


                # Validate response
                if response is None:

                    raise RuntimeError(
                        "Gemini returned no response."
                    )


                if not getattr(
                    response,
                    "text",
                    None
                ):

                    raise RuntimeError(
                        "Gemini returned empty text."
                    )


                print()
                print(
                    "Gemini response received successfully."
                )


                print()
                print(
                    "Model used:"
                )

                print(
                    model_name
                )


                return response.text


            except Exception as error:

                error_text = str(
                    error
                ).lower()


                print()
                print(
                    "Gemini request failed."
                )


                print()
                print(
                    "Error:"
                )

                print(
                    error
                )


                # =================================================
                # TEMPORARY 503 / 429
                # =================================================

                temporary_error = (

                    "503" in error_text

                    or

                    "unavailable" in error_text

                    or

                    "high demand" in error_text

                    or

                    "429" in error_text

                    or

                    "resource exhausted" in error_text

                    or

                    "overloaded" in error_text
                )


                if temporary_error:

                    if attempt < 3:

                        wait_seconds = (
                            attempt * 5
                        )


                        print()
                        print(
                            "Temporary Gemini error."
                        )

                        print(
                            f"Waiting "
                            f"{wait_seconds} seconds..."
                        )


                        time.sleep(
                            wait_seconds
                        )


                        continue


                    print()
                    print(
                        "This model remains "
                        "temporarily unavailable."
                    )

                    break


                # =================================================
                # 404 / MODEL NOT FOUND
                # =================================================

                if (
                    "404" in error_text
                    or
                    "not found" in error_text
                    or
                    "not supported" in error_text
                ):

                    print()
                    print(
                        "This model is not supported "
                        "for this API."
                    )

                    break


                # =================================================
                # OTHER ERROR
                # =================================================

                print()
                print(
                    "Trying the next available model."
                )

                break


        print()
        print(
            f"Model failed: {model_name}"
        )


    # ========================================================
    # ALL MODELS FAILED
    # ========================================================

    raise RuntimeError(
        """
All available Gemini Flash models failed.

CNN:
SUCCESS

RAG:
SUCCESS

Gemini:
FAILED

The problem is only in the Gemini
generation stage.
"""
    )


# ============================================================
# 27. SEND TO GEMINI
# ============================================================

print()
print("=" * 70)
print("SENDING RAG CONTEXT TO GEMINI")
print("=" * 70)


answer = generate_with_gemini(
    prompt
)


# ============================================================
# 28. DISPLAY FINAL ANSWER
# ============================================================

print()
print("=" * 70)
print("             FINAL GEMINI EXPLANATION")
print("=" * 70)

print()

print(answer)


# ============================================================
# 29. DISPLAY CNN PROBABILITIES
# ============================================================

print()
print("=" * 70)
print("CNN CLASS PROBABILITIES")
print("=" * 70)


for label, probability in probabilities.items():

    print(
        f"{label:40s} "
        f"{probability:.2f}%"
    )


# ============================================================
# 30. DISPLAY SOURCES
# ============================================================

print()
print("=" * 70)
print("RAG SOURCES")
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
# 31. PIPELINE SUMMARY
# ============================================================

print()
print("=" * 70)
print("       AUTOMATIC CNN -> RAG -> GEMINI COMPLETED")
print("=" * 70)

print()

print("IMAGE")
print("  ↓")
print("CNN")
print("  ↓")
print("PREDICTION")
print("  ↓")
print("AUTOMATIC RAG QUERY")
print("  ↓")
print("EMBEDDING")
print("  ↓")
print("CHROMADB")
print("  ↓")
print("RETRIEVED MEDICAL KNOWLEDGE")
print("  ↓")
print("GEMINI")
print("  ↓")
print("EDUCATIONAL EXPLANATION")

print()