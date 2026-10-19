import json
from pathlib import Path
import hashlib

import chromadb
from sentence_transformers import SentenceTransformer


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_DIR = BASE_DIR / "models" / "chroma_db"


# ============================================================
# REQUIRED FIELDS
# ============================================================

REQUIRED_FIELDS = [
    "id",
    "subject",
    "topic",
    "subtopic",
    "content",
    "source",
    "source_url",
    "license",
    "verification_status"
]


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("=" * 60)
print("MEDICAL AI - RAG INGESTION")
print("=" * 60)

print("\nLoading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded successfully.")


# ============================================================
# CONNECT TO CHROMADB
# ============================================================

print("\nConnecting to ChromaDB...")

client = chromadb.PersistentClient(path=str(DB_DIR))

# Fresh collection for the complete dataset
try:
    client.delete_collection("medical_knowledge")
    print("Old collection removed.")
except Exception:
    pass

collection = client.create_collection(
    name="medical_knowledge"
)

print("New ChromaDB collection created.")


# ============================================================
# FIND DATASET FILES
# ============================================================

json_files = sorted(DATA_DIR.glob("*.json"))

if not json_files:
    print("\nERROR: No JSON files found in data folder.")
    exit()


print(f"\nJSON files found: {len(json_files)}")


# ============================================================
# TRACKING
# ============================================================

total_records = 0
total_added = 0
total_duplicates = 0
total_invalid = 0

subject_counts = {}

seen_content = set()


# ============================================================
# PROCESS EACH FILE
# ============================================================

for file in json_files:

    print("\n" + "-" * 60)
    print(f"Processing: {file.name}")
    print("-" * 60)

    try:

        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

    except Exception as e:

        print(f"ERROR reading {file.name}: {e}")
        continue


    # --------------------------------------------------------
    # Check dataset format
    # --------------------------------------------------------

    if not isinstance(data, list):

        print("WARNING: Dataset is not a JSON list.")
        print("Skipping this file.")

        continue


    print(f"Records found: {len(data)}")


    # --------------------------------------------------------
    # Process records
    # --------------------------------------------------------

    for index, record in enumerate(data):

        total_records += 1


        # ----------------------------------------------------
        # Validate record
        # ----------------------------------------------------

        if not isinstance(record, dict):

            print(
                f"Invalid record at index {index}: "
                "not an object"
            )

            total_invalid += 1
            continue


        missing_fields = [
            field
            for field in REQUIRED_FIELDS
            if field not in record
        ]


        if missing_fields:

            print(
                f"Invalid record {index}: "
                f"missing {missing_fields}"
            )

            total_invalid += 1
            continue


        # ----------------------------------------------------
        # Basic content validation
        # ----------------------------------------------------

        content = str(record["content"]).strip()

        subject = str(record["subject"]).strip()
        topic = str(record["topic"]).strip()
        subtopic = str(record["subtopic"]).strip()


        if not content:

            print(
                f"Invalid record {index}: "
                "empty content"
            )

            total_invalid += 1
            continue


        if not subject:

            print(
                f"Invalid record {index}: "
                "empty subject"
            )

            total_invalid += 1
            continue


        # ----------------------------------------------------
        # Duplicate detection
        # ----------------------------------------------------

        content_hash = hashlib.sha256(
            content.lower().encode("utf-8")
        ).hexdigest()


        if content_hash in seen_content:

            total_duplicates += 1

            continue


        seen_content.add(content_hash)


        # ----------------------------------------------------
        # Generate unique ID
        # ----------------------------------------------------

        record_id = str(record["id"]).strip()

        if not record_id:

            record_id = f"{file.stem}_{index}"


        # Add filename/index to guarantee uniqueness
        chroma_id = f"{file.stem}_{record_id}_{index}"


        # ----------------------------------------------------
        # Generate embedding
        # ----------------------------------------------------

        embedding = model.encode(
            content,
            show_progress_bar=False
        ).tolist()


        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        metadata = {

            "subject": subject,

            "topic": topic,

            "subtopic": subtopic,

            "source": str(
                record["source"]
            ),

            "source_url": str(
                record["source_url"]
            ),

            "license": str(
                record["license"]
            ),

            "verification_status": str(
                record["verification_status"]
            ),

            "file": file.name

        }


        # ----------------------------------------------------
        # Add to ChromaDB
        # ----------------------------------------------------

        collection.add(

            ids=[chroma_id],

            embeddings=[embedding],

            documents=[content],

            metadatas=[metadata]

        )


        total_added += 1


        # ----------------------------------------------------
        # Subject count
        # ----------------------------------------------------

        if subject not in subject_counts:

            subject_counts[subject] = 0

        subject_counts[subject] += 1


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print("RAG INGESTION COMPLETED")
print("=" * 60)

print(f"\nDataset records     : {total_records}")
print(f"Records added       : {total_added}")
print(f"Duplicate records   : {total_duplicates}")
print(f"Invalid records     : {total_invalid}")
print(f"ChromaDB records    : {collection.count()}")


# ============================================================
# SUBJECT SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("SUBJECT SUMMARY")
print("=" * 60)

for subject, count in sorted(subject_counts.items()):

    print(f"{subject:<40} {count}")


# ============================================================
# FINAL STATUS
# ============================================================

print("\n" + "=" * 60)

if total_invalid == 0:

    print("STATUS : SUCCESS")

else:

    print("STATUS : COMPLETED WITH INVALID RECORDS")


print("=" * 60)

print("\nChromaDB location:")
print(DB_DIR)

print("\nYour complete medical dataset is now ready for RAG.")