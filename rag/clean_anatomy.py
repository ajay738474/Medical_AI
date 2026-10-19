import json
from pathlib import Path


# ==========================================
# 1. Paths
# ==========================================

PROJECT_DIR = Path(__file__).parent.parent

INPUT_FILE = PROJECT_DIR / "data" / "Anatomy.json"

OUTPUT_FILE = PROJECT_DIR / "data" / "Anatomy_clean.json"


# ==========================================
# 2. Load original dataset
# ==========================================

print("Loading Anatomy dataset...")

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

print("Original records:", len(data))


# ==========================================
# 3. Remove exact duplicate content
# ==========================================

unique_records = []
seen_content = set()

duplicate_count = 0


for record in data:

    content = record["content"].strip()

    # Case-insensitive duplicate checking
    content_key = content.lower()

    if content_key in seen_content:

        duplicate_count += 1
        continue

    seen_content.add(content_key)

    unique_records.append(record)


# ==========================================
# 4. Save cleaned dataset
# ==========================================

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

    json.dump(
        unique_records,
        f,
        ensure_ascii=False,
        indent=2
    )


# ==========================================
# 5. Final report
# ==========================================

print("\n==========================================")
print("       ANATOMY DATASET CLEANED")
print("==========================================")

print("Original records :", len(data))
print("Unique records   :", len(unique_records))
print("Duplicates removed:", duplicate_count)

print("\nCreated:")
print(OUTPUT_FILE)

print("==========================================")