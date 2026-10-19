import json
from pathlib import Path
from collections import Counter


# ==========================================
# 1. Dataset path
# ==========================================

PROJECT_DIR = Path(__file__).parent.parent
DATA_FILE = PROJECT_DIR / "data" / "Anatomy.json"


# ==========================================
# 2. Load dataset
# ==========================================

with open(DATA_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)


print("Total records:", len(data))


# ==========================================
# 3. Check duplicate content
# ==========================================

content_counter = Counter(
    record["content"].strip().lower()
    for record in data
)


duplicate_groups = {
    content: count
    for content, count in content_counter.items()
    if count > 1
}


# ==========================================
# 4. Results
# ==========================================

print("\n==========================================")
print("DUPLICATE CHECK")
print("==========================================")

print("Total records:", len(data))
print("Unique content:", len(content_counter))
print("Duplicate groups:", len(duplicate_groups))

duplicate_records = sum(
    count - 1
    for count in duplicate_groups.values()
)

print("Duplicate records:", duplicate_records)


# ==========================================
# 5. Show examples
# ==========================================

print("\nDuplicate examples:")
print("------------------------------------------")

for index, (content, count) in enumerate(
    duplicate_groups.items()
):

    if index >= 10:
        break

    print(f"\nDuplicate {index + 1}")
    print("Count:", count)
    print("Content:", content)


print("\n==========================================")