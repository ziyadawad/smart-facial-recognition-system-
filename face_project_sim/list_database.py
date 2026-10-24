import os
import json
import numpy as np

# ====== مسارات الـ database ======
DB_DIR = "local_db"
EMBEDDINGS_FILE = os.path.join(DB_DIR, "embeddings.npy")
PROFILES_FILE = os.path.join(DB_DIR, "profiles.json")

# ====== التحقق من وجود الـ database ======
if not os.path.exists(EMBEDDINGS_FILE) or not os.path.exists(PROFILES_FILE):
    print("❌ Database is empty!")
    exit()

embeddings = np.load(EMBEDDINGS_FILE, allow_pickle=True)
with open(PROFILES_FILE, "r") as f:
    profiles = json.load(f)

if len(embeddings) != len(profiles):
    print("❌ Embeddings and profiles count mismatch!")
    exit()

# ====== عرض كل الأشخاص ======
print(f"🔹 Total registered people: {len(profiles)}\n")
for i, profile in enumerate(profiles, start=1):
    print(f"{i}. Name: {profile['name']}")
    print(f"   ID: {profile['id_number']}")
    print(f"   Age: {profile['age']}")
    print(f"   Legal Case: {profile['legal_case']}")
    print("-" * 40)
