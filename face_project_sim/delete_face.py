import os
import json
import numpy as np

# ====== مسارات الـ database ======
DB_DIR = "local_db"
EMBEDDINGS_FILE = os.path.join(DB_DIR, "embeddings.npy")
PROFILES_FILE = os.path.join(DB_DIR, "profiles.json")

# ====== تحميل الـ database ======
if not os.path.exists(EMBEDDINGS_FILE) or not os.path.exists(PROFILES_FILE):
    print("❌ Database is empty!")
    exit()

embeddings = np.load(EMBEDDINGS_FILE, allow_pickle=True)

with open(PROFILES_FILE, "r") as f:
    profiles = json.load(f)

if len(profiles) == 0:
    print("❌ No profiles found!")
    exit()

if len(embeddings) < len(profiles):
    print("❌ ERROR: Embeddings أقل من profiles! Database corrupted.")
    exit()

# ====== إدخال الاسم أو الـ ID مباشرة ======
value = input("Enter name or ID to delete: ").strip()

# ====== تحديد هل ID ولا name ======
if value.isdigit():
    # ID
    indices_profiles = [
        i for i, profile in enumerate(profiles)
        if profile.get("id_number", "") == value
    ]
else:
    # Name
    value_lower = value.lower()
    indices_profiles = [
        i for i, profile in enumerate(profiles)
        if profile.get("name", "").lower() == value_lower
    ]

# ====== check ======
if not indices_profiles:
    print(f"❌ '{value}' not found!")
    exit()

# ====== حساب عدد embeddings لكل شخص ======
num_embeddings_per_person = len(embeddings) // len(profiles)

# ====== حذف البيانات ======
for index in reversed(indices_profiles):
    # حذف profile
    profiles.pop(index)

    # حذف embeddings المرتبطة
    start_idx = index * num_embeddings_per_person
    end_idx = start_idx + num_embeddings_per_person
    embeddings = np.delete(embeddings, np.s_[start_idx:end_idx], axis=0)

# ====== حفظ التغييرات ======
np.save(EMBEDDINGS_FILE, embeddings)

with open(PROFILES_FILE, "w") as f:
    json.dump(profiles, f, indent=4)

print(f"🗑️ Deleted '{value}' successfully!")






