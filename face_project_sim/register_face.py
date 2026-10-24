import os
import json
import numpy as np
from PIL import Image
import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

# =========================
# 📌 Database Paths
# =========================

DB_DIR = "local_db"
os.makedirs(DB_DIR, exist_ok=True)

EMBEDDINGS_FILE = os.path.join(DB_DIR, "embeddings.npy")
PROFILES_FILE = os.path.join(DB_DIR, "profiles.json")

# =========================
# 📌 Load Existing Database
# =========================

if os.path.exists(EMBEDDINGS_FILE):
    embeddings = np.load(EMBEDDINGS_FILE, allow_pickle=True)
else:
    embeddings = np.empty((0, 512))

if os.path.exists(PROFILES_FILE):
    with open(PROFILES_FILE, "r") as f:
        profiles = json.load(f)
else:
    profiles = []

# =========================
# 📌 Initialize Models
# =========================

mtcnn = MTCNN(image_size=160, margin=0)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# =========================
# 📌 User Input
# =========================

name = input("Name: ").strip()
id_number = input("ID Number: ").strip()
age = input("Age: ").strip()
legal_case = input("Legal case (clean / suspect / criminal / theif): ").strip().lower()
image_path = input("Enter image path (example: test_images/ziad.jpg): ").strip()

# =========================
# 📌 Validate Inputs
# =========================

if not name or not id_number:
    print("❌ Missing required fields!")
    exit()

# 🔥 FIX: proper validation (NO unknown unless truly invalid)
if legal_case not in ["clean", "suspect", "criminal", "theif"]:
    print("⚠️ Invalid legal_case → defaulting to 'theif'")
    legal_case = "theif"

if not os.path.exists(image_path):
    print("❌ Image not found!")
    exit()

# =========================
# 📌 Load Image
# =========================

img_cv = cv2.imread(image_path)
img_pil = Image.open(image_path)

# =========================
# 📌 Face Detection
# =========================

faces_to_embed = []

# 1️⃣ Haar Cascade (Primary)
gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)

if len(faces) > 0:
    x, y, w, h = faces[0]
    face_crop = img_cv[y:y+h, x:x+w]
    face_crop_pil = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
    faces_to_embed.append(face_crop_pil)
    print("✅ Face detected by Haar Cascade")

else:
    # 2️⃣ MTCNN (Backup)
    face_mtcnn = mtcnn(img_pil)

    if face_mtcnn is not None:
        faces_to_embed.append(face_mtcnn)
        print("✅ Face detected by MTCNN")
    else:
        print("❌ No face detected!")
        exit()

# =========================
# 📌 Generate Embeddings
# =========================

for face in faces_to_embed:

    if isinstance(face, Image.Image):
        face_tensor = mtcnn(face)
        if face_tensor is None:
            continue
    else:
        face_tensor = face

    emb = resnet(face_tensor.unsqueeze(0)).detach().numpy()

    embeddings = np.vstack([embeddings, emb])

    profiles.append({
        "name": name,
        "id_number": id_number,
        "age": age,
        "legal_case": legal_case
    })

# =========================
# 📌 Save Database
# =========================

np.save(EMBEDDINGS_FILE, embeddings)

with open(PROFILES_FILE, "w") as f:
    json.dump(profiles, f, indent=4)

print(f"✅ Registered '{name}' successfully!")
print(f"📊 Legal Case: {legal_case}")