import os
import numpy as np
import json
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import random

TRAIN_DIR = "train_db"

mtcnn = MTCNN(image_size=160)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

embeddings = []
profiles = []

print("🚀 Building database (ONE IMAGE PER PERSON)...")

for person in os.listdir(TRAIN_DIR):

    person_path = os.path.join(TRAIN_DIR, person)

    if not os.path.isdir(person_path):
        continue

    images = os.listdir(person_path)

    if len(images) == 0:
        continue

    # 🔥 اختار صورة واحدة عشوائي لكل شخص
    img_name = random.choice(images)
    img_path = os.path.join(person_path, img_name)

    try:
        img = Image.open(img_path).convert("RGB")
    except:
        continue

    face = mtcnn(img)
    if face is None:
        continue

    emb = resnet(face.unsqueeze(0)).detach().numpy()

    embeddings.append(emb[0])
    profiles.append({
        "name": person,
        "id": person
    })

embeddings = np.array(embeddings)

os.makedirs("local_db", exist_ok=True)

np.save("local_db/embeddings.npy", embeddings)

with open("local_db/profiles.json", "w") as f:
    json.dump(profiles, f)

print("✅ Database built successfully")
print("Total persons (embeddings):", len(embeddings))