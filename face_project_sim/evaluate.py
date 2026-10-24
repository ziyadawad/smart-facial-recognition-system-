import os
import numpy as np
import json
import time
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns


# =====================================
# 📌 LOAD TRAIN DATABASE
# =====================================
embeddings = np.load("local_db/embeddings.npy", allow_pickle=True)

with open("local_db/profiles.json") as f:
    profiles = json.load(f)

# =====================================
# 🧠 SAFE ID EXTRACTION
# =====================================
def get_id(profile):
    if "id" in profile:
        return str(profile["id"])
    elif "id_number" in profile:
        return str(profile["id_number"])
    elif "name" in profile:
        return str(profile["name"])
    else:
        return "unknown"

# =====================================
# 🤖 MODELS
# =====================================
mtcnn = MTCNN(image_size=160)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# =====================================
# 📊 METRICS
# =====================================
TP = FP = FN = TN = 0
total_time = 0
N = 0

# =====================================
# 📈 ROC STORAGE
# =====================================
y_true = []
y_scores = []

THRESHOLD = 0.9

DATASET_PATH = "photo_database"

print("🚀 Starting FULL 5743 PERSON Evaluation...\n")

# =====================================
# 🧪 LOOP OVER ALL 5743 PEOPLE
# =====================================
for person in os.listdir(DATASET_PATH):

    person_folder = os.path.join(DATASET_PATH, person)

    if not os.path.isdir(person_folder):
        continue

    for img_name in os.listdir(person_folder):

        img_path = os.path.join(person_folder, img_name)

        try:
            img = Image.open(img_path).convert('RGB')
        except:
            continue

        start = time.time()

        face = mtcnn(img)
        if face is None:
            print(f"❌ No face in {img_name}")
            continue

        emb = resnet(face.unsqueeze(0)).detach().numpy()

        sims = cosine_similarity(emb, embeddings)[0]

        best_idx = np.argmax(sims)
        best_score = sims[best_idx]

        predicted = get_id(profiles[best_idx])
        true = person

        # =====================================
        # 📈 ROC DATA
        # =====================================
        if predicted == true:
            y_true.append(1)
        else:
            y_true.append(0)

        y_scores.append(best_score)

        end = time.time()
        latency = end - start

        total_time += latency
        N += 1

        # =====================================
        # 📌 DECISION LOGIC (SAME AS YOUR CODE)
        # =====================================
        if best_score >= THRESHOLD:
            if predicted == true:
                TP += 1
                result = "✅ TP"
            else:
                FP += 1
                result = "❌ FP"
        else:
            if predicted == true:
                FN += 1
                result = "⚠️ FN"
            else:
                TN += 1
                result = "✔ TN"

        print(f"{img_name} → Predicted: {predicted} | True: {true} | Score: {best_score:.2f} | {result}")

# =====================================
# 📊 FINAL METRICS
# =====================================
accuracy = (TP + TN) / (TP + FP + FN + TN + 1e-9)
precision = TP / (TP + FP + 1e-9)
recall = TP / (TP + FN + 1e-9)

FAR = FP / (FP + TN + 1e-9)
FRR = FN / (TP + FN + 1e-9)

latency = total_time / (N + 1e-9)

print("\n==========================")
print("📊 FINAL RESULTS (FULL 5743 DATASET)")
print("==========================")
print(f"Samples   : {N}")
print(f"TP        : {TP}")
print(f"FP        : {FP}")
print(f"FN        : {FN}")
print(f"TN        : {TN}")
print("--------------------------")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"FAR       : {FAR:.4f}")
print(f"FRR       : {FRR:.4f}")
print(f"Latency   : {latency:.4f} sec")
print("==========================")



# =====================================
# 📈 ROC CURVE
# =====================================
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7,6))

plt.plot(
    fpr,
    tpr,
    linewidth=2,
    label=f'ROC Curve (AUC = {roc_auc:.4f})'
)

plt.plot([0, 1], [0, 1], linestyle='--')

plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("ROC Curve - Face Recognition System")

plt.legend(loc="lower right")

plt.show()
