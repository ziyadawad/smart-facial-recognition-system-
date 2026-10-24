
import cv2
import os
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
import base64
import torch
from torchvision import transforms

# ====== مجلدات الصور ======

TEST_IMAGES_DIR = "test_images"
SEND_DIR = "to_pi"
os.makedirs(SEND_DIR, exist_ok=True)

# ====== قراءة أول صورة ======

images = [f for f in os.listdir(TEST_IMAGES_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
if not images:
    print("❌ No test images found in 'test_images/' folder!")
    exit()

image_path = os.path.join(TEST_IMAGES_DIR, images[0])
img = cv2.imread(image_path)
send_path = os.path.join(SEND_DIR, "face_to_pi.jpg")

# ====== Haar Cascade (الأساسي) ======

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

if len(faces) > 0:
    x, y, w, h = faces[0]
    face_crop = img[y:y+h, x:x+w]
    print("✅ Face detected successfully by Haar Cascade")
else:
    # ====== MTCNN backup ======
    mtcnn = MTCNN(keep_all=True)
    boxes, probs = mtcnn.detect(img)
    if boxes is not None and len(boxes) > 0:
        x1, y1, x2, y2 = boxes[0]
        face_crop = img[int(y1):int(y2), int(x1):int(x2)]
        print("✅ Face detected successfully by MTCNN")
    else:
        print("❌ No face detected by Haar Cascade or MTCNN")
        exit()

cv2.imwrite(send_path, face_crop)

# ====== ArcFace Embedding ======
resnet = InceptionResnetV1(pretrained='vggface2').eval()
img_pil = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))

# تحويل PIL Image لـ Tensor
preprocess = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor()
])
img_tensor = preprocess(img_pil).unsqueeze(0)  # shape: [1, 3, 160, 160]

embedding = resnet(img_tensor)
#print("✅ Embedding generated, shape:", embedding.shape)

# ====== AES-256 CBC Encryption ======
AES_KEY = b"12345678901234567890123456789012"  # 32 bytes

def encrypt_image(image_bytes, key):
    iv = os.urandom(16)
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
    encryptor = cipher.encryptor()
    padder = padding.PKCS7(128).padder()
    padded_data = padder.update(image_bytes) + padder.finalize()
    encrypted = encryptor.update(padded_data) + encryptor.finalize()
    return base64.b64encode(iv + encrypted).decode()

def prepare_encrypted_packet(image_path):
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    return encrypt_image(img_bytes, AES_KEY)

encrypted_face = prepare_encrypted_packet(send_path)

#print("🔐 ESP Simulator: Encrypted face image ready (Base64 preview):")
#print(encrypted_face[:100] + "...")
