import os
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import threading
import socket
import socketio
import requests
import time
import json
import sys
from datetime import datetime
import time as t

# ==========================================
# 🔵 Auto Detect IP
# ==========================================
def get_my_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip

# ==========================================
# 🟦 Node Setup
# ==========================================
node_id = "Gate 2"
target_nodes = ["192.168.8.4"]
port = 5000
WS_SERVERS = [
    "http://192.168.8.5:8080"
]
ws = None
my_ip = get_my_ip()
print(f"🔵 My detected IP: {my_ip}")

# ==========================================
# 🟩 Send & Receive Functions
# ==========================================
def send_data(data, target_ip, port=5000):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((target_ip, port))
        message = json.dumps(data) + "\n"
        s.sendall(message.encode())
        s.close()
    except Exception as e:
        print(f"❌ Error sending data to {target_ip}: {e}")

def recv_all(conn):
    buffer = b""
    while True:
        part = conn.recv(4096)
        if not part:
            break
        buffer += part
    return buffer

lock = threading.Lock()
best_score_overall = -1
match_found = False
THRESHOLD = 0.65

# ==========================================
# 🟧 Handle Incoming Connections
# ==========================================
def handle_client(conn):
    global best_score_overall, best_profile_overall, match_found
    data = recv_all(conn)
    try:
        request = json.loads(data.decode())

        if request.get("type") == "search":
            embedding_received = np.array(request["embedding"])
            embedding_received = normalize(embedding_received, axis=1)
            sims = cosine_similarity(embedding_received, embeddings_normalized)
            best_idx = np.argmax(sims)
            best_score = max(0.0, float(sims[0][best_idx]))

            if best_score > THRESHOLD:
                response = {
                    "type": "match_response",
                    "profile": profiles[best_idx],
                    "score": best_score,
                    "node_id": node_id
                }
                send_data(response, request["sender_ip"], port)
            else:
                print(f"❌ No match found on this node, Best similarity: {best_score:.2f}")

        elif request.get("type") == "match_response":
            print("📥 MATCH RESPONSE RECEIVED")
            with lock:
                score = max(0.0, request['score'])
                if score > best_score_overall:
                    best_score_overall = score
                    best_profile_overall = request['profile']
                match_found = best_score_overall >= THRESHOLD

        elif request.get("type") == "block":
            try:
                block_dict = {
                    "index": request["index"],
                    "timestamp": request["timestamp"],
                    "data": request["data"],
                    "previous_hash": request["previous_hash"],
                    "hash": request["hash"]
                }
                blockchain.add_block_remote(block_dict)
                print(f"🟢 Block from {request.get('node_id','another_node')} added successfully")
            except Exception as e:
                print(f"❌ Failed to add remote block: {e}")

    except Exception as e:
        print(f"❌ Error handling client: {e}")
    finally:
        conn.close()

# ==========================================
# 🟥 Server Thread
# ==========================================
def start_server(port=5000):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(('', port))
    server.listen(5)
    print(f"🟢 SERVER STARTED on {my_ip}:{port}")
    while True:
        conn, addr = server.accept()
        threading.Thread(target=handle_client, args=(conn,)).start()

threading.Thread(target=start_server, daemon=True).start()

# ==========================================
# 🟦 Blockchain System
# ==========================================
sys.path.append("blockchain")
os.makedirs("blockchain", exist_ok=True)
from blockchain_flask import Blockchain
blockchain = Blockchain(file_path="blockchain/blockchain.json")

# ==========================================
# 🟪 Database
# ==========================================
DB_DIR = "local_db"
EMBEDDINGS_FILE = os.path.join(DB_DIR, "embeddings.npy")
PROFILES_FILE = os.path.join(DB_DIR, "profiles.json")

embeddings = np.load(EMBEDDINGS_FILE, allow_pickle=True) if os.path.exists(EMBEDDINGS_FILE) else np.zeros((1,512))
with open(PROFILES_FILE, "r") as f:
    profiles = json.load(f)

assert len(embeddings) == len(profiles), "❌ Embeddings count and profiles count mismatch!"
embeddings_normalized = normalize(embeddings, axis=1)

# ==========================================
# 🟨 Face Models
# ==========================================
mtcnn = MTCNN(image_size=160, margin=0)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# ==========================================
sio = socketio.Client()
connected_server = None

def init_ws():
    global connected_server
    for server in WS_SERVERS:
        try:
            sio.connect(server)
            connected_server = server
            print(f"✅ Socket.IO connected to {server}")
            return
        except Exception as e:
            print(f"❌ Failed to connect {server}: {e}")
    print("❌ All servers failed")

def send_match_event(data):
    global connected_server
    try:
        if sio.connected:
            sio.emit("new_match", data)
            print("📤 Sent to website")
        else:
            print("⚠️ Not connected, reconnecting...")
            init_ws()
    except Exception as e:
        print("❌ Emit error:", e)
        time.sleep(1)
        init_ws()

init_ws()

# ==========================================
# 🟫 Load Image from to_pi/
# ==========================================
SEND_DIR = "to_pi"
images = [f for f in os.listdir(SEND_DIR) if f.lower().endswith(('.jpg','.png','.jpeg'))]
if not images:
    print("❌ No images in to_pi/")
    while True:
        pass

image_path = os.path.join(SEND_DIR, images[0])
img = Image.open(image_path)
face = mtcnn(img)
if face is None:
    print("❌ No face detected!")
    while True:
        pass

embedding = resnet(face.unsqueeze(0)).detach().numpy()
embedding_normalized = normalize(embedding, axis=1)

# ==========================================
# 🟦 Local Matching
# ==========================================
sims = cosine_similarity(embedding_normalized, embeddings_normalized)
best_idx = np.argmax(sims)
best_score_local = max(0.0, float(sims[0][best_idx]))
best_profile_local = profiles[best_idx] if best_idx < len(profiles) else None

if best_score_local > best_score_overall:
    best_score_overall = best_score_local
    best_profile_overall = best_profile_local

match_found = best_score_overall >= THRESHOLD

# ==========================================
# 🟦 Remote Search
# ==========================================
if not match_found:
    request = {
        "type": "search",
        "embedding": embedding.tolist(),
        "sender_ip": my_ip
    }
    for target in target_nodes:
        send_data(request, target, 5001)

    start_time = t.time()
    while t.time() - start_time < 3:
        t.sleep(0.1)

# ==========================================
# 📌 FINAL RESULT
# ==========================================
print("🔍 FINAL RESULT")
print(f"Similarity Score: {best_score_overall:.2f}")

if match_found and best_profile_overall:
    print("🟢 MATCH FOUND")
    print(f"Name: {best_profile_overall['name']}")
    print(f"ID: {best_profile_overall['id_number']}")
    print(f"Age: {best_profile_overall['age']}")
    print(f"Legal Case: {best_profile_overall['legal_case']}")
else:
    print("❌ NO MATCH (below threshold)")

# ==========================================
# ⛓️ BLOCK CREATION (UPDATED ONLY)
# ==========================================
if best_profile_overall is None:
    best_profile_overall = {
        "name": "Unknown",
        "id_number": None,
        "age": None,
        "legal_case": None
    }

block_data = {
    "type": "match",
    "person_name": best_profile_overall["name"] if best_profile_overall else "Unknown",
    "person_id": best_profile_overall["id_number"] if best_profile_overall else None,
    "age": best_profile_overall["age"] if best_profile_overall else None,
    "legal_case": best_profile_overall["legal_case"] if best_profile_overall else None,
    "score": float(best_score_overall),
    "node_id": node_id,
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "match_status": "matched" if match_found else "no_match"
}

block = blockchain.add_block(block_data)
print("🟡 Block saved to blockchain.json ✅")
send_match_event(block_data)

for target in target_nodes:
    send_data({
        "type": "block",
        "index": block.index,
        "timestamp": block.timestamp,
        "data": block.data,
        "previous_hash": block.previous_hash,
        "hash": block.hash
    }, target, port)

try:
    while True:
        t.sleep(1)
except KeyboardInterrupt:
    print("\n🛑 Script stopped by user")
    sys.exit(0)