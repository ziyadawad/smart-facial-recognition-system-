import os
import random
import shutil

SOURCE_DIR = "photo_database"
TRAIN_DIR = "train_db"
TEST_DIR = "test_db"

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

all_people = os.listdir(SOURCE_DIR)
random.shuffle(all_people)

split_idx = int(0.8 * len(all_people))

train_people = all_people[:split_idx]
test_people = all_people[split_idx:]

def copy_people(people, target_dir):
    for person in people:
        src = os.path.join(SOURCE_DIR, person)
        dst = os.path.join(target_dir, person)
        shutil.copytree(src, dst)

copy_people(train_people, TRAIN_DIR)
copy_people(test_people, TEST_DIR)

print(f"Train: {len(train_people)} people")
print(f"Test: {len(test_people)} people")