import os
import shutil

SOURCE = "photo_database"
TEST = "test_images"

os.makedirs(TEST, exist_ok=True)

for person in os.listdir(SOURCE):

    person_path = os.path.join(SOURCE, person)

    if not os.path.isdir(person_path):
        continue

    images = os.listdir(person_path)

    if len(images) < 2:
        continue

    test_img = images[0]

    src = os.path.join(person_path, test_img)
    dst = os.path.join(TEST, person + ".jpg")

    shutil.move(src, dst)

print("✅ Test set created automatically")

