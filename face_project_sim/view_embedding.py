import numpy as np

# تحميل الـ database
db_embeddings = np.load("local_db/embeddings.npy")
db_ids = np.load("local_db/ids.npy")

# عرض كل الـ vectors بشكل مختصر
for i, (person, embedding) in enumerate(zip(db_ids, db_embeddings)):
    print(f"\nPerson #{i+1}: {person}")
    print("Embedding vector (first 10 numbers, rounded to 1 decimal):")
    print([round(x, 1) for x in embedding[:10]])  # أول 10 أرقام
    print("Full vector shape:", embedding.shape)


