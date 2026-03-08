import json
import numpy as np
import cohere
import faiss
import os

api_key = os.getenv("COHERE_API_KEY")
co = cohere.ClientV2(api_key)

with open("university_data.json", "r", encoding="utf-8") as f:
    knowledge_base = json.load(f)

texts = [doc["content"] for doc in knowledge_base]

print("Generating embeddings...")

response = co.embed(
    model="embed-english-v3.0",
    input_type="search_document",
    texts=texts
)

embeddings = np.array(response.embeddings.float)

np.save("embeddings.npy", embeddings)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

faiss.write_index(index, "faiss_index.index")

print("Embeddings built successfully.")