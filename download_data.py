import os
import urllib.request

# Create the folder structure
os.makedirs("data/RAGTruth/dataset", exist_ok=True)

print("Downloading response.jsonl (this might take a minute)...")
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset/response.jsonl",
    "data/RAGTruth/dataset/response.jsonl"
)

print("Downloading source_info.jsonl (this might take another minute)...")
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset/source_info.jsonl",
    "data/RAGTruth/dataset/source_info.jsonl"
)

print("Success! Both files are downloaded and placed in the correct folder.")