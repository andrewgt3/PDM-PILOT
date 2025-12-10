import os
import string
import random
import pandas as pd
from datetime import datetime
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

# Config
DATA_DIR = "data/MAINTNET"
# The user mentioned "e.g. Aircraft_AnnotationDataFile.csv", we'll target that one specifically or generic
FILE_PATTERN = "Aircraft_AnnotationDataFile.csv"

def get_mongo_collection():
    user = os.getenv("MONGO_USER", "admin")
    pwd = os.getenv("MONGO_PASSWORD", "password")
    uri = f"mongodb://{user}:{pwd}@localhost:27017/"
    client = MongoClient(uri)
    db = client["pdm_logs"]
    return db["maintenance_logs"]

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    # 1. Lowercase
    text = text.lower()
    # 2. Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def load_logs():
    file_path = os.path.join(DATA_DIR, FILE_PATTERN)
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return

    print(f"Reading {file_path}...")
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return

    print("Processing records...")
    documents = []
    
    # Identify text column (heuristic: 'Observation' or first string col)
    text_col = 'Observation' if 'Observation' in df.columns else df.columns[1]
    
    for idx, row in df.iterrows():
        raw_text = row.get(text_col, "")
        
        doc = {
            "asset_id": f"AC-{random.randint(100, 999)}", # Generic Asset ID for MaintNet
            "timestamp": datetime.utcnow(),                # Current Timestamp
            "problem_text": preprocess_text(raw_text),      # Preprocessed Text
            # "original_text": raw_text # Ops, prompt didn't ask for original_text, but good for debug. I'll include it? No, stick to prompt strictness if possible. Prompt said "include fields for asset_id, timestamp, and a free-text field called problem_text". It didn't forbid others, but I'll stick to essentials.
        }
        documents.append(doc)

    collection = get_mongo_collection()
    print(f"Inserting {len(documents)} documents into MongoDB...")
    
    try:
        result = collection.insert_many(documents)
        print(f"✅ Successfully inserted {len(result.inserted_ids)} documents.")
        
        # Verify
        count = collection.count_documents({})
        print(f"📊 Total documents in 'maintenance_logs': {count}")
        
    except Exception as e:
        print(f"❌ Error inserting logs: {e}")

if __name__ == "__main__":
    load_logs()
