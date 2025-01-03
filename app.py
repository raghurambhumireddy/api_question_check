from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client import models
from sentence_transformers import SentenceTransformer
import re

app = FastAPI()

# Load the secret key from secret_key.txt
with open("secret_key.txt", "r") as f:
    SECRET_KEY = f.read().strip()

# Initialize the SentenceTransformer model for embedding
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Connect to Qdrant
# qdrant_client = QdrantClient(
#     url="https://c8e964b9-aa62-4a53-87dd-7252467c2b02.europe-west3-0.gcp.cloud.qdrant.io:6333",
#     api_key="bVrqBpHHscZySyi4qKnTWDpm4JdmMXMnFbhLCD2dxvJKZbrmUQ11pA"
# )
qdrant_client = QdrantClient(
    url="https://2519ed63-9687-4a30-a5eb-91a3b7e4c194.us-east4-0.gcp.cloud.qdrant.io:6333", 
    api_key="ft66KkUd5M2gbqID2Ly9FT9UwxbjMn0LMiVIm0S4vJLFe0XtU-bTiQ",
)

# collect_name = "question_answer_4"
collect_name = "testing_7_master_sample_data"

class SearchRequest(BaseModel):
    query: str  # Sentence to be searched
    secret_key: str  # Secret key for authentication

def clean_value(value):
    value = re.sub(' +', ' ', value)
    return value.strip()

def clean_record(record):
    if 'paragraph' in record:
        record['paragraph'] = clean_value(record['paragraph'])
    if 'phrase' in record:
        record['phrase'] = clean_value(record['phrase'])
    return record

def replace_special_characters(record):
    for key, value in record.items():
        if isinstance(value, str):
            value = value.replace('\n', ' ').replace('\t', ' ')
            record[key] = value
    return record

@app.post("/search")
def search_query(request: SearchRequest):
    # Verify the secret key
    if request.secret_key != SECRET_KEY:
        raise HTTPException(status_code=403, detail="Authentication failed")

    query_sentence = request.query

    try:
        # Convert the query sentence into a vector embedding
        query_embedding = model.encode(query_sentence).tolist()

        # Perform a similarity search on Qdrant using the embedding
        search_result = qdrant_client.search(
            collection_name=collect_name,
            query_vector=query_embedding,
            limit=3,  # Return top 3 records
            search_params=models.SearchParams(
                hnsw_ef=128,  # Optional, can help tune search speed vs accuracy
                exact=False   # Cosine similarity is approximate
            )
        )

        # Process the results
        if search_result:
            data = []
            for result in search_result:
                record = {'id': result.id, 'score': result.score}
                record.update(result.payload)
                record = replace_special_characters(record)
                record = clean_record(record)
                data.append(record)
            
            return {"data": data}  # Return the top 3 results
        else:
            raise HTTPException(status_code=404, detail="No matching records found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
