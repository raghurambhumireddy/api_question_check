from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client import models
from sentence_transformers import SentenceTransformer
import re

app = FastAPI()

# Initialize the SentenceTransformer model for embedding
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Connect to Qdrant
qdrant_client = QdrantClient(
    url="https://c8e964b9-aa62-4a53-87dd-7252467c2b02.europe-west3-0.gcp.cloud.qdrant.io:6333",
    api_key="bVrqBpHHscZySyi4qKnTWDpm4JdmMXMnFbhLCD2dxvJKZbrmUQ11pA"
)

collect_name = "question_answer_4"

class SearchRequest(BaseModel):
    query: str  # Sentence to be searched

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
    query_sentence = request.query

    try:
        # Convert the query sentence into a vector embedding
        query_embedding = model.encode(query_sentence).tolist()
        # print(query_embedding)

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

        print(search_result)
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
