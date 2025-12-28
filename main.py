from fastapi import FastAPI
from pydantic import BaseModel
from zhipuai import ZhipuAI
from pinecone import Pinecone
import os

app = FastAPI()

# 1. Setup GLM Client (Your friend's API key)
client = ZhipuAI(api_key="YOUR_GLM_API_KEY")

# 2. Setup Pinecone
pc = Pinecone(api_key="YOUR_PINECONE_API_KEY")
index = pc.Index("nctb-index")

class Query(BaseModel):
    question: str

@app.post("/ask")
async def chat_tutor(query: Query):
    # Logic: 1. Search Pinecone for textbook context
    # (Simplified for now - we assume you've uploaded textbook embeddings)
    context = "নিউটনের দ্বিতীয় সূত্র: বস্তুর ভরবেগের পরিবর্তনের হার তার উপর প্রযুক্ত বলের সমানুপাতিক।" 

    # Logic: 2. Tell GLM to be a Logic-based interactive tutor
    response = client.chat.completions.create(
        model="glm-4",  # or "glm-4-flash" for speed
        messages=[
            {
                "role": "system", 
                "content": """You are a helpful Class 9-10 Tutor for Bangladesh students. 
                Use the textbook context provided to explain the LOGIC. 
                Don't just give the answer; explain 'why' so the student learns. 
                Answer in friendly Bangla."""
            },
            {"role": "user", "content": f"Context: {context}\nQuestion: {query.question}"}
        ],
        top_p=0.9,
        temperature=0.7
    )
    
    return {"answer": response.choices[0].message.content}