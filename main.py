import json
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from dotenv import load_dotenv

load_dotenv()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://3.140.46.146:3000"],  # Add your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class UserQuery(BaseModel):
    message: str


@app.post("/process_user_query")
async def process_user_query(user_query: UserQuery):
    # Just echo back the query to show it's connected to frontend
    print(f"Processed query: {user_query.message}")
    citations= [
        { "id": 1, "text": "Advanced computational models", "url": "https://example.com/computational-models", "pdfUrl": "https://example.com/computational-models.pdf" },
        { "id": 2, "text": "Mathematical algorithms", "url": "https://example.com/algorithms", "pdfUrl": "https://example.com/algorithms.pdf" }
    ]
    steps = [                                                                                                                                                                      
        "Analyze the computational query",                                                                                                                                        
        "Identify key variables and constraints",
        "Search relevant mathematical models and algorithms",
        "Synthesize computational approach",
        "Formulate a comprehensive solution"
    ]
    
    nash_response = {
        "user_query": user_query.message,
        "research_plan": {
            "steps": steps
        },
        "citations": citations,
        "response": "This is a response"
    }

    return json.dumps(nash_response)
