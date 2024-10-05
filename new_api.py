from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
import os
import asyncio
from dotenv import load_dotenv
# Import our custom modules
from utils import setup_llm_and_embedding, load_index
from workflow import RAGSystem

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Global variables to be initialized on startup
index = None
llm = None
rag_system = None

# Define the request models
class QueryInput(BaseModel):
    query: str

class FeedbackInput(BaseModel):
    query: str
    feedback: str
    initial_response: str

def set_api_keys():
    required_keys = ['LLAMA_CLOUD_API_KEY', 'GROQ_API_KEY', 'SERPAPI_KEY']
    missing_keys = [key for key in required_keys if not os.getenv(key)]
    
    if missing_keys:
        raise ValueError(f"Missing required API keys: {', '.join(missing_keys)}. Please check your .env file.")
    
    for key in required_keys:
        os.environ[key] = os.getenv(key)

# On startup, initialize the global components
@app.on_event("startup")
async def startup_event():
    global index, llm, rag_system
    
    try:
        # Set API keys from environment variables
        set_api_keys()
        
        # Initialize LLM and embedding model
        llm, embed_model = setup_llm_and_embedding()
        
        # Load or create the index
        index = load_index(embed_model)
        
        # Initialize the global RAGSystem instance
        rag_system = RAGSystem(index=index, llm=llm)
    except Exception as e:
        print(f"Error during startup: {str(e)}")
        # You might want to exit the application here if it can't start properly
        import sys
        sys.exit(1)

# Define the /query endpoint
@app.post("/query")
async def handle_query(query_input: QueryInput, request: Request):
    global rag_system
    if rag_system is None:
        raise HTTPException(status_code=500, detail="RAGSystem not initialized.")
    
    user_query = query_input.query
    context = {}  # You can add any additional context here if needed
    
    try:
        response = await rag_system.process_query(user_query, context)
        # Ensure the response always includes a "response" key
        if "response" not in response:
            response["response"] = response.get("message", "No response available")
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# Define the /feedback endpoint
@app.post("/feedback")
async def handle_feedback(feedback_input: FeedbackInput, request: Request):
    global rag_system
    if rag_system is None:
        raise HTTPException(status_code=500, detail="RAGSystem not initialized.")
    
    user_query = feedback_input.query
    user_feedback = feedback_input.feedback
    initial_response = feedback_input.initial_response
    
    # Convert feedback to decision
    user_decision = "satisfied" if user_feedback.lower() == "yes" else "unsatisfied"
    
    context = {
        "original_query": user_query,
        "initial_response": initial_response
    }
    
    try:
        response = await rag_system.process_user_feedback(user_decision, context)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "rag_system_initialized": rag_system is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)