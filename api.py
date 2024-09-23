from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_setup import load_index, setup_llm_and_embedding, create_query_engine

app = FastAPI()

# Load LLM and embedding model
llm, embed_model = setup_llm_and_embedding()

# Load the index
index = load_index(embed_model)

# Create a query engine
query_engine = create_query_engine(index,llm)


class Query(BaseModel):
    query: str

@app.get("/")
async def read_root():
    return {"message": "Welcome to the RAG API! Use the /query endpoint to ask questions."}

@app.post("/query/")
async def ask_question(query: Query):
    # Retrieve the answer from the engine
    response = query_engine.query(query.query)
    return {"response": str(response)}