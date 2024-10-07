from typing import Dict, Any , List
from llama_index.core import VectorStoreIndex
from llama_index.core.llms import ChatMessage
from groq import Groq
from serpapi.google_search import GoogleSearch
import os 

async def query_engine(index: VectorStoreIndex, query: str) -> Dict[str, Any]:
    query_engine = index.as_query_engine(similarity_top_k=5, verbose=True)
    print(f"QueryEngine: Executing query: {query}")
    response = query_engine.query(query)
    return {
        "response": response.response,
        "source_nodes": [node.node.get_content() for node in response.source_nodes],
    }

async def generate_initial_response(llm: Groq, query: str, source_nodes: List[str]) -> Dict[str, Any]:
    messages = [
        ChatMessage(role="system", content="Generate a response based ONLY on the source information. You can make abbreviations and assign variables if needed but do not assume any numerical or categorical value which is not present in the source infromation.Fomulate your answers in a proper fashion like writing the answers for a exam"),
        ChatMessage(role="user", content=f"Query: {query}\n\nSource Information:\n{' '.join(source_nodes)}"),
    ]
    response = llm.chat(messages)
    return {"response": response.message.content}


async def analyze_response(llm: Groq, initial_response: str, query: str) -> Dict[str, Any]:
    print(f"AdvancedQueryAgent: Analyzing response for query: {query}")
    messages = [
        ChatMessage(role="system", content="Analyze the initial_response and determine if additional information is needed based on what the query is asking and what is required to complete the answer. If so, specify it as a simple, direct question without using any variables or abbreviations. "),
        ChatMessage(role="user", content=f"Query: {query}\nInitial Response: {initial_response}\n\nIs additional information required? If yes, what specific information is needed? Please provide the follow-up question in simple, clear language that is formulted to ask a Vector-store. Only generate the follow-up question")
    ]
    response = llm.chat(messages)
    return {"analysis": response.message.content}

async def generate_final_answer(llm: Groq, query: str, initial_response: str, additional_info: str) -> Dict[str, Any]:
    print(f"AdvancedQueryAgent: Generating final answer for query: {query}")
    messages = [
        ChatMessage(role="system", content="Generate a comprehensive final answer based on the given information."),
        ChatMessage(role="user", content=f"Original query: {query}\nInitial response: {initial_response}\nAdditional information: {additional_info}\n\nPlease provide a complete and accurate final answer.")
    ]
    response = llm.chat(messages)
    return {"final_answer": response.message.content}

async def web_search_tool(query: str) -> Dict[str, Any]:
    print(f"Web Search Tool: Searching for query: {query}")
    
    serpapi_params = {
        "engine": "google",
        "api_key": os.getenv("SERPAPI_KEY"),
        "q": query,
        "num": 5
    }
    
    search = GoogleSearch(serpapi_params)
    results = search.get_dict().get("organic_results", [])
    
    if results:
        contexts = "\n---\n".join(
            [f"Title: {x['title']}\nSnippet: {x['snippet']}\nLink: {x['link']}" for x in results]
        )
    else:
        contexts = "No results found."
    
    return {
        "response": contexts,
        "source": "Web"
    }

async def handle_human_message(llm: Groq, message: str) -> Dict[str, Any]:
    print(f"LLM Agent: Handling human message: {message}")
    
    messages = [
        ChatMessage(role="system", content="Respond as a friendly assistant. If the message seems like a greeting or informal question, reply accordingly."),
        ChatMessage(role="user", content=f"Message: {message}")
    ]
    
    response = llm.chat(messages)
    return {"response": response.message.content}

async def classify_query(llm: Groq, query: str) -> str:
    print(f"LLM Router: Classifying query: {query}")
    
    messages = [
        ChatMessage(role="system", content="""You are an expert in routing user queries to the appropriate category: 'greeting', 'index_search', or 'web_search'.

        If the user's message is a greeting or a casual interaction, categorize it as 'greeting'.
        Use 'index_search' for queries related to sound and its properties. This includes but is not limited to:
        Production and propagation of sound waves
        Wave characteristics (frequency, amplitude, etc.)
        Sound reflection, echo, and reverberation
        Speed of sound in different media
        Human hearing and perception of sound
        Infrasound, ultrasound, and their applications (e.g., medical, industrial)
        You do not need to rely strictly on keywords but should focus on whether the query is broadly related to sound and its scientific aspects.
        For all other queries that don't fit these categories, choose 'web_search'.

        Your only response should be one of the following: 'greeting', 'index_search', or 'web_search'.
"""),
        ChatMessage(role="user", content=f"Query: {query}")
    ]
    
    response = llm.chat(messages)
    classification = response.message.content.strip().lower()
    print(classification)
    return classification

# Add any other tool functions here