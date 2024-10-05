import os
from dotenv import load_dotenv
from typing import Dict, Any
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import MarkdownElementNodeParser
from groq import Groq
import chromadb



from typing import Dict, List, Any
from llama_index.core.workflow import Workflow, StartEvent, StopEvent, step, Event
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.llms import MessageRole
from llama_index.core.tools import FunctionTool
from groq import Groq


import os
import getpass
from llama_index.llms.groq import Groq
from llama_parse import LlamaParse
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.core import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from IPython.display import Markdown, display
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import Settings

from llama_index.core.node_parser import MarkdownElementNodeParser
from llama_index.core.llms import ChatMessage
from serpapi.google_search import GoogleSearch




# Environment setup for API keys
def set_api_keys():
    os.environ["LLAMA_CLOUD_API_KEY"] = getpass.getpass("LLamaParse API Key:")
    os.environ["GROQ_API_KEY"] = getpass.getpass("GROQ_API_KEY:")
    os.environ["SerpAPI_key"] = getpass.getpass("SerpAPI_key:")



def setup_llm_and_embedding():
    llm = Groq(model="llama3-70b-8192", api_key="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")
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    # Set global settings
    Settings.llm = llm
    Settings.embed_model = embed_model

    return llm, embed_model



def setup_parser():
    return LlamaParse(
        result_type="markdown",
        verbose=True,
        language="en",
        num_workers=1,
        premium_mode=True
    )



def create_and_save_index(embed_model, db_path="./chroma_db_new"):
    parser = setup_parser()
    documents = parser.load_data([r"C:\Users\nisha\Desktop\RAG_Assign\RAG_Assign\iesc111.pdf"])
    
    node_parser = MarkdownElementNodeParser(llm=Settings.llm, num_workers=8)
    nodes_markdown_element = node_parser.get_nodes_from_documents(documents=documents)
    base_nodes, objects = node_parser.get_nodes_and_objects(nodes_markdown_element)
    
    # Create or connect to the persistent ChromaDB
    db = chromadb.PersistentClient(path=db_path)
    chroma_collection = db.get_or_create_collection("quickstart")

    # Create ChromaVectorStore
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    # Create storage context
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Create index
    index = VectorStoreIndex(nodes=base_nodes+objects, storage_context=storage_context)
    
    return index


def load_index(embed_model, db_path="./chroma_db_new"):
    # Load the index from persistent storage
    db = chromadb.PersistentClient(path=db_path)
    chroma_collection = db.get_or_create_collection("quickstart")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # Create the index from the vector store
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model
    )

    return index

def create_query_engine(index, llm, top_k=5, verbose=True):
    return index.as_query_engine(llm=llm, similarity_top_k=top_k, verbose=verbose)

# Add any other utility functions here