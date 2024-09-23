
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


# Environment setup for API keys
def set_api_keys():
    os.environ["LLAMA_CLOUD_API_KEY"] = getpass.getpass("LLamaParse API Key:")
    os.environ["GROQ_API_KEY"] = getpass.getpass("GROQ_API_KEY:")


# Parser setup
def setup_parser():
    return LlamaParse(
        result_type="markdown",
        verbose=True,
        language="en",
        num_workers=1,
        premium_mode=True
    )

def setup_llm_and_embedding():
    llm = Groq(model="llama3-70b-8192", api_key="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")
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    # Set global settings
    Settings.llm = llm
    Settings.embed_model = embed_model

    return llm, embed_model


def index_setup(documents, embed_model, db_path="./chroma_db"):
    # Create or connect to the persistent ChromaDB
    db = chromadb.PersistentClient(path=db_path)
    chroma_collection = db.get_or_create_collection("quickstart")

    # Create ChromaVectorStore
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    # Create storage context and index
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, embed_model=embed_model
    )

    return index


def load_index(embed_model, db_path="./chroma_db"):
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


# Query engine setup
def create_query_engine(index,llm, top_k=5, verbose=True):
    return index.as_query_engine(llm = llm  ,similarity_top_k=top_k, verbose=verbose)



# Main application setup
def main():
    # Set API keys
    set_api_keys()

    # Setup parser
    #parser = setup_parser()

    # Load documents
    #documents = parser.load_data([r"C:\Users\nisha\Desktop\RAG_Assign\RAG_Assign\iesc111.pdf"])

    # Setup LLM and embedding model
    llm, embed_model = setup_llm_and_embedding()

    # Setup index
    index = load_index(embed_model)

    # Create query engine
    recursive_query_engine = create_query_engine(index,llm)

    # Example query
    query = "Why is sound wave called a longitudinal wave?"
    response = recursive_query_engine.query(query)
    print(response)


if __name__ == "__main__":
    main()
