# RAG_Assign
 
# Agentic RAG System Explanation

This system is an advanced Retrieval-Augmented Generation (RAG) chatbot that utilizes an agentic approach, leveraging tools and workflows to handle complex queries and produce refined results. Let's break down each component and explain how they contribute to the system's agentic nature.

## File Breakdown and Functionality

### 1. tools.py

This file defines the core tools used by the agentic RAG system:

- `query_engine`: Retrieves information from the vector index.
- `analyze_response`: Determines if additional information is needed based on the initial response.
- `generate_final_answer`: Produces a comprehensive final answer using all available information.
- `web_search_tool`: Performs web searches for queries outside the system's knowledge base.
- `handle_human_message`: Processes informal or greeting-type messages.
- `classify_query`: Categorizes incoming queries to determine the appropriate action.

These tools act as the system's capabilities, allowing it to perform various tasks autonomously.

### 2. utils.py

This file contains utility functions for setting up the system:

- `set_api_keys`: Ensures all necessary API keys are available.
- `setup_llm_and_embedding`: Initializes the language model and embedding model.
- `create_and_save_index`: Creates and persists the vector index.
- `load_index`: Loads a previously created index.
- `create_query_engine`: Sets up the query engine with specified parameters.

These utilities provide the foundational components needed for the agentic system to operate.

### 3. workflow.py

This is the heart of the agentic system, defining the `RAGSystem` class which inherits from `Workflow`. It orchestrates the entire process of handling queries and generating responses:

- `handle_initial_query`: Classifies the query and determines the initial action.
- `process_initial_response`: Handles the output from the initial query processing.
- `handle_user_decision`: Processes user feedback on the initial response.
- `process_analysis`: Analyzes the need for additional information.
- `fetch_additional_info`: Retrieves extra information if needed.
- `generate_final_answer`: Produces the final, comprehensive answer.

The workflow uses the `@step` decorator to define each stage of the process, allowing the system to make decisions and take actions based on the current state and available information.

### 4. new_api.py

This file sets up the FastAPI backend, which serves as the interface between the frontend and the agentic RAG system:

- Initializes the global components (index, LLM, RAG system) on startup.
- Defines endpoints for handling queries (`/query`) and user feedback (`/feedback`).
- Manages the interaction between the frontend and the RAG system.

### 5. frontend.py

The Streamlit-based frontend provides the user interface for interacting with the agentic RAG system:

- Displays the chat history and handles user input.
- Sends queries to the backend and displays responses.
- Manages the feedback mechanism for refining responses.

## Agentic Properties and Functionality

The system demonstrates its agentic nature through several key features:

1. **Autonomous Decision Making**: The `classify_query` function allows the system to autonomously decide how to handle different types of queries (index-related, web search, or human messages).

2. **Tool Selection**: Based on the query classification and analysis, the system selects the appropriate tools (query_engine, web_search_tool, etc.) to handle the request.

3. **Workflow-based Processing**: The `RAGSystem` class in `workflow.py` defines a series of steps that the agent follows, making decisions at each stage based on the current context and available information.

4. **Self-improvement through Feedback**: The system can refine its responses based on user feedback, demonstrating a form of self-improvement and adaptation.

5. **Multi-query Type Handling**: 
   - For index-related queries, it uses the `query_engine` to retrieve information from the vector store.
   - For web searches, it employs the `web_search_tool` to fetch information from external sources.
   - For informal queries or greetings, it uses the `handle_human_message` function to generate appropriate responses.

6. **Self-query for Better Results**: The system can perform self-queries to improve its responses:
   - After the initial response, it uses the `analyze_response` tool to determine if more information is needed.
   - If additional information is required, it formulates a follow-up query (self-query) using the `process_analysis` step.
   - It then uses this self-generated query to fetch additional information and produce a more comprehensive final answer.

## System Flow

1. User submits a query through the Streamlit frontend.
2. The query is sent to the FastAPI backend.
3. The RAG system classifies the query and selects the appropriate tool.
4. An initial response is generated and returned to the user.
5. The user provides feedback on the response quality.
6. If the response is unsatisfactory, the system performs self-analysis and generates a follow-up query.
7. Additional information is retrieved based on the self-query.
8. A final, refined answer is generated and presented to the user.

This agentic approach allows the RAG system to handle complex queries, adapt to user needs, and continuously improve its responses through a combination of retrieval, generation, and self-querying mechanisms.


## Setup

### Prerequisites

- Python 3.7+
- pip

### Installation

1. Clone the repository:
   ```
   git clone [repository-url]
   cd [repository-name]
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   Create a `.env` file in the project root and add the following:
   ```
   LLAMA_CLOUD_API_KEY=your_llama_cloud_api_key
   GROQ_API_KEY=your_groq_api_key
   SERPAPI_KEY=your_serpapi_key
   ```

### Running the Application

1. Start the FastAPI backend:
   ```
   uvicorn new_api:app --reload
   ```

2. In a separate terminal, run the Streamlit frontend:
   ```
   streamlit run frontend.py
   ```

3. Open a web browser and navigate to `http://localhost:8501` to access the chat interface.

## Usage

1. Enter your query in the chat input box.
2. The system will process your query through the following steps:
   a. Query classification
   b. Initial response generation
   c. User feedback collection
   d. Response refinement (if necessary)
3. If the system requests feedback, indicate whether the response was satisfactory.
4. For unsatisfactory responses, the system will perform an advanced query to provide more detailed information.


## NOTE: 
1. The src.ipynb file has the raw code for the system.
2. The files streamfile.py , api.py , rag_setup.py are the files for the basic RAG doc QnA system. The usage is similar to the Agentic RAG system.


## Contributing

Contributions to improve the system are welcome. Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Make your changes
4. Commit your changes (`git commit -am 'Add some feature'`)
5. Push to the branch (`git push origin feature/your-feature`)
6. Create a new Pull Request


## Acknowledgments

- LlamaIndex for vector indexing and retrieval
- Groq for language modeling
- FastAPI and Streamlit for backend and frontend frameworks