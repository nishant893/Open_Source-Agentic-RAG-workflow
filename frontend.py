import streamlit as st
import requests
import json

# Set the API endpoints
QUERY_ENDPOINT = "http://localhost:8000/query"
FEEDBACK_ENDPOINT = "http://localhost:8000/feedback"

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")

st.title("RAG Chatbot")

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'awaiting_feedback' not in st.session_state:
    st.session_state.awaiting_feedback = False
if 'current_query' not in st.session_state:
    st.session_state.current_query = ""
if 'current_response' not in st.session_state:
    st.session_state.current_response = ""

# Function to send query to API and get response
def query_api(query):
    try:
        response = requests.post(QUERY_ENDPOINT, json={"query": query})
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"API request failed: {e}")
        return None

# Function to send feedback to API and get refined response
def send_feedback(query, feedback, initial_response):
    try:
        response = requests.post(FEEDBACK_ENDPOINT, json={"query": query, "feedback": feedback, "initial_response": initial_response})
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"API request failed: {e}")
        return None

# Chat interface
st.subheader("Chat")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user feedback if awaiting
if st.session_state.awaiting_feedback:
    st.info("Is the above response satisfactory?")
    st.info("If you select 'No', the system will perform an advanced query search to provide a more comprehensive response.")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Yes"):
            st.session_state.awaiting_feedback = False
            st.rerun()
    with col2:
        if st.button("No"):
            with st.spinner("Generating a more detailed response..."):
                refined_response = send_feedback(st.session_state.current_query, "no", st.session_state.current_response)
            if refined_response:
                response_content = refined_response.get("response", "No refined response available.")
                st.session_state.chat_history.append({"role": "assistant", "content": response_content})
                with st.chat_message("assistant"):
                    st.markdown(response_content)
            st.session_state.awaiting_feedback = False
            st.rerun()

# Input for user query
if not st.session_state.awaiting_feedback:
    user_query = st.chat_input("Type your message here...")

    if user_query:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Get API response
        with st.spinner("Thinking..."):
            api_response = query_api(user_query)

        if api_response:
            # Add assistant response to chat history
            response_content = api_response.get("response", "No response")
            st.session_state.chat_history.append({"role": "assistant", "content": response_content})
            with st.chat_message("assistant"):
                st.markdown(response_content)

            # Store current query and response for potential feedback
            st.session_state.current_query = user_query
            st.session_state.current_response = response_content

            # Check if feedback is required
            if api_response.get("requires_feedback", False):
                st.session_state.awaiting_feedback = True
                st.rerun()

# Sidebar for additional options
st.sidebar.title("Options")

# Clear chat history
if st.sidebar.button("Clear Chat History"):
    st.session_state.chat_history = []
    st.session_state.awaiting_feedback = False
    st.session_state.current_query = ""
    st.session_state.current_response = ""
    st.rerun()

# Display system information
st.sidebar.subheader("System Information")
st.sidebar.info(
    "This RAG (Retrieval-Augmented Generation) chatbot uses a vector database to retrieve relevant information "
    "and generate responses based on your queries. It can handle questions about sound and its properties, "
    "as well as perform web searches for other topics. You can provide feedback on responses to get more detailed information."
)
