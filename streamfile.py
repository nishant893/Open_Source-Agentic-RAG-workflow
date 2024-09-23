import streamlit as st
import requests

# Title for the web app
st.title("RAG-Based Query Application")

# Input field for query
query = st.text_input("Enter your query:")

# Button to submit the query
if st.button("Submit"):
    if query:
        # Define the FastAPI URL
        url = "http://127.0.0.1:8000/query/"
        
        # Send the request to FastAPI
        response = requests.post(url, json={"query": query})
        
        # Parse and display the response
        if response.status_code == 200:
            result = response.json()["response"]
            st.write(f"**Query:** {query}")
            st.write(f"**Response:** {result}")
        else:
            st.error(f"Error: {response.status_code}")
    else:
        st.warning("Please enter a query!")
