import streamlit as st
import requests

# Configuration
API_URL = "http://localhost:8000/ask"

def ask_question(question: str):
    try:
        response = requests.post(
            API_URL,
            json={"question": question},
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {str(e)}")
        return None

# Streamlit UI
st.set_page_config(page_title="RAG System", layout="wide")

st.title("RAG System with Bert Fine Tuned")

# Sidebar
with st.sidebar:
    st.header("API Status")
    try:
        health = requests.get(API_URL.replace("/ask", ""), timeout=2)
        if health.status_code == 200:
            st.success("API Connected")
        else:
            st.warning(f"API Response: {health.status_code}")
    except:
        st.error("API Not Reachable")

# Main interface
question = st.text_input("Enter your question:", key="question_input")

if st.button("Submit"):
    if question:
        with st.spinner("Processing your question..."):
            result = ask_question(question)
            
            if result and "error" not in result:
                st.subheader("Answer:")
                st.success(result["answer"])
                
                st.subheader("Sources:")
                for i, source in enumerate(result["sources"], 1):
                    with st.expander(f"Source {i}"):
                        st.text(source)
                
                st.caption(f"Processed in {result['inference_time']:.2f} seconds")
            elif result and "error" in result:
                st.error(result["error"])
    else:
        st.warning("Please enter a question")