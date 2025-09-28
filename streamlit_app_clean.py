import streamlit as st
import os
import re

# Import the RAG system components
from src.pipeline import pipelinefn

# Set page config
st.set_page_config(
    page_title="RAG Chat Interface",
    page_icon="💬",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .user-message {
        background-color: #007bff;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 1rem;
        margin: 0.5rem 0;
        margin-left: 2rem;
        text-align: right;
    }
    .bot-message {
        background-color: #28a745;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 1rem;
        margin: 0.5rem 0;
        margin-right: 2rem;
        text-align: left;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'embeddings_loaded' not in st.session_state:
    st.session_state.embeddings_loaded = False

def load_qa_system():
    """Load the QA system with embeddings"""
    try:
        # Check if embeddings directory exists
        if os.path.exists("embeddings"):
            with st.spinner("Loading QA system..."):
                qa_chain = pipelinefn("embeddings")
                st.session_state.qa_chain = qa_chain
                st.session_state.embeddings_loaded = True
                st.success("QA system loaded successfully!")
                return True
        else:
            st.error("No embeddings found. Please create embeddings first by processing documents.")
            return False
    except Exception as e:
        st.error(f"Error loading QA system: {str(e)}")
        return False

def process_answer(result):
    """Process and clean the answer from the QA chain"""
    if isinstance(result, dict) and 'result' in result:
        answer = result['result']
    else:
        answer = str(result)
    
    # Clean up answer
    match = re.search(r"Helpful Answer:(.*)", answer, re.DOTALL)
    if match:
        answer = match.group(1).strip()
    match2 = re.search(r"Answer:(.*)", answer, re.DOTALL)
    if match2:
        answer = match2.group(1).strip()
    answer = answer.split('\n')[0].strip()
    
    if not answer:
        answer = result['result'] if isinstance(result, dict) and 'result' in result else str(result)
    
    return answer

def main():
    st.markdown('<h1 class="main-header">💬 RAG Chat Interface</h1>', unsafe_allow_html=True)
    
    # Load QA system if not already loaded
    if not st.session_state.embeddings_loaded:
        st.markdown("### Initialize Chat System")
        if st.button("Load QA System", type="primary", key="load_qa_system_main"):
            load_qa_system()
    
    # Show chat interface if system is loaded
    if st.session_state.qa_chain is not None:
        st.markdown("### Chat with Your Documents")
        
        # Display chat history
        if st.session_state.chat_history:
            st.markdown("#### Chat History")
            for question, answer in st.session_state.chat_history:
                st.markdown(f'<div class="user-message"><strong>You:</strong> {question}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="bot-message"><strong>Bot:</strong> {answer}</div>', unsafe_allow_html=True)
        
        # Question input
        with st.form("chat_form"):
            question = st.text_input("Ask a question:", placeholder="What would you like to know?")
            col1, col2 = st.columns([1, 1])
            with col1:
                submit_button = st.form_submit_button("Send", type="primary")
            with col2:
                clear_button = st.form_submit_button("Clear History")
        
        if clear_button:
            st.session_state.chat_history = []
            st.rerun()
        
        if submit_button and question:
            with st.spinner("Thinking..."):
                try:
                    # Get answer from QA chain
                    result = st.session_state.qa_chain.invoke(question)
                    answer = process_answer(result)
                    
                    # Add to chat history
                    st.session_state.chat_history.append((question, answer))
                    
                    # Rerun to show updated chat
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error getting answer: {str(e)}")
    
    else:
        st.markdown("""
        ### Welcome to RAG Chat Interface
        
        This is a simple chat interface for your Retrieval-Augmented Generation system.
        
        **To get started:**
        1. Make sure you have processed documents and created embeddings in the `embeddings` folder
        2. Click "Load QA System" to initialize the chat system
        3. Start asking questions about your documents!
        
        **Available embeddings folders:**
        """)
        
        # Show available embedding folders
        if os.path.exists("embeddings"):
            for item in os.listdir("embeddings"):
                if os.path.isdir(os.path.join("embeddings", item)):
                    st.write(f"- {item}")
        else:
            st.write("No embeddings folder found. Please create embeddings first.")

if __name__ == "__main__":
    main()