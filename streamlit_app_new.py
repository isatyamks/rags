import streamlit as st
import os
import re
import tempfile

# Import the RAG system components
from src.embedder import vector_from_jsonl
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
    .upload-section {
        border: 2px dashed #ccc;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'document_processed' not in st.session_state:
    st.session_state.document_processed = False
if 'current_document' not in st.session_state:
    st.session_state.current_document = None

def process_document(uploaded_file):
    """Process uploaded document: save, create embeddings, and initialize QA system"""
    try:
        # Create directories if they don't exist
        os.makedirs("data/books/raw", exist_ok=True)
        
        # Save uploaded file
        file_name = uploaded_file.name.replace('.txt', '')
        file_path = os.path.join("data/books/raw", uploaded_file.name)
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"✅ File saved: {uploaded_file.name}")
        
        # Create embeddings
        with st.spinner("🔄 Creating embeddings..."):
            vector_from_jsonl(file_path, save_path="embeddings")
            st.success("✅ Embeddings created successfully!")
        
        # Initialize QA system
        with st.spinner("🤖 Initializing chat system..."):
            qa_chain = pipelinefn("embeddings")
            st.session_state.qa_chain = qa_chain
            st.session_state.document_processed = True
            st.session_state.current_document = uploaded_file.name
            st.success("✅ Chat system ready!")
        
        st.balloons()
        return True
        
    except Exception as e:
        st.error(f"❌ Error processing document: {str(e)}")
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
    
    # Show current document status
    if st.session_state.current_document:
        st.info(f"📄 Current document: {st.session_state.current_document}")
    
    # Document upload section
    if not st.session_state.document_processed:
        st.markdown("""
        <div class="upload-section">
        <h3>📁 Upload Your Document</h3>
        <p>Upload a .txt file to start chatting with your document</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose a text file", 
            type=['txt'],
            help="Upload a .txt file and it will be automatically processed for chat"
        )
        
        if uploaded_file is not None:
            # Show file preview
            content = uploaded_file.read().decode('utf-8')
            st.markdown("### 📖 Document Preview")
            st.text_area(
                "Content Preview", 
                content[:500] + "..." if len(content) > 500 else content, 
                height=150,
                disabled=True
            )
            
            # Process automatically
            if st.button("🚀 Process Document & Start Chat", type="primary", key="process_doc"):
                # Reset the file pointer
                uploaded_file.seek(0)
                if process_document(uploaded_file):
                    st.rerun()
    
    # Chat interface (only show if document is processed)
    elif st.session_state.qa_chain is not None:
        st.markdown("### 💬 Chat with Your Document")
        
        # Reset button
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🔄 Upload New Document", key="reset_doc"):
                st.session_state.qa_chain = None
                st.session_state.document_processed = False
                st.session_state.current_document = None
                st.session_state.chat_history = []
                st.rerun()
        
        # Display chat history
        if st.session_state.chat_history:
            st.markdown("#### 📝 Chat History")
            for question, answer in st.session_state.chat_history:
                st.markdown(f'<div class="user-message"><strong>You:</strong> {question}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="bot-message"><strong>Bot:</strong> {answer}</div>', unsafe_allow_html=True)
        
        # Question input
        with st.form("chat_form"):
            question = st.text_input(
                "Ask a question about your document:", 
                placeholder="What would you like to know about the document?",
                key="question_input"
            )
            col1, col2 = st.columns([1, 1])
            with col1:
                submit_button = st.form_submit_button("💬 Send", type="primary")
            with col2:
                clear_button = st.form_submit_button("🗑️ Clear History")
        
        if clear_button:
            st.session_state.chat_history = []
            st.rerun()
        
        if submit_button and question:
            with st.spinner("🤔 Thinking..."):
                try:
                    # Get answer from QA chain
                    result = st.session_state.qa_chain.invoke(question)
                    answer = process_answer(result)
                    
                    # Add to chat history
                    st.session_state.chat_history.append((question, answer))
                    
                    # Rerun to show updated chat
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error getting answer: {str(e)}")
    
    else:
        st.error("❌ Something went wrong. Please refresh the page and try again.")
        if st.button("🔄 Refresh", key="refresh_page"):
            st.rerun()

if __name__ == "__main__":
    main()