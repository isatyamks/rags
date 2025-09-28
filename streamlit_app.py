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
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📚 Features")
        st.markdown("""
        - **Document Embedding**: Convert text files to FAISS vector indexes
        - **Retrieval Pipeline**: Fast similarity search over embedded documents
        - **LLM Integration**: Uses fine-tuned models or GPT-2 for generation
        - **Interactive Chat**: Ask questions about your documents
        - **Evaluation Tools**: Automated accuracy reporting
        - **System Analysis**: Analyze FAISS indexes and embeddings
        """)
    
    with col2:
        st.markdown("### 🛠 How to Use")
        st.markdown("""
        1. **Document Processing**: Upload and process your text documents
        2. **Chat Interface**: Ask questions about your processed documents
        3. **Evaluation**: Test system performance with Q&A datasets
        4. **System Analysis**: Analyze your embeddings and indexes
        """)
    
    st.markdown("### 📊 System Status")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Check available documents
        raw_files = []
        if os.path.exists("data/books/raw"):
            raw_files = [f for f in os.listdir("data/books/raw") if f.endswith('.txt')]
        st.metric("Available Documents", len(raw_files))
    
    with col2:
        # Check available embeddings
        embedding_dirs = []
        if os.path.exists("embeddings"):
            embedding_dirs = [d for d in os.listdir("embeddings") if os.path.isdir(os.path.join("embeddings", d))]
        st.metric("Embedding Indexes", len(embedding_dirs))
    
    with col3:
        # Check if QA chain is loaded
        chain_status = "Loaded" if st.session_state.qa_chain else "Not Loaded"
        st.metric("QA System", chain_status)

def show_document_processing():
    st.markdown('<h2 class="section-header">📄 Document Processing</h2>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Upload Document", "Process Existing", "View Documents"])
    
    with tab1:
        st.markdown("### Upload New Document")
        uploaded_file = st.file_uploader("Choose a text file", type=['txt'])
        
        if uploaded_file is not None:
            # Create data directories if they don't exist
            os.makedirs("data/books/raw", exist_ok=True)
            
            # Save uploaded file
            file_path = os.path.join("data/books/raw", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.success(f"File uploaded successfully: {uploaded_file.name}")
            
            # Preview file content
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            st.markdown("### File Preview")
            st.text_area("Content Preview", content[:1000] + "..." if len(content) > 1000 else content, height=200)
            
            # Process file button
            if st.button("Process Document", key="process_uploaded"):
                process_document(uploaded_file.name.replace('.txt', ''))
    
    with tab2:
        st.markdown("### Process Existing Documents")
        
        # List available documents
        raw_files = []
        if os.path.exists("data/books/raw"):
            raw_files = [f.replace('.txt', '') for f in os.listdir("data/books/raw") if f.endswith('.txt')]
        
        if raw_files:
            selected_file = st.selectbox("Select a document to process:", raw_files)
            
            if st.button("Process Selected Document", key="process_existing"):
                process_document(selected_file)
        else:
            st.warning("No text files found in data/books/raw directory.")
    
    with tab3:
        st.markdown("### Available Documents")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Raw Documents")
            if os.path.exists("data/books/raw"):
                raw_files = [f for f in os.listdir("data/books/raw") if f.endswith('.txt')]
                for file in raw_files:
                    st.write(f"📄 {file}")
            else:
                st.write("No raw documents found")
        
        with col2:
            st.markdown("#### Processed Embeddings")
            if os.path.exists("embeddings"):
                embedding_dirs = [d for d in os.listdir("embeddings") if os.path.isdir(os.path.join("embeddings", d))]
                for dir_name in embedding_dirs:
                    st.write(f"🗂️ {dir_name}")
            else:
                st.write("No embeddings found")

def process_document(file_name):
    """Process a document by creating embeddings and initializing the QA chain"""
    try:
        with st.spinner(f"Processing document: {file_name}..."):
            # Create embeddings
            st.info("Creating embeddings...")
            vector_from_jsonl(f"data/books/raw/{file_name}.txt", save_path="embeddings")
            
            # Get the latest embedding directory
            embedding_dirs = [d for d in os.listdir("embeddings") if d.startswith(file_name)]
            if embedding_dirs:
                latest_dir = max(embedding_dirs)  # Get the most recent one
                
                st.info("Initializing QA system...")
                qa_chain = pipelinefn(embeddings_dir=latest_dir)
                
                # Store in session state
                st.session_state.qa_chain = qa_chain
                st.session_state.embeddings_created = True
                st.session_state.current_embeddings_dir = latest_dir
                
                st.markdown(f"""
                <div class="success-box">
                <h4>✅ Processing Complete!</h4>
                <p>Document: <strong>{file_name}</strong></p>
                <p>Embeddings saved to: <strong>embeddings/{latest_dir}</strong></p>
                <p>QA system initialized and ready for use!</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.balloons()
            else:
                st.error("Failed to create embeddings")
                
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")

def show_chat_interface():
    st.markdown('<h2 class="section-header">💬 Chat Interface</h2>', unsafe_allow_html=True)
    
    if st.session_state.qa_chain is None:
        st.markdown("""
        <div class="warning-box">
        <h4>⚠️ QA System Not Initialized</h4>
        <p>Please process a document first in the <strong>Document Processing</strong> section.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Display current system info
    if st.session_state.current_embeddings_dir:
        st.info(f"Currently using embeddings: {st.session_state.current_embeddings_dir}")
    
    # Chat interface
    st.markdown("### Ask Questions About Your Documents")
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("#### Chat History")
        for i, (question, answer) in enumerate(st.session_state.chat_history):
            with st.expander(f"Q{i+1}: {question[:50]}...", expanded=False):
                st.markdown(f"**Question:** {question}")
                st.markdown(f"**Answer:** {answer}")
    
    # Question input
    question = st.text_input("Enter your question:", placeholder="What would you like to know about the document?")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        ask_button = st.button("Ask Question", type="primary")
    with col2:
        clear_button = st.button("Clear History")
    
    if clear_button:
        st.session_state.chat_history = []
        st.rerun()
    
    if ask_button and question:
        with st.spinner("Thinking..."):
            try:
                # Get answer from QA chain
                result = st.session_state.qa_chain.invoke(question)
                
                # Extract answer
                if isinstance(result, dict) and 'result' in result:
                    answer = result['result']
                else:
                    answer = str(result)
                
                # Clean up answer
                import re
                match = re.search(r"Helpful Answer:(.*)", answer, re.DOTALL)
                if match:
                    answer = match.group(1).strip()
                match2 = re.search(r"Answer:(.*)", answer, re.DOTALL)
                if match2:
                    answer = match2.group(1).strip()
                answer = answer.split('\n')[0].strip()
                
                if not answer:
                    answer = result['result'] if isinstance(result, dict) and 'result' in result else str(result)
                
                # Add to chat history
                st.session_state.chat_history.append((question, answer))
                
                # Display current answer
                st.markdown("### Answer")
                st.markdown(f"**Question:** {question}")
                st.markdown(f"**Answer:** {answer}")
                
            except Exception as e:
                st.error(f"Error getting answer: {str(e)}")

def show_evaluation():
    st.markdown('<h2 class="section-header">📊 Evaluation</h2>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Evaluate System", "View Reports"])
    
    with tab1:
        st.markdown("### System Evaluation")
        
        if st.session_state.qa_chain is None:
            st.warning("Please initialize the QA system first by processing a document.")
            return
        
        # File upload for evaluation dataset
        uploaded_csv = st.file_uploader("Upload Q&A Dataset (CSV)", type=['csv'])
        
        if uploaded_csv is not None:
            try:
                df = pd.read_csv(uploaded_csv)
                st.markdown("#### Dataset Preview")
                st.dataframe(df.head())
                
                # Check if required columns exist
                required_cols = ['Question']
                if all(col in df.columns for col in required_cols):
                    if st.button("Run Evaluation"):
                        run_evaluation(df)
                else:
                    st.error(f"CSV must contain columns: {required_cols}")
                    
            except Exception as e:
                st.error(f"Error reading CSV: {str(e)}")
        
        # Use existing evaluation files
        st.markdown("#### Or use existing Q&A files")
        qa_files = []
        if os.path.exists("data"):
            for root, dirs, files in os.walk("data"):
                qa_files.extend([os.path.join(root, f) for f in files if f.endswith('.csv') and 'qa' in f.lower()])
        
        if qa_files:
            selected_qa_file = st.selectbox("Select existing Q&A file:", qa_files)
            if st.button("Run Evaluation on Selected File"):
                try:
                    df = pd.read_csv(selected_qa_file)
                    run_evaluation(df)
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
    
    with tab2:
        st.markdown("### Evaluation Reports")
        
        # List available reports
        if os.path.exists("reports"):
            report_files = [f for f in os.listdir("reports") if f.endswith('.csv')]
            
            if report_files:
                selected_report = st.selectbox("Select report to view:", report_files)
                
                if selected_report:
                    try:
                        report_path = os.path.join("reports", selected_report)
                        df = pd.read_csv(report_path)
                        
                        st.markdown(f"#### Report: {selected_report}")
                        st.dataframe(df)
                        
                        # Download button
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="Download Report",
                            data=csv,
                            file_name=selected_report,
                            mime='text/csv'
                        )
                        
                    except Exception as e:
                        st.error(f"Error reading report: {str(e)}")
            else:
                st.info("No evaluation reports found.")
        else:
            st.info("Reports directory not found.")

def run_evaluation(df):
    """Run evaluation on the provided dataset"""
    try:
        with st.spinner("Running evaluation..."):
            # Save temporary CSV for the evaluation function
            temp_csv_path = "temp_evaluation.csv"
            df.to_csv(temp_csv_path, index=False)
            
            # Import improve function from interface
            from src.interface import improve
            
            # Create timestamp for report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = f"reports/streamlit_eval_{timestamp}.csv"
            
            # Run evaluation
            improve(st.session_state.qa_chain, df, csv_path=report_path)
            
            # Clean up temp file
            if os.path.exists(temp_csv_path):
                os.remove(temp_csv_path)
            
            # Show results
            if os.path.exists(report_path):
                result_df = pd.read_csv(report_path)
                st.success("Evaluation completed!")
                st.markdown("#### Results")
                st.dataframe(result_df)
                
                # Calculate metrics if possible
                if 'ActualAnswer' in result_df.columns and 'RAG' in result_df.columns:
                    # Simple accuracy calculation (you might want to implement semantic similarity)
                    total_questions = len(result_df)
                    st.metric("Total Questions", total_questions)
                    
                    # Download results
                    csv = result_df.to_csv(index=False)
                    st.download_button(
                        label="Download Results",
                        data=csv,
                        file_name=f"evaluation_results_{timestamp}.csv",
                        mime='text/csv'
                    )
            else:
                st.error("Evaluation completed but no results file found.")
                
    except Exception as e:
        st.error(f"Error during evaluation: {str(e)}")

def show_system_analysis():
    st.markdown('<h2 class="section-header">🔍 System Analysis</h2>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["FAISS Index Analysis", "System Statistics"])
    
    with tab1:
        st.markdown("### FAISS Index Analysis")
        
        # List available embedding directories
        if os.path.exists("embeddings"):
            embedding_dirs = [d for d in os.listdir("embeddings") if os.path.isdir(os.path.join("embeddings", d))]
            
            if embedding_dirs:
                selected_dir = st.selectbox("Select embedding directory to analyze:", embedding_dirs)
                
                if st.button("Analyze Index"):
                    try:
                        index_path = os.path.join("embeddings", selected_dir)
                        
                        # Capture the analysis output
                        import io
                        import sys
                        
                        old_stdout = sys.stdout
                        sys.stdout = buffer = io.StringIO()
                        
                        try:
                            analyze_faiss_index(index_path)
                            analysis_output = buffer.getvalue()
                        finally:
                            sys.stdout = old_stdout
                        
                        st.markdown("#### Analysis Results")
                        st.code(analysis_output, language="text")
                        
                    except Exception as e:
                        st.error(f"Error analyzing index: {str(e)}")
            else:
                st.info("No embedding directories found.")
        else:
            st.info("Embeddings directory not found.")
    
    with tab2:
        st.markdown("### System Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### File Statistics")
            
            # Count raw files
            raw_count = 0
            if os.path.exists("data/books/raw"):
                raw_count = len([f for f in os.listdir("data/books/raw") if f.endswith('.txt')])
            st.metric("Raw Text Files", raw_count)
            
            # Count JSONL files
            jsonl_count = 0
            if os.path.exists("data/books/jsonl"):
                jsonl_count = len([f for f in os.listdir("data/books/jsonl") if f.endswith('.jsonl')])
            st.metric("JSONL Files", jsonl_count)
            
            # Count embedding directories
            embedding_count = 0
            if os.path.exists("embeddings"):
                embedding_count = len([d for d in os.listdir("embeddings") if os.path.isdir(os.path.join("embeddings", d))])
            st.metric("Embedding Indexes", embedding_count)
        
        with col2:
            st.markdown("#### Model Information")
            
            # Check for fine-tuned models
            model_count = 0
            if os.path.exists("models"):
                model_count = len([d for d in os.listdir("models") if os.path.isdir(os.path.join("models", d))])
            st.metric("Fine-tuned Models", model_count)
            
            # Current model info
            if st.session_state.qa_chain:
                st.success("QA System: Active")
            else:
                st.info("QA System: Inactive")
        
        # Directory structure
        st.markdown("#### Directory Structure")
        structure = []
        
        for root, dirs, files in os.walk("."):
            # Skip hidden directories and __pycache__
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            level = root.replace(".", "").count(os.sep)
            indent = " " * 2 * level
            structure.append(f"{indent}{os.path.basename(root)}/")
            
            subindent = " " * 2 * (level + 1)
            for file in files[:5]:  # Limit to first 5 files per directory
                if not file.startswith('.') and not file.endswith('.pyc'):
                    structure.append(f"{subindent}{file}")
            if len(files) > 5:
                structure.append(f"{subindent}... and {len(files) - 5} more files")
        
        st.code("\n".join(structure[:50]))  # Limit output

if __name__ == "__main__":
    main()