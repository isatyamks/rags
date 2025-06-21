import gradio as gr

def launch_ui(qa_chain):
    def answer_question(query):
        if not query.strip():
            return "Please enter a question."
        result = qa_chain.run(query)
        return result

    interface = gr.Interface(
        fn=answer_question,
        inputs=gr.Textbox(label="Ask a question about the book"),
        outputs=gr.Textbox(label="Answer"),
        title="📚 Chat with Your Book",
        description="Powered by a local LLM and FAISS vector store"
    )

    interface.launch()
