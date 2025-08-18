
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import gradio as gr

def launch_ui(qa_chain):
    def answer_question(query):
        if not query.strip():
            return "Please enter a question."
        result = qa_chain.invoke(query)
        return result

    interface = gr.Interface(
        fn=answer_question,
        inputs=gr.Textbox(label="Ask a question about the book"),
        outputs=gr.Textbox(label="Answer"),
        title="Rag based LLM",
        description="Powered by a local LLM and FAISS vector store"
    )

    interface.launch()

    

def terminal(qa_chain):

    print("\n")
    print("RAG based LLM start to talk.. (type 'exit' or 'quit' to stop)")
    print("\n")
    while True:
        query = input("say!\n")
        if query.strip().lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        if not query.strip():
            print("please enter a question!")
            continue
        result = qa_chain.invoke(query)
        if isinstance(result, dict) and 'result' in result:
            print(result['result'])
        else:
            print(result)
    




