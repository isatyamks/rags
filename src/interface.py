
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
        query = input()
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
    



def improve(qa_chain,df,csv):
    import os
    questions = df["Question"].tolist()
    answers = df['Answer'].tolist()
    csv_path = 'testing/improve2.csv'
    fieldnames = ['Question', 'ModelResult', 'ActualAnswer']
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
    for i in range(min(100, len(questions))):
        query = questions[i]
        print(query)
        result = qa_chain.invoke(query)
        if isinstance(result, dict) and 'result' in result:
            model_answer = result['result']
        else:
            model_answer = str(result)
        print(model_answer)
        actual_answer = answers[i] if i < len(answers) else ''
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow({'Question': query, 'ModelResult': model_answer, 'ActualAnswer': actual_answer})
    print('Results saved to testing/improve.csv')
