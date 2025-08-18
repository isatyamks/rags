
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
    questions = df["Question"].tolist()
    answers = df['Answer'].tolist()
    results = []
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
        results.append({'Question': query, 'ModelResult': model_answer, 'ActualAnswer': actual_answer})
    # Save results to a new CSV file
    with open('testing/improve.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['Question', 'ModelResult', 'ActualAnswer'])
        writer.writeheader()
        writer.writerows(results)
    print('Results saved to testing/model_vs_actual.csv')
