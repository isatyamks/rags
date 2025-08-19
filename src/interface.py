
import os
import csv

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def terminalchat(qa_chain):

    import re
    while True:
        query = input("\n---->\n")
        if query.strip().lower() == 'exit':
            break
        result = qa_chain.invoke(query)
        if isinstance(result, dict) and 'result' in result:
            answer = result['result']
        else:
            answer = str(result)
        match = re.search(r"Helpful Answer:(.*)", answer, re.DOTALL)
        if match:
            answer = match.group(1).strip()
        match2 = re.search(r"Answer:(.*)", answer, re.DOTALL)
        if match2:
            answer = match2.group(1).strip()
        answer = answer.split('\n')[0].strip()
        if not answer:
            answer = result['result'] if isinstance(result, dict) and 'result' in result else str(result)
        print(f"\n-->{answer}")
    



def improve(qa_chain, df, csv_path='reports/improve8.csv'):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    questions = df["Question"].tolist()
    answers = df['Answer'].tolist()

    fieldnames = ['Question', 'RAG', 'ActualAnswer']

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(questions)):
            query = questions[i]
            print(query)


            result = qa_chain.invoke(query)
            if isinstance(result, dict) and 'result' in result:
                model_answer = result['result']
            else:
                model_answer = str(result)

            import re
            answer = model_answer
            match = re.search(r"Helpful Answer:(.*)", answer, re.DOTALL)
            if match:
                answer = match.group(1).strip()
            match2 = re.search(r"Answer:(.*)", answer, re.DOTALL)
            if match2:
                answer = match2.group(1).strip()
            answer = answer.split('\n')[0].strip()

            print(answer)

            actual_answer = answers[i] if i < len(answers) else ''
            writer.writerow({
                'Question': query,
                'RAG': answer,
                'ActualAnswer': actual_answer
            })

    print(f'Results saved to {csv_path}')
