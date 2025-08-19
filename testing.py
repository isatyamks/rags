
from src.evaluation import evaluate_and_save, report_eval
import os

def main():
    
    improve_num = int(input("improve number: "))
    eval_path = f"reports/eval{improve_num}.csv"
    
    if os.path.exists(eval_path):
        report_eval(improve_num)
    
    else:
        evaluate_and_save(improve_num)
        report_eval(improve_num)

if __name__ == "__main__":
    main()