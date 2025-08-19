from src.evaluation import evaluate_report

def main():
    improve_num = int(input("improve number: "))
    evaluate_report(improve_num, save=True)  # handles both evaluate + report

if __name__ == "__main__":
    main()
