"""
Run the correctness evaluation and 
upoad the results to langsmith

Run with : python evaluate.py

"""

#display the marksheet to the parents of students


from hr_assistant.evaluation import run_evaluation

def main():
    print("RUNING THE HR POLICY ASSISTANT EVAL....")
    results = run_evaluation()
    print("Done open your langsmith and see your experiment in the langsmith")
    print(results)

if __name__ == "__main__":
    run_evaluation()