""" Command - line demo of the HR Policy Assistant.

Run with : python main.py

"""

from hr_assistant.pipeline import ask , build_hr_assistant

def main():
    
    print("Building the HR policy assistant...")
    agent = build_hr_assistant()
    print("Assistant ready!\n")

    demo_questions = [
        "What is the maternity leave policy?",
        "What is the notice period during probation?",
        "What is the company's policy on cryptocurrency trading?",
    ]

    for question in demo_questions:
        print("=" * 60)
        print("QUESTION:", question)
        print("-" * 60)
        answer = ask(agent, question)
        print("ANSWER:", answer)
        print("=" * 60)
        print()
        
       


if __name__ == "__main__":
    main()