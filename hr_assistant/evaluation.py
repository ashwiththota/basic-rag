"""
Step 9: evaluate answer quality against a fixed set 
of test questions.

Unlike tracing (which just records what happened), 
evaluation runs the
agent against a known set of question/reference-answer 
pairs and scores
each answer using a second LLM as a judge.
Results are uploaded to
LangSmith as a Dataset + Experiment, 
so quality can be compared across
runs (after a prompt change, a new model, a new guardrail, etc).

The judge model is routed through Portkey too,
using the same slug as
the main app's LLM (gateway.py's PRIMARY_PROVIDER) 
but a different
underlying model (JUDGE_MODEL_NAME) - 
so it isn't grading its own
output verbatim, without needing a second slug set up.

"""




from langchain_openai import ChatOpenAI
from hr_assistant import config
from hr_assistant.gateway import PRIMARY_TARGET , JUDGE_PROVIDER
from hr_assistant.logger import get_logger
from hr_assistant.pipeline import ask , build_hr_assistant
from hr_assistant.vector_store import get_retriever,load_vector_store

from langsmith import Client

from openevals.llm import create_llm_as_judge

from openevals.prompts import CORRECTNESS_PROMPT , RAG_GROUNDEDNESS_PROMPT

from portkey_ai import createHeaders , PORTKEY_GATEWAY_URL

logger = get_logger(__name__)

# preparing the question paper -> dataset
DATASET_NAME = "hr-policy-qna"

TEST_CASES = [
    # --- CHAPTER 1 & 2: LEADERSHIP & WORK SCHEDULES ---
    {
        "question": "Who is the CEO of Apex Global Technologies, and when does he hold open office hours?",
        "answer": "The CEO is Dr. Aris Thorne, and he holds weekly open-office hours on Thursdays from 3:00 PM to 4:00 PM EST.",
        "reference_doc": "Document 1, Chapter 1.2"
    },
    {
        "question": "What are the core hours during which all full-time employees must be available?",
        "answer": "Core Hours are from 10:00 AM to 4:00 PM local time.",
        "reference_doc": "Document 1, Chapter 2.1"
    },
    {
        "question": "How many days per week are hybrid employees required to work in the office?",
        "answer": "Hybrid employees are required to work in-office a minimum of 3 designated days per week.",
        "reference_doc": "Document 1, Chapter 2.2"
    },
    {
        "question": "What internet speed is required for fully remote employees?",
        "answer": "A minimum download speed of 50 Mbps.",
        "reference_doc": "Document 1, Chapter 2.2"
    },
    {
        "question": "How many unexcused tardy incidents trigger an informal verbal warning from HR?",
        "answer": "3 unexcused tardy incidents within a 30-day rolling window.",
        "reference_doc": "Document 1, Chapter 2.3"
    },

    # --- CHAPTER 3: LEAVE POLICIES ---
    {
        "question": "How many days of Paid Time Off (PTO) do full-time employees accrue per year?",
        "answer": "18 days per calendar year (accrued at 1.5 days per completed calendar month).",
        "reference_doc": "Document 1, Chapter 3.1"
    },
    {
        "question": "How many unused PTO days can be carried forward into the next year, and by what date must they be used?",
        "answer": "A maximum of 5 unused PTO days can be carried forward, and they must be used by March 31 of the new year.",
        "reference_doc": "Document 1, Chapter 3.1"
    },
    {
        "question": "How many paid sick days do employees get per year, and when is a medical certificate required?",
        "answer": "Employees receive 10 paid sick days per year. A doctor's medical certificate is required for sick leave extending to 3 or more consecutive business days.",
        "reference_doc": "Document 1, Chapter 3.2"
    },
    {
        "question": "How long is paid maternity leave, and how far in advance must it be requested?",
        "answer": "26 weeks of paid maternity leave, which must be requested 60 days prior to the expected delivery date.",
        "reference_doc": "Document 1, Chapter 3.3"
    },
    {
        "question": "How many days of paid bereavement leave are allowed for immediate family members?",
        "answer": "Up to 5 consecutive paid working days.",
        "reference_doc": "Document 1, Chapter 3.4"
    },

    # --- CHAPTER 4 & 5: CODE OF CONDUCT & PERFORMANCE ---
    {
        "question": "What are the rules for corporate password length and password updates?",
        "answer": "Passwords must be a minimum of 14 characters long and updated every 90 days.",
        "reference_doc": "Document 1, Chapter 4.2"
    },
    {
        "question": "How long is the probationary period for new hires, and when are formal reviews conducted?",
        "answer": "The probationary period is 90 days. Formal reviews are conducted at 45 days and 90 days.",
        "reference_doc": "Document 1, Chapter 5.2"
    },
    {
        "question": "When do performance reviews take place each year?",
        "answer": "Bi-annually: the Mid-Year Review occurs in July and the End-of-Year Review occurs in December.",
        "reference_doc": "Document 1, Chapter 5.1"
    },

    # --- CHAPTER 6 & 7: EXPENSES & RESIGNATION ---
    {
        "question": "What is the monthly wellness allowance stipend, and when must reimbursement receipts be submitted?",
        "answer": "$100 USD (or equivalent local currency) per month, with receipts submitted by the 20th of each month.",
        "reference_doc": "Document 1, Chapter 6.2"
    },
    {
        "question": "What are the daily meal allowance caps for domestic and international business travel?",
        "answer": "$75 per day for domestic travel and $120 per day for international travel.",
        "reference_doc": "Document 1, Chapter 6.3"
    },
    {
        "question": "Within how many days after an expense occurs must an expense report be submitted before it is rejected?",
        "answer": "Expense reports submitted more than 30 days after the expense occurred will be automatically rejected.",
        "reference_doc": "Document 1, Chapter 6.3"
    },
    {
        "question": "What is the notice period for voluntary resignation for individual contributors versus managers and executives?",
        "answer": "Individual contributors must give 30 days' written notice; managers, directors, and executives must give 60 days' written notice.",
        "reference_doc": "Document 1, Chapter 7.1"
    },
    {
        "question": "Within how many days after the last working day is the full and final settlement processed?",
        "answer": "Within 14 calendar days.",
        "reference_doc": "Document 1, Chapter 7.2"
    },

    # --- DOCUMENT 2: IT SECURITY & DATA GOVERNANCE ---
    {
        "question": "Within how many hours must a lost or stolen device be reported to IT Security?",
        "answer": "Within 2 hours of discovery.",
        "reference_doc": "Document 2, Section 1.2"
    },
    {
        "question": "What is the policy regarding storing Tier 4 data on local developer drives?",
        "answer": "Tier 4 data must never be downloaded onto local drives. Storing it locally is a Tier 1 Security Violation that can lead to suspension.",
        "reference_doc": "Document 2, Section 3.1"
    },
    {
        "question": "What is the required notification timeframe for a Severity 1 security breach?",
        "answer": "The CEO must be notified within 15 minutes, and public/customer disclosure must be initiated within 72 hours.",
        "reference_doc": "Document 2, Section 4.1"
    },

    # --- CROSS-DOCUMENT / MULTI-HOP TEST CASES ---
    {
        "question": "Who is responsible for handling IT hardware asset returns when an employee leaves the company, and what is their contact email?",
        "answer": "Kevin Zhang (HR Systems Admin / IT Systems Admin), email: kevin.zhang@apexglobal.tech.",
        "reference_doc": "Document 1, Chapter 7.2 & Document 2, Quick Contact Reference"
    },
    {
        "question": "Who must approve business travel expenses, and who approves overtime work for non-exempt employees?",
        "answer": "Marcus Vance (CFO) must approve both business travel and overtime work in writing.",
        "reference_doc": "Document 1, Chapter 2.3 & Chapter 6.3"
    },
    {
        "question": "Who should an employee contact if they experience workplace misconduct, and who investigates it?",
        "answer": "Report to Sarah Jenkins (VP of HR) or via ethics@apexglobal.tech. The investigation is conducted by a committee consisting of Sarah Jenkins and Dr. Aris Thorne (CEO).",
        "reference_doc": "Document 1, Chapter 1.3 & Chapter 4.1"
    },
    {
        "question": "Can I use public AI tools like free ChatGPT for company code or customer data?",
        "answer": "No. Employees are strictly forbidden from pasting Tier 3 or Tier 4 data, including source code and customer records, into public LLM tools.",
        "reference_doc": "Document 2, Section 3.2"
    }
]

JUDGE_MODEL_NAME = "openai/gpt-oss-20b"

# making llm as a judge

def _get_judge_llm() -> ChatOpenAI:
    """Return a judge model routed through Portkey, same slug as the main app."""
    headers = createHeaders(api_key=config.PORTKEY_API_KEY, 
             provider=JUDGE_PROVIDER)
    return ChatOpenAI(
        api_key=config.PORTKEY_API_KEY, 
        base_url=PORTKEY_GATEWAY_URL, 
        default_headers=headers, 
        model=JUDGE_MODEL_NAME)

# if dataset is there reuse it , if not create a new dataset 
# question paper 
def _ensure_dataset(client: Client):
    """Create the LangSmith dataset if it doesn't exist yet, and upload the test cases."""
    if client.has_dataset(dataset_name=DATASET_NAME):
        logger.info("Dataset '%s' already exists, reusing it", DATASET_NAME)
        return client.read_dataset(dataset_name=DATASET_NAME)

    logger.info("Creating dataset '%s' with %d example(s)", 
        DATASET_NAME, len(TEST_CASES))
    dataset = client.create_dataset(dataset_name=DATASET_NAME)
    client.create_examples(
        dataset_id=dataset.id,
        examples=[
            {"inputs": {"question": case["question"]},
            "outputs": {"answer": case["answer"]}}
            for case in TEST_CASES
        ],
    )
    return dataset

# start the exam
def run_evaluation():
    """Upload the dataset (if needed) 
    and run the correctness evaluation."""
    client = Client()
    dataset = _ensure_dataset(client)

    # Built once and reused for every test case, instead of rebuilding
    # the whole agent (and reconnecting to Qdrant) 10 times over.
    agent = build_hr_assistant()
    retriever = get_retriever(load_vector_store())

    # write the answers 
    def target(inputs: dict) -> dict:
        """
        Run one test question through the real agent, 
        and also capture
        the retrieved chunks 
        so groundedness can check the answer against
        what was actually retrieved 
        (not just the reference answer).
        """
        answer = ask(agent, inputs["question"])
        chunks = retriever.invoke(inputs["question"])
        context = "\n\n".join(chunk.page_content for chunk in chunks)
        return {"answer": answer, "context": context}

    # giving marks 
    correctness_evaluator = create_llm_as_judge(
        prompt=CORRECTNESS_PROMPT,
        feedback_key="correctness",
        judge=_get_judge_llm(),
    )

    groundedness_judge = create_llm_as_judge(
        prompt=RAG_GROUNDEDNESS_PROMPT,
        feedback_key="groundedness",
        judge=_get_judge_llm(),
    )

    def groundedness_evaluator(outputs: dict, **kwargs) -> dict:
        """Check the answer is supported by the retrieved context, not invented."""
        return groundedness_judge(outputs={"answer": outputs["answer"]}, context=outputs["context"])

    logger.info("Running evaluation against dataset '%s'", DATASET_NAME)
    return client.evaluate(
        target,
        data=dataset.name,
        evaluators=[correctness_evaluator,
                groundedness_evaluator],
        experiment_prefix="hr-policy-evalzz",
        description="HR policy assistant correctness + groundedness evaluation",
    )


