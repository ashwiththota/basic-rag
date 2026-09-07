"""One-off script to delete the old LangSmith dataset."""

from hr_assistant import config  # this triggers load_dotenv() as a side effect

from langsmith import Client

client = Client()
client.delete_dataset(dataset_name="hr-policy-qna")
print("Deleted.")