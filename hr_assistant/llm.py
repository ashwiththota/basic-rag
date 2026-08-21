""" STEP 6 : connect to the LLM (the "brain" of the assistant)"""

from langchain_groq import ChatGroq

from hr_assistant import config


def get_llm():
    """ Return a Groq model. Reads GROQ_API_KEYS from the environment."""
    return ChatGroq(model=config.LLM_MODEL_NAME,temperature=0)
