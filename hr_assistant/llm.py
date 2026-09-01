""" STEP 6 : connect to the LLM (the "brain" of the assistant)"""

from langchain_groq import ChatGroq

from hr_assistant import config
from hr_assistant.logger import get_logger

logger = get_logger(__name__)

def get_llm():
    """ Return a Groq model. Reads GROQ_API_KEYS from the environment."""
    logger.info("Intiliazing the LLM '%s'" , config.LLM_MODEL_NAME)
    return ChatGroq(model=config.LLM_MODEL_NAME,temperature=0)
