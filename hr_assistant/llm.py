""" STEP 6 : connect to the LLM (the "brain" of the assistant)"""

from hr_assistant.gateway import get_gateway_llm
from hr_assistant import config
from hr_assistant.logger import get_logger

logger = get_logger(__name__)

def get_llm():
    """ Return a Groq model. Reads GROQ_API_KEYS from the environment."""
    logger.info("Intiliazing the LLM via portkey")
    return get_gateway_llm()
