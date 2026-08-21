
"""  STEP 3: turn text into numbers (vectors) using jina"""

from langchain_community.embeddings import JinaEmbeddings

from hr_assistant import config
# from hr_assistant import splitter


def get_embeddings_model():
    """
    return a Jina embeddings model.
    Reads JINA_API_KEY from the environment.
    
    """

    return JinaEmbeddings( model_name = config.embeddings_model)