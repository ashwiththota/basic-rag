""" STEP 2 : chop the document data into chunks and define chunk size and chunk overlap size"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from hr_assistant import config 


def split_into_chunks(documents):
    """
    Split document into small overlapping chunks.
    
    """
    text_splitters = RecursiveCharacterTextSplitter(
            chunk_size = config.CHUNK_SIZE,
            chunk_overlap = config.CHUNK_OVERLAP
    )
    return text_splitters.split_documents(documents)