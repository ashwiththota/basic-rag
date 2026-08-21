""" 
STEP 4 : store chunk for similarity search 
and this is the last stage of our data ingestion pipline

"""

import os

from langchain_community.vectorstores import FAISS

from hr_assistant import config
from hr_assistant.embeedings import get_embeddings_model


# building a vectore store

def build_vector_store(chunks):
    """
    EMBED EVERY CHUNK AND BUILD 
    A SEARCHABLE FAISS INDEX IN MEMORY.
    """
    embedding_model   = get_embeddings_model()

    return FAISS.from_documents(chunks , embedding_model)


 # from_documents is the method for converting chunks into embedding model


#   save the vector store 
def save_vector_store(vector_store , path : str = config.VECTOR_STORE_PATH)->None:
    """Save the FAISS index to disk
    so we can't have to rebuild it everytime.
    
    """

    vector_store.save_local(path)

def load_vector_store(path : str = config.VECTOR_STORE_PATH):
    """Load a previously saved FAISS index from disk."""
    
    embeddings_model = get_embeddings_model()
    return FAISS.load_local(
        path , 
        embeddings_model , 
        allow_dangerous_deserialization = True
        )



def vector_store_exists(path : str = config.VECTOR_STORE_PATH) -> bool :
    """ Check if a saved FAISS index already exists on disk."""
    return os.path.exists(os.path.join(path , "index.faiss"))



def get_retriever(vector_store, k: int = config.TOP_K_RESULTS):
    """Turn a vector store into a retriever
    that returns the top-k matching chunks."""

    return vector_store.as_retriever(search_kwargs={"k": k})