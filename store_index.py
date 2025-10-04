from src.helper import get_embeddings, load_documents, split_documents
from langchain.vectorstores import Chroma
import os





# url = "https://github.com/entbappy/End-to-end-Medical-Chatbot-Generative-AI"

# repo_ingestion(url)


documents = load_documents("./cloned_repo")
text_chunks = split_documents(documents)
embeddings = get_embeddings()



#storing vector in choramdb
vectordb = Chroma.from_documents(text_chunks, embedding=embeddings, persist_directory='./db')
vectordb.persist()