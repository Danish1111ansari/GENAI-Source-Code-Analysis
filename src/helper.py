#importing the requirement
from git import Repo
from langchain.text_splitter import Language
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFacePipeline
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch



# Clone repository
def clone_repo(repo_url, local_path):
    if not os.path.exists(local_path):
        Repo.clone_from(repo_url, local_path)
    return local_path



# Load documents from source code
def load_documents(repo_path):
    loader = GenericLoader.from_filesystem(
        repo_path,
        glob="**/*",
        suffixes=[".py", ".java", ".js", ".ts", ".cpp", ".c", ".h", ".html", ".css"],
        parser=LanguageParser(language=Language.JAVA, parser_threshold=500)
    )
    documents = loader.load()
    return documents



# Split documents into chunks
def split_documents(documents):
    java_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.JAVA,
        chunk_size=800,
        chunk_overlap=80
    )
    texts = java_splitter.split_documents(documents)
    return texts


# Initialize Hugging Face embeddings (completely free, no token needed)
def get_embeddings():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return embeddings


# Initialize local LLM (completely free, no token needed)
def get_llm():
    # Using a very lightweight model that downloads automatically
    model_name = "microsoft/DialoGPT-small"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
     # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=150,  # Output tokens only
        temperature=0.3,
        top_p=0.85,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        truncation=True  # Important: truncate long inputs
    )
    
    llm = HuggingFacePipeline(
        pipeline=pipe,
        model_kwargs={"temperature": 0.3}
    )
    return llm



# Alternative: Use a more capable model
def get_better_llm():
    try:
        model_name = "microsoft/DialoGPT-medium"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=200,
            temperature=0.4,
            top_p=0.9,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            truncation=True
        )
        
        return HuggingFacePipeline(pipeline=pipe)
    except:
        return get_llm()  # Fallback to smaller model
    

# Create vector store (local database, no token needed)
def create_vector_store(texts, embeddings, persist_directory="./chroma_db"):
    vector_store = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    return vector_store



# Create conversation chain
def create_conversation_chain(vector_store, llm):
    memory = ConversationSummaryMemory(
        llm=llm,
        memory_key="chat_history",
        return_messages=True,
        max_token_limit=300  # Smaller memory
    )
    
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 2,  # Only get 2 most relevant documents
            "score_threshold": 0.5  # Minimum similarity score
        }
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        verbose=True,
        return_source_documents=True,
        max_tokens_limit=512,  # Limit total context
        chain_type="stuff"  # Use "stuff" for simpler processing
    )
    return chain


# Simple search function as fallback
def simple_code_search(vector_store, query):
    """Simple search without LLM for reliable results"""
    try:
        results = vector_store.similarity_search(query, k=2)
        if results:
            print(f"\n📁 Found {len(results)} relevant files:")
            for i, doc in enumerate(results, 1):
                source = doc.metadata.get('source', 'Unknown')
                filename = os.path.basename(source)
                print(f"\n{i}. File: {filename}")
                print(f"   Path: {source}")
                print(f"   Preview: {doc.page_content[:300]}...")
            return results
        else:
            print("No relevant code found for your query.")
            return []
    except Exception as e:
        print(f"Search error: {e}")
        return []