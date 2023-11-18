from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import time
import hashlib
import json
## Setting environment variables for OpenAI

MODEL_ID = 'sentence-transformers/all-mpnet-base-v2'
model_kwargs = {'device': 'cuda'}


# Read a json file C:\Users\vishw\OneDrive\Desktop\Projects\daemon-dialoguers\openAI_api.json
with open('C:/Users/vishw/OneDrive/Desktop/Projects/daemon-dialoguers/openAI_api.json') as f:
    key = json.load(f)


def convert_logs_to_embeddings(file_path):

    # Load the embedding

    # embedding = HuggingFaceEmbeddings(
    # model_name=MODEL_ID,
    # model_kwargs=model_kwargs
    # )

    embedding = OpenAIEmbeddings(
    openai_organization=key['openai_organization'],
    openai_api_key = key['openai_api_key'],
    model="text-embedding-ada-002",
    max_retries=10,
    )

    # Get the name of the file
    md5hash = hashlib.md5(open(file_path,'rb').read()).hexdigest()
    # Get the md5 hash of the file
    # Load the document loader
    loader = TextLoader(file_path)
    documents = loader.load()

    # Load the text splitter with small chunk size
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    # Creating the Chroma vector store
    Chroma.from_documents(docs, persist_directory=f'./chromadb/{md5hash}_small', embedding=embedding)

    # Add a time delay of 30 seconds
    time.sleep(30)

    # Load the text splitter with small chunk size
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    # Creating the Chroma vector store
    Chroma.from_documents(docs, persist_directory=f'./chromadb/{md5hash}_large', embedding=embedding)


def get_context(query, datastore):
    # Load the embedding
    # embedding = HuggingFaceEmbeddings(
    # model_name=MODEL_ID,
    # model_kwargs=model_kwargs
    # )

    embedding = OpenAIEmbeddings(
    openai_organization=key['openai_organization'],
    openai_api_key = key['openai_api_key'],
    model="text-embedding-ada-002",
    max_retries=10,
    )

    # Loading the datastore
    vector_db = Chroma(persist_directory=datastore, embedding_function=embedding)

    # Loading the relevant documents
    relevant_documents = vector_db.similarity_search_with_relevance_scores(query, k=15)
    documents, scores = [i for i,j in relevant_documents], [j for i,j in relevant_documents]

    # Return the relevant documents vectorstore
    return documents, scores
