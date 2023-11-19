from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.document_loaders.telegram import text_to_docs
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain, RetrievalQAWithSourcesChain, ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain

import textwrap
from time import monotonic

import context_model as cm
import os

import pickle
import pandas as pd
import json

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

with open('C:/Users/vishw/OneDrive/Desktop/Projects/daemon-dialoguers/openAI_api.json') as f:
    key = json.load(f)

os.environ["OPENAI_API_KEY"] = key['openai_api_key']


def chat_with_bot(question, log_file_hash, chat_id, memory=False, chroma_path='./chromadb', memory_path='./memory'):
    '''
    This function is used to chat with the bot. It takes in the question, log_file_hash, chat_id and memory as input and returns the answer, source and confidence as output.

    Parameters
    ----------
    question : str
        The question to ask the bot.
    log_file_hash : str
        The MD5 hash of the log file.
    chat_id : str
        The ID of the chat.
    memory : bool, optional
        Whether to use the memory or not. The default is False.

    Returns
    -------
    dict
        The answer, source and confidence.
    '''
    print("--Chatbot Started--")

    ## Creating the embedding object
    embedding = OpenAIEmbeddings(
                openai_organization=key['openai_organization'],
                openai_api_key = key['openai_api_key'],
                model="text-embedding-ada-002",
                max_retries=10,
                )


    print("--Embedding Created--")
    if not memory:
        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, input_key='question', output_key='answer')
        ## Save memory as pickle file
        with open(f'{memory_path}/{chat_id}_memory.pkl', 'wb') as f:
            pickle.dump(memory, f)
    else:
        with open(f'{memory_path}/{chat_id}_memory.pkl', 'rb') as f:
            memory = pickle.load(f)

    print("--Memory Created--")

    ## Fetching the context
    docs_small,conf_small = cm.get_context(question, f"{chroma_path}/{log_file_hash}_small")
    docs_large,conf_large = cm.get_context(question, f"{chroma_path}/{log_file_hash}_large")

    ## Add both large and small to a single list and sort them by confidence
    docs = docs_small + docs_large
    conf = conf_small + conf_large

    print("--Context fetched--")

    ## Creating the context vectorstore
    df = pd.DataFrame({'docs':docs, 'conf':conf})
    df.sort_values(by='conf', ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    vectorstore_relevant = Chroma.from_documents(list(df['docs'].values), embedding=embedding)

    print("--Vectorstore created--")
    
    ## Creating the chatbot
    llm = ChatOpenAI(
        openai_organization=key['openai_organization'],
        model="gpt-4",
        max_tokens=500,
        max_retries=10,
        )
    ## Creating the retriever
    retriver = vectorstore_relevant.as_retriever(search_kwargs={"k": 5,"score_threshold": .5})
    chat = ConversationalRetrievalChain.from_llm(llm, 
                                                retriever=retriver, 
                                                memory=memory, 
                                                return_source_documents=True,
                                                max_tokens_limit=500)

    result = chat({"question": question})

    print("--Chatbot Executed--")
    ## Saving the memory
    with open(f'{memory_path}/{chat_id}_memory.pkl', 'wb') as f:
        pickle.dump(memory, f)

    return {'answer': result['answer'], 'Source':[i.page_content for i in df['docs'].values[:10]], 'Confidence': [i for i in df['conf'].values[:10]]}


def summarize_log(log_file_hash, log_file_path, chroma_path='./chromadb'):
    '''
    Summarizes the log file. Returns a paragraph and some key points.

    Parameters
    ----------
    log_file_hash : str
        The MD5 hash of the log file.
        
    Returns
    -------
    str
        The summary of the log file.
    '''
    # Getting the GPT model
    llm = ChatOpenAI(
        openai_organization=key['openai_organization'],
        model="gpt-4"
        )

    ### GETTING GENERAL LOG ###
    print("--Getting General Log--")
    # Making a prompt template
    prompt_template = """Write a concise summary of the following. Start with a small paragraph and then mention some key events as bullet points:


    {text}

    Paragraph:

    Important points:"""

    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

    # Getting the important docs
    question = 'What are important processes in the log?'
    docs,_ = cm.get_context(question, f"{chroma_path}/{log_file_hash}_small", k = 50)
    chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
    general_summary = chain.run(docs)

    print("--General Summary Created--")

    ### GETTING DEAMON SPECIFIC LOG ###
    print("--Getting Daemon Specific Log--")
    prompt_template = """Give a brief summary and mention the key points in form of bullet points :


    {text}
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

    warnings, errors, outliers = cm.get_anomalies(log_file_path)
    warnings, errors, outliers = text_to_docs(warnings), text_to_docs(errors), text_to_docs(outliers)

    if len(warnings) > 0:
        chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
        warnings_summary = chain.run(warnings)

    if len(errors) > 0:
        chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
        errors_summary = chain.run(errors)

    if len(outliers) > 0:
        chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
        outliers_summary = chain.run(outliers)

    print("--Daemon Specific Summary Created--")

    summary = "***LOG SUMMARY***\n\n" + general_summary + "\n\n***WARNINGS***\n\n" + warnings_summary + "\n\n***ERRORS***\n\n" + errors_summary + "\n\n***ANOMALIES***\n\n" + outliers_summary + "\n\n***END OF SUMMARY***"

    return summary

def edit_summary(edit_prompt, summary_text, chat_id, memory=False, memory_path='./memory'):
    '''
    Edits the summary of the log file. Returns a paragraph and some key points.

    Parameters
    ----------
    edit_prompt : str
        The prompt to edit the summary.
    summary_text : str
        The summary of the log file.
    chat_id : str
        The ID of the chat.
    memory : bool, optional
        Whether to use the memory or not. The default is False.
        
    Returns
    -------
    str
        The summary of the log file.
    '''
    ## Creating the summary doc
    summary_doc = text_to_docs([summary_text])
    ## Creating the embedding object
    embedding = OpenAIEmbeddings(
                openai_organization=key['openai_organization'],
                openai_api_key = key['openai_api_key'],
                model="text-embedding-ada-002",
                max_retries=10,
                )
    vectorstore_relevant = Chroma.from_documents(summary_doc, embedding=embedding)

    ## Creating the chatbot
    llm = ChatOpenAI(
        openai_organization=key['openai_organization'],
        model="gpt-4",
        max_tokens=2000,
        max_retries=10,
        )

    ## Creating the retriever
    retriver = vectorstore_relevant.as_retriever(search_kwargs={"k": 5,"score_threshold": .5})

    if not memory:
        memory = ConversationBufferMemory()
        ## Save memory as pickle file
        with open(f'{memory_path}/{chat_id}_editmemory.pkl', 'wb') as f:
            pickle.dump(memory, f)

        chat = ConversationChain(llm=llm, memory=memory)
        result = chat.predict(input=f"'{summary_text}'\n\n {edit_prompt}")

        ## Saving the memory
        with open(f'{memory_path}/{chat_id}_editmemory.pkl', 'wb') as f:
            pickle.dump(memory, f)

        return result

    else:
        with open(f'{memory_path}/{chat_id}_editmemory.pkl', 'rb') as f:
            memory = pickle.load(f)
        chat = ConversationChain(llm=llm, memory=memory)
        result = chat.predict(input=edit_prompt)

        ## Saving the memory
        with open(f'{memory_path}/{chat_id}_editmemory.pkl', 'wb') as f:
            pickle.dump(memory, f)

        return result

        

def find_log_anomalies(log_file_hash):
    '''
    Returns a summary of the anomalies in the log file in the form of bullet points.

    Parameters
    ----------
    log_file_hash : str
        The MD5 hash of the log file.

    Returns
    -------
    str
        The summary of the anomalies in the log file.

    '''
    # Getting the db
    embedding = OpenAIEmbeddings(
                    openai_organization=key['openai_organization'],
                    openai_api_key = key['openai_api_key'],
                    model="text-embedding-ada-002",
                    max_retries=10,
                    )
    vector_db = Chroma(persist_directory=f"./chromadb/{log_file_hash}_small", embedding_function=embedding)

    # Reducing the dimensionality of the embeddings
    embeddings = np.array(vector_db.get(include=['embeddings'])['embeddings'])
    pca = PCA(n_components=10)
    pca.fit(embeddings)
    pca_embeddings = pca.transform(embeddings)

    # Cluster the embeddings
    dbscan = DBSCAN(eps=0.1, min_samples=3)
    dbscan.fit(pca_embeddings)

    # Get all the core points and outliers
    core_points = np.where(dbscan.core_sample_indices_)[0]
    outliers = np.where(dbscan.labels_ == -1)[0]
    outlier_behaviour = list(np.array(vector_db.get(include=['documents'])['documents'])[outliers][:80])
    outlier_docs = text_to_docs(outlier_behaviour)

    # Summarizing the anomalies
    prompt_template = """Write these annomalies in form of bullet points:


    {text}

    Anomalies:"""

    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
    llm = ChatOpenAI(
    openai_organization=key['openai_organization'],
    model="gpt-4",
    )
    chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
    summary = chain.run(outlier_docs)

    return summary
