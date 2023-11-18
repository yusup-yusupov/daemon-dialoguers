from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain, RetrievalQAWithSourcesChain
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

with open('C:/Users/vishw/OneDrive/Desktop/Projects/daemon-dialoguers/openAI_api.json') as f:
    key = json.load(f)

os.environ["OPENAI_API_KEY"] = key['openai_api_key']


def chat_with_bot(question, log_file_hash, chat_id, memory=False):

    ## Creating the embedding object
    embedding = OpenAIEmbeddings(
                openai_organization=key['openai_organization'],
                openai_api_key = key['openai_api_key'],
                model="text-embedding-ada-002",
                max_retries=10,
                )

    if not memory:
        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, input_key='question', output_key='answer')
        ## Save memory as pickle file
        with open(f'./memory/{chat_id}_memory.pkl', 'wb') as f:
            pickle.dump(chat_id, f)
    else:
        with open(f'./memory/{chat_id}_memory.pkl', 'rb') as f:
            memory = pickle.load(f)

    ## Fetching the context
    docs_small,conf_small = cm.get_context(question, f"./chromadb/{log_file_hash}_small")
    docs_large,conf_large = cm.get_context(question, f"./chromadb/{log_file_hash}_large")

    ## Add both large and small to a single list and sort them by confidence
    docs = docs_small + docs_large
    conf = conf_small + conf_large

    ## Creating the context vectorstore
    df = pd.DataFrame({'docs':docs, 'conf':conf})
    df.sort_values(by='conf', ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    vectorstore_relevant = Chroma.from_documents(list(df['docs'].values), embedding=embedding)
    
    ## Creating the chatbot
    llm = ChatOpenAI(
        openai_organization=key['openai_organization'],
        model="gpt-4",
        max_tokens=500,
        max_retries=10,
        )

    ## Creating the retriever
    retriver = vectorstore_relevant.as_retriever(search_kwargs={"k": 5,"score_threshold": .5})
    chat = RetrievalQAWithSourcesChain.from_llm(llm, 
                                                retriever=retriver, 
                                                memory=memory, 
                                                return_source_documents=True,
                                                max_tokens_limit=500, 
                                                reduce_k_below_max_tokens=True)

    result = chat({"question": question})

    ## Saving the memory
    with open(f'./memory/{chat_id}_memory.pkl', 'wb') as f:
        pickle.dump(memory, f)

    return {'answer': result['answer'], 'Source':[i.page_content for i in df['docs'].values[:10]], 'Confidence': [i for i in df['conf'].values[:10]]}


def summarize_log(log_file_hash):

    # Making a prompt template
    prompt_template = """Write a concise summary of the following. Make them bullet points:


    {text}


    Important points:"""

    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

    # Getting the important docs
    question = 'What are errors and important processes in the log?'
    docs,_ = cm.get_context(question, f"./chromadb/{log_file_hash}_small", k = 50)

    # Getting the GPT model
    llm = ChatOpenAI(
        openai_organization=key['openai_organization'],
        model="gpt-4"
        )

    chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
    summary = chain.run(docs)

    return summary


