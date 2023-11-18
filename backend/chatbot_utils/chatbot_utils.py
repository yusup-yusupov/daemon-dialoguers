from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain, RetrievalQAWithSourcesChain
from langchain.memory import ConversationBufferMemory

from context_model import *
import os

import pickle
import pandas as pd
import json

with open('C:/Users/vishw/OneDrive/Desktop/Projects/daemon-dialoguers/openAI_api.json') as f:
    key = json.load(f)

os.environ["OPENAI_API_KEY"] = key['openai_api_key']


def chat_with_bot(question, log_file_hash, chat_id, memory=False):

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

    df = pd.DataFrame({'docs':docs, 'conf':conf})
    df.sort_values(by='conf', ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)