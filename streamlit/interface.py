import sys
sys.path.append('../backend/chatbot_utils')
from chatbot_utils import chat_with_bot, summarize_log, find_log_anomalies, edit_summary
from context_model import convert_logs_to_embeddings
import streamlit as st
import hashlib
from glob import glob
import shutil
import os

CHROMA_DB_PATH = 'C:/Users/vishw/OneDrive/Desktop/Projects/daemon-dialoguers/backend/chatbot_utils/chromadb'
MEMORY_PATH = 'C:/Users/vishw/OneDrive/Desktop/Projects/daemon-dialoguers/backend/chatbot_utils/memory'
DATA_DIR = 'C:/Users/vishw/OneDrive/Desktop/Projects/daemon-dialoguers/Dataset'

hash_file_dict = {}
for f in glob(f"{DATA_DIR}/*.out"):
    hash_file_dict[f] = hashlib.md5(open(f,'rb').read()).hexdigest()



option = st.sidebar.selectbox('Select Option', ('Upload Log','Chat with Bot','Summarize Log','Edit Summary'))

summary = None
results = None

if option == 'Upload Log':
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type,"FileSize":uploaded_file.size}
        st.write(file_details)
        with open(f'./{uploaded_file.name}','wb') as f:
            f.write(uploaded_file.getbuffer())
        st.write('File Uploaded')

        ## Add to hash_file_dict
        hash_file_dict[f"./{uploaded_file.name}"] = hashlib.md5(open(f"./{uploaded_file.name}",'rb').read()).hexdigest()
        
        st.write('Embedding the log file')
        convert_logs_to_embeddings(f'./{uploaded_file.name}', chroma_path=CHROMA_DB_PATH)
        st.write('File Embedded')
        ## Move the file to the Dataset folder
        shutil.move(f"./{uploaded_file.name}", f"{DATA_DIR}/{uploaded_file.name}")

if option == 'Chat with Bot':
    ## Dropdown for selecting the log file
    log_file = st.selectbox('Select Log File', list(hash_file_dict.keys()))
    file_hash = hash_file_dict[log_file]
    ## Input box for the question
    question = st.text_input('Enter your question')
    ## Checkbox for memory
    memory = st.checkbox('Use Memory')
    ## Chat ID
    chat_id = st.text_input('Enter Chat ID')
    ## Chat with bot
    if st.button('Chat'):
        print(f"Started Chatting: {file_hash}")
        response = chat_with_bot(question, file_hash, chat_id, memory=memory, chroma_path=CHROMA_DB_PATH, memory_path=MEMORY_PATH)

        ## Show response['Answer'] as a markdown
        st.subheader('Answer')
        st.markdown(response['answer'])

        ## Show response['Source'] and response['Confidence'] as two columns of a table
        st.subheader('Source and Confidence')
        st.write(response['Source'])
        st.write(response['Confidence'])

if option == 'Summarize Log':
    st.subheader('Summarize Log')
    ## Dropdown for selecting the log file
    log_file = st.selectbox('Select Log File', list(hash_file_dict.keys()))
    file_hash = hash_file_dict[log_file]
    ## Summarize the log
    if st.button('Summarize'):
        print(f"Started Summarizing: {file_hash}")
        summary = summarize_log(file_hash, log_file, chroma_path=CHROMA_DB_PATH)
        st.subheader('Summary')
        st.markdown(summary)
        with open(f"{DATA_DIR}/{file_hash}.txt", 'w') as f:
            f.write(summary)


if option == 'Edit Summary':

    log_file = st.selectbox('Select Log File', list(hash_file_dict.keys()))
    file_hash = hash_file_dict[log_file]
    summary = None
    ## Load the summary if it exists
    if os.path.exists(f"{DATA_DIR}/{file_hash}.txt"):
        with open(f"{DATA_DIR}/{file_hash}.txt", 'r') as f:
            summary = f.read()
        st.markdown(summary)
    else:
        st.subheader('No Summary Found')

    ## Begin edit summary
    st.subheader('Edit Summary')
    edit_prompt = st.text_input('Enter your edit prompt')
    memory = st.checkbox('Use Memory')
    chat_id = st.text_input('Enter Chat ID')
    
    if st.button('Edit'):
        with open(f"{DATA_DIR}/{file_hash}.txt", 'r') as f:
            summary = f.read()
        results = edit_summary(edit_prompt, summary, chat_id, memory=memory, memory_path=MEMORY_PATH)

    st.subheader('Edited')
    st.markdown(results)



    

