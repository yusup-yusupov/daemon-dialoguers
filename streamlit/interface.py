import sys
sys.path.append('../backend/chatbot_utils')
from chatbot_utils import chat_with_bot, summarize_log, find_log_anomalies
from context_model import convert_logs_to_embeddings
import streamlit as st
import hashlib

CHROMA_DB_PATH = 'C:/Users/vishw/OneDrive/Desktop/Projects/daemon-dialoguers/backend/chatbot_utils/chromadb'
MEMORY_PATH = 'C:/Users/vishw/OneDrive/Desktop/Projects/daemon-dialoguers/backend/chatbot_utils/memory'
DATA_DIR = 'C:/Users/vishw/OneDrive/Desktop/Projects/daemon-dialoguers/Dataset'

hash_file_dict = { f"{DATA_DIR}/test_log2.out": hashlib.md5(open(f"{DATA_DIR}/test_log2.out",'rb').read()).hexdigest(),
                   f"{DATA_DIR}/test_log1.out": hashlib.md5(open(f"{DATA_DIR}/test_log1.out",'rb').read()).hexdigest(),
                   f"{DATA_DIR}/final_log.out": hashlib.md5(open(f"{DATA_DIR}/final_log.out",'rb').read()).hexdigest(),
}



option = st.sidebar.selectbox('Select Option', ('Upload Log','Chat with Bot','Summarize Log'))

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
        response = chat_with_bot(question, file_hash, chat_id, memory=memory, chroma_path=CHROMA_DB_PATH, memory_path=MEMORY_PATH)
        st.write(f"Answer: {response['answer']}")
        st.write(f"Source: {response['source']}")
        st.write(f"Confidence: {response['confidence']}")

