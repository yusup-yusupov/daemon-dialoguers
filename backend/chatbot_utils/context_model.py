from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import time
import hashlib
import json
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
import numpy as np
## Setting environment variables for OpenAI

MODEL_ID = 'sentence-transformers/all-mpnet-base-v2'
model_kwargs = {'device': 'cuda'}


# Read a json file C:\Users\vishw\OneDrive\Desktop\Projects\daemon-dialoguers\openAI_api.json
with open('https://firebasestorage.googleapis.com/v0/b/daemon-dialoguers.appspot.com/o/openAI_api.json?alt=media&token=374a5458-393b-4d2b-8ec9-78bd99730586') as f:
    key = json.load(f)

##### AUTOMATED CONTEXT SEARCH FUNCTIONS #####

def convert_logs_to_embeddings(file_path,chroma_path='./chromadb'):

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
    loader = TextLoader(file_path, encoding='latin1')
    documents = loader.load()

    # Load the text splitter with small chunk size
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    # Creating the Chroma vector store
    Chroma.from_documents(docs, persist_directory=f'{chroma_path}/{md5hash}_small', embedding=embedding)

    # Add a time delay of 30 seconds
    time.sleep(30)

    # Load the text splitter with small chunk size
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    # Creating the Chroma vector store
    Chroma.from_documents(docs, persist_directory=f'{chroma_path}/{md5hash}_large', embedding=embedding, )

def get_context(query, datastore, k=15):
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
    relevant_documents = vector_db.similarity_search_with_relevance_scores(query, k=k)
    documents, scores = [i for i,j in relevant_documents], [j for i,j in relevant_documents]

    # Return the relevant documents vectorstore
    return documents, scores


##### MANUAL TEXT PARSING FUNCTIONS #####
def parse_log_file(file_path):
    """
    Parses the log files and groups by daemons in a dataframe

    Parameters
    ----------
    file_path : str
        Path to the log file

    Returns
    -------
    df : pandas dataframe
        Dataframe with the log messages and the daemons
    """
    parsed_logs = []

    with open(file_path, 'r') as file:
        for line in file:
            parsed_line =line.strip() # Strip to remove newline characters
            parsed_logs.append(parsed_line)

    log_level_pattern = r' (info|error|warning)(:|\s|$)'
    daemon_pattern = r'([A-Za-z]* \d{2} \d{2}:\d{2}:\d{2} [^\s]+) ([^\[\]:]*)'
    remove_numerical_pattern = r"(?:0x[a-fA-F0-9]+)|(?<![a-zA-Z])[^a-zA-Z\s]+(?![a-zA-Z])"
    daemons_indexes = []
    log_levels = []
    remove_numerical = []
    for entry in parsed_logs:
        level_search = re.search(log_level_pattern, entry.lower())
        if level_search:
            log_levels.append(level_search.group().strip().strip(':'))
        else:
            log_levels.append(None)
        deamon_search = re.search(daemon_pattern, entry)
        if deamon_search:
            daemons_indexes.append(deamon_search.group(2).strip())
        else:
            daemons_indexes.append(None)
        entry = re.sub(remove_numerical_pattern, '', re.sub(daemon_pattern, "", entry))
        remove_numerical.append(entry)

    df_encoded = pd.DataFrame({'log_level': log_levels, 'log_message': parsed_logs, 'daemons':daemons_indexes, 'removed_numerical':remove_numerical})
    df_encoded = pd.get_dummies(df_encoded, columns=['log_level'])# 'time_of_day'
    df_encoded = df_encoded.iloc[:-3]*1

    return df_encoded


def fetch_daemons(df_encoded, daemon):
    # get outliers
    vectorizer = CountVectorizer(stop_words='english', strip_accents='ascii')
    tmp = df_encoded[df_encoded['daemons'] == daemon][['removed_numerical']]
    vectorizer.fit(tmp)
    log_message_vectors = []
    for index, row in tmp.iterrows():
        log_message_vectors.append(vectorizer.transform([row['removed_numerical']]))

    dense_vectors = [chunk.toarray() for chunk in log_message_vectors]
    reshaped_vectors = [chunk.reshape(-1) if chunk.ndim > 1 else chunk for chunk in dense_vectors]

    #df_text_features = pd.DataFrame(log_message_vectors.toarray(), columns=vectorizer.get_feature_names_out())
    tfidf_transformer = TfidfTransformer()
    tfidf_matrix = tfidf_transformer.fit_transform(reshaped_vectors)
    # Assuming tfidf_matrix and tmp are already defined

    uniqueness_scores = np.sum(tfidf_matrix, axis=1)
    sorted_indices = np.argsort(uniqueness_scores.A.ravel())

    # Ensure we don't access more documents than available
    num_documents = tfidf_matrix.shape[0]
    top_n = 4  # Number of top unique documents to display
    top_n = min(top_n, num_documents)  # Adjust top_n if it exceeds the number of documents
    #sorted_indices = sorted_indices[::-1]
    # Display the indices of the top unique documents
    text = []
    for i in sorted_indices[-top_n:]:
        # Additional check to ensure index is within bounds
        if i < len(tmp):
            text.append(tmp.iloc[i][0].strip())
        else:
            print(f"Index {i} is out of bounds.")
    return ";".join(text)


def overall_outliers(df_encoded):

    vectorizer = CountVectorizer(stop_words='english', strip_accents='ascii')
    tmp = df_encoded[['removed_numerical']]
    vectorizer.fit(tmp)
    log_message_vectors = []
    for index, row in tmp.iterrows():
        log_message_vectors.append(vectorizer.transform([row['removed_numerical']]))

    dense_vectors = [chunk.toarray() for chunk in log_message_vectors]
    reshaped_vectors = [chunk.reshape(-1) if chunk.ndim > 1 else chunk for chunk in dense_vectors]

    #df_text_features = pd.DataFrame(log_message_vectors.toarray(), columns=vectorizer.get_feature_names_out())
    tfidf_transformer = TfidfTransformer()
    tfidf_matrix = tfidf_transformer.fit_transform(reshaped_vectors)
    # Assuming tfidf_matrix and tmp are already defined

    uniqueness_scores = np.sum(tfidf_matrix, axis=1)
    uniqueness_scores = np.sum(tfidf_matrix, axis=1)
    sorted_indices = np.argsort(uniqueness_scores.A.ravel())

    # Ensure we don't access more documents than available
    num_documents = tfidf_matrix.shape[0]
    top_n = 10  # Number of top unique documents to display
    top_n = min(top_n, num_documents)  # Adjust top_n if it exceeds the number of documents
    text = []
    for i in sorted_indices[-top_n:]:
        # Additional check to ensure index is within bounds
        if i < len(tmp):
            text.append(tmp.iloc[i][0].strip())
        else:
            print(f"Index {i} is out of bounds.")

    return ";".join(text)


def get_anomalies(file_path):

    # Getting the encoded dataframe
    df_encoded = parse_log_file(file_path)
    gk = df_encoded.groupby('daemons')

    # Top 5 errors
    errors = gk["log_level_error"].sum().sort_values(ascending=False)[:10]
    warnings = gk["log_level_warning"].sum().sort_values(ascending=False)[:10]

    ## Collecting the errors and warnings
    allWarnings = []
    allErrors = []
    for i in range(len(warnings)):
        index_to_use = warnings.index[-(i + 1)]
        allWarnings.append(fetch_daemons(df_encoded,index_to_use))

    for i in range(len(errors)):
        index_to_use = errors.index[-(i + 1)]
        allErrors.append((fetch_daemons(df_encoded,index_to_use)))

    outliers = overall_outliers(df_encoded)

    return np.array(allWarnings).reshape(-1), np.array(allErrors).reshape(-1), np.array(outliers).reshape(-1)
