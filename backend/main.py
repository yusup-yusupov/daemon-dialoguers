from flask import Flask, jsonify, request
import sys
sys.path.append('./chatbot_utils')
from chatbot_utils import chat_with_bot, summarize_log, find_log_anomalies
from context_model import convert_logs_to_embeddings
app = Flask(__name__)


@app.route('/api/get-answer', methods=['POST'])
def get_answer():
    data = request.json
    message = data.get('message', '')
    answer = f"Your message-length was: {len(message)}"
    return jsonify({'answer': answer})

@app.route('/api/embedd-logs', methods=['POST'])
def embedd_logs():
    data = request.json
    file_path = data.get('file_path', '') #Path to the log file
    answer = convert_logs_to_embeddings(file_path)
    return jsonify({'status': 'Success'})

@app.route('/api/get_summary', methods=['POST'])
def get_summary():
    # data = request.json
    return jsonify({'body': 'this is a summary'})

    
@app.route('/api/chat-bot', methods=['POST'])
def get_summary():
    data = request.json
    ## Getting the data
    question = data.get('question', '') #Question to ask the bot
    log_file_hash = data.get('log_file_hash', '') #MD5 hash of the log file
    chat_id = data.get('chat_id', '') #ID of the chat
    memory = data.get('memory', '') #TRUE/FALSE

    ## Getting the response from the bot
    json_response = chat_with_bot(question, log_file_hash, chat_id, memory)
    return jsonify(json_response)

if __name__ == '__main__':
    app.run(threaded=True, port=5000)
