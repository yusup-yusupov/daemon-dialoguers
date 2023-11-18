from flask import Flask, jsonify, request
import sys
sys.path.append('./chatbot_utils')
from chatbot_utils import chat_with_bot
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
    file_path = data.get('file_path', '')
    answer = convert_logs_to_embeddings(file_path)
    return jsonify({'status': 'Success'})

@app.route('/api/get_summary', methods=['POST'])
def get_summary():
    # data = request.json
    return jsonify({'body': 'this is a summary'})

if __name__ == '__main__':
    app.run(threaded=True, port=5000)
