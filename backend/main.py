from flask import Flask, jsonify, request

app = Flask(__name__)


@app.route('/api/get-answer', methods=['POST'])
def get_answer():
    data = request.json
    message = data.get('message', '')
    answer = f"Your message-length was: {len(message)}"
    return jsonify({'answer': answer})


if __name__ == '__main__':
    app.run(threaded=True, port=5000)
