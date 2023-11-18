from flask import Flask, jsonify, request

app = Flask(__name__)


@app.route('/api/get-answer', methods=['POST'])
def get_answer():
    data = request.json
    message = data.get('message', '')
    # Here, we simply echo back the received message. Modify as needed.
    answer = f"Your message-length was: {len(message)}"
    return jsonify({'answer': answer})


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)  # Set host to '0.0.0.0'
