from flask import Flask, jsonify

from infra.make_response import make_response

app = Flask(__name__)

@app.route('/')
def home():
    return make_response("Hello, World!")

@app.route('/api/data', methods=['GET', 'POST'])
def handle_data():
    return make_response("This is a GET request")

@app.route('/api/exc', methods=['GET', 'POST'])
def handle_data1():
    raise Exception("This is a GET request")
    # return make_response("This is a GET request")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)