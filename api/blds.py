from flask import Flask, jsonify, request

from app.services.bert_model import BertModelSingleton
from infra.make_response import make_response
app = Flask(__name__)
model = BertModelSingleton()
@app.route('/')
def home():
    return make_response("Hello, World!")
@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.json
    return make_response(model.classify(input_text))

@app.route('/api/data', methods=['GET', 'POST'])
def handle_data():
    return make_response("This is a GET request")

@app.route('/api/exc', methods=['GET', 'POST'])
def handle_data1():
    raise Exception("This is a GET request")
    # return make_response("This is a GET request")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)