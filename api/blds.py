from flask import Flask, jsonify, request
from flask_cors import CORS
import logging

from app.services.bert_model import BertModelSingleton
from infra.make_response import make_response

# 配置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # 启用 CORS 支持
model = BertModelSingleton()

@app.route('/')
def home():
    return make_response("Hello, World!")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        logger.debug(f"Received request data: {request.data}")
        input_text = request.json
        logger.debug(f"Parsed JSON: {input_text}")
        
        if not input_text:
            logger.error("No input text provided")
            return make_response(None, status=False, message="No input text provided")
            
        result = model.classify(input_text)
        logger.debug(f"Model classification result: {result}")
        return make_response(result, status=True, message="Success")
    except Exception as e:
        logger.error(f"Error in predict endpoint: {str(e)}", exc_info=True)
        return make_response(None, status=False, message=str(e))

@app.route('/api/data', methods=['GET', 'POST'])
def handle_data():
    return make_response("This is a GET request")

@app.route('/api/exc', methods=['GET', 'POST'])
def handle_data1():
    raise Exception("This is a GET request")
    # return make_response("This is a GET request")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)  # 使用 0.0.0.0 允许所有网络接口访问