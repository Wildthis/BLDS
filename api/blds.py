from flask import Flask, jsonify, request, Blueprint
from flask_cors import CORS
import logging

from app.services.bert_model import BertModelSingleton, BertModel
from app.services.bias_service import BiasService
from infra.make_response import make_response
from infra.exception_handler import handle_exception

# 配置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # 启用 CORS 支持
model = BertModelSingleton()

blds = Blueprint('blds', __name__)

@app.route('/')
def home():
    return make_response("Hello, World!")

@blds.route('/predict', methods=['POST'])
@handle_exception
def predict():
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return make_response(code=400, message='Text is required')
    
    result = model.predict(text)
    
    # 保存检测记录
    BiasService.save_detection_record(
        text_content=text,
        bias_type=result['bias_type'],
        is_biased=result['is_biased'],
        confidence=result['confidence']
    )
    
    return make_response(data=result)

@blds.route('/feedback', methods=['POST'])
@handle_exception
def submit_feedback():
    data = request.get_json()
    record_id = data.get('record_id')
    is_correct = data.get('is_correct')
    feedback_content = data.get('feedback_content')
    
    if not record_id or is_correct is None:
        return make_response(code=400, message='Record ID and feedback are required')
    
    BiasService.save_user_feedback(record_id, is_correct, feedback_content)
    return make_response(message='Feedback submitted successfully')

@blds.route('/history', methods=['GET'])
@handle_exception
def get_history():
    page = int(request.args.get('page', 1))
    page_size = int(request.args.get('page_size', 10))
    offset = (page - 1) * page_size
    
    history = BiasService.get_detection_history(limit=page_size, offset=offset)
    return make_response(data=history)

@blds.route('/stats', methods=['GET'])
@handle_exception
def get_stats():
    stats = BiasService.get_detection_stats()
    return make_response(data=stats)

@app.route('/api/data', methods=['GET', 'POST'])
def handle_data():
    return make_response("This is a GET request")

@app.route('/api/exc', methods=['GET', 'POST'])
def handle_data1():
    raise Exception("This is a GET request")
    # return make_response("This is a GET request")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)  # 使用 0.0.0.0 允许所有网络接口访问