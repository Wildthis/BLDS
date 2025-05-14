from flask import Flask, jsonify, request, Blueprint
from flask_cors import CORS
import logging

from app.services.bert_model import BertModelSingleton
from app.services.bias_service import BiasService
from app.services.user_service import UserService
from infra.make_response import make_response
from infra.exception_handler import handle_exception
from infra.auth_middleware import login_required, admin_required

# 配置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# 创建 Flask 应用
app = Flask(__name__)
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:3000"],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True
    }
})  # 启用 CORS 支持，并配置允许的方法和头部

# 创建蓝图
blds = Blueprint('blds', __name__)

# 初始化模型
model = BertModelSingleton()

@blds.route('/auth/register', methods=['POST'])
@handle_exception
def register():
    """用户注册"""
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return make_response(code=400, message='用户名和密码不能为空')
    
    result = UserService.register(username, password)
    return make_response(data=result)

@blds.route('/auth/login', methods=['POST'])
@handle_exception
def login():
    """用户登录"""
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return make_response(code=400, message='用户名和密码不能为空')
    
    result = UserService.login(username, password)
    return make_response(data=result)

@blds.route('/auth/logout', methods=['POST'])
@handle_exception
def logout():
    """用户登出"""
    return make_response(message='登出成功')

@blds.route('/predict', methods=['POST'])
@login_required
@handle_exception
def predict():
    try:
        data = request.get_json()
        if not data:
            return make_response(code=400, message='No JSON data provided')
            
        text = data.get('text', '')
        if not text:
            return make_response(code=400, message='Text is required')
        
        # 获取偏见类型
        bias_type = model.classify(text)
        
        # 计算置信度（这里使用固定值，实际应该从模型输出中获取）
        confidence = 0.8 if bias_type != 'false' else 0.2
        
        # 保存检测记录
        record_id = BiasService.save_detection_record(
            text_content=text,
            bias_type=bias_type,
            is_biased=bias_type != 'false',
            confidence=confidence,
            user_id=request.user['user_id']
        )
        
        return make_response(data={
            'id': record_id,
            'bias_type': bias_type,
            'is_biased': bias_type != 'false',
            'confidence': confidence,
            'text': text
        })
    except Exception as e:
        logger.error(f"Error in predict: {str(e)}", exc_info=True)
        return make_response(code=500, message=str(e))

@blds.route('/feedback', methods=['POST'])
@login_required
@handle_exception
def submit_feedback():
    data = request.get_json()
    record_id = data.get('record_id')
    is_correct = data.get('is_correct')
    feedback_content = data.get('feedback_content')
    
    if not record_id or is_correct is None:
        return make_response(code=400, message='Record ID and feedback are required')
    
    BiasService.save_user_feedback(record_id, is_correct, feedback_content, request.user['user_id'])
    return make_response(code=0, message='Feedback submitted successfully')

@blds.route('/stats/charts', methods=['GET'])
@admin_required
@handle_exception
def get_chart_stats():
    """获取图表统计数据"""
    try:
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        # 获取偏见类型分布
        bias_stats = BiasService.get_bias_type_stats(start_date, end_date)
        
        # 获取偏见率趋势
        trend_data = BiasService.get_bias_rate_trend(start_date, end_date)
        
        return make_response(code=0, data={
            **bias_stats,
            'trend': trend_data
        })
    except Exception as e:
        logger.error(f"获取图表统计数据失败: {str(e)}")
        return make_response(code=500, message=str(e))

@blds.route('/feedback/list', methods=['GET'])
@admin_required
@handle_exception
def get_feedback_list():
    try:
        page = int(request.args.get('page', 1))
        page_size = int(request.args.get('page_size', 10))
        feedback_type = request.args.get('feedback_type', 'all')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        result = BiasService.get_feedback_list(
            page=page,
            page_size=page_size,
            feedback_type=feedback_type,
            start_date=start_date,
            end_date=end_date
        )
        return make_response(code=0, data=result)
    except Exception as e:
        logger.error(f"Error in get_feedback_list: {str(e)}", exc_info=True)
        return make_response(code=500, message=str(e))

@blds.route('/feedback/export', methods=['GET'])
@admin_required
@handle_exception
def export_feedback():
    try:
        result = BiasService.export_feedback()
        return make_response(code=0, data=result)
    except Exception as e:
        logger.error(f"Error in export_feedback: {str(e)}", exc_info=True)
        return make_response(code=500, message=str(e))

@blds.route('/history', methods=['GET'])
@login_required
@handle_exception
def get_history():
    page = int(request.args.get('page', 1))
    page_size = int(request.args.get('page_size', 10))
    offset = (page - 1) * page_size
    
    history = BiasService.get_detection_history(
        limit=page_size, 
        offset=offset,
        user_id=request.user['user_id']
    )
    return make_response(data=history)

@blds.route('/stats', methods=['GET'])
@admin_required
@handle_exception
def get_stats():
    stats = BiasService.get_detection_stats()
    return make_response(code=0, data=stats)

# 注册蓝图
app.register_blueprint(blds, url_prefix='/api')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)  # 使用 0.0.0.0 允许所有网络接口访问