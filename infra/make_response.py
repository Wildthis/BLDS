from flask import jsonify

def make_response(data=None, status=True, message="Success"):
    """
    创建标准化的 API 响应
    
    Args:
        data: 响应数据
        status: 业务状态（True/False）
        message: 响应消息
    
    Returns:
        tuple: (json_response, http_status_code)
    """
    http_status = 200 if status else 400
    response = {
        "code": http_status,
        "status": status,
        "message": message,
        "data": data
    }
    return jsonify(response), http_status