from flask import jsonify

def make_response(data=None, code=200, message="Success"):
    """
    创建标准响应格式
    :param data: 响应数据
    :param code: 状态码
    :param message: 响应消息
    :return: JSON响应
    """
    response = {
        "code": code,
        "message": message,
        "data": data
    }
    return jsonify(response)