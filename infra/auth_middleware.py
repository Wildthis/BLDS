from functools import wraps
from flask import request, jsonify
from app.services.user_service import UserService
import logging

logger = logging.getLogger(__name__)

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({
                'code': 401,
                'message': '未登录'
            }), 401

        try:
            # 移除 'Bearer ' 前缀
            if token.startswith('Bearer '):
                token = token[7:]
            
            # 验证token
            payload = UserService.verify_token(token)
            # 将用户信息添加到请求上下文
            request.user = payload
            return f(*args, **kwargs)
        except ValueError as e:
            return jsonify({
                'code': 401,
                'message': str(e)
            }), 401
        except Exception as e:
            logger.error(f"Token验证失败: {str(e)}")
            return jsonify({
                'code': 401,
                'message': 'Token验证失败'
            }), 401

    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({
                'code': 401,
                'message': '未登录'
            }), 401

        try:
            # 移除 'Bearer ' 前缀
            if token.startswith('Bearer '):
                token = token[7:]
            
            # 验证token
            payload = UserService.verify_token(token)
            
            # 检查是否是管理员
            if payload.get('role') != 'admin':
                return jsonify({
                    'code': 403,
                    'message': '需要管理员权限'
                }), 403
            
            # 将用户信息添加到请求上下文
            request.user = payload
            return f(*args, **kwargs)
        except ValueError as e:
            return jsonify({
                'code': 401,
                'message': str(e)
            }), 401
        except Exception as e:
            logger.error(f"Token验证失败: {str(e)}")
            return jsonify({
                'code': 401,
                'message': 'Token验证失败'
            }), 401

    return decorated_function 