from infra.db.db_utils import DatabaseManager
from datetime import datetime
import bcrypt
import jwt
import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)

class UserService:
    JWT_SECRET = 'your-secret-key'  # 在生产环境中应该使用环境变量
    JWT_ALGORITHM = 'HS256'

    @staticmethod
    def register(username: str, password: str) -> Dict:
        """注册新用户"""
        try:
            # 检查用户名是否已存在
            check_sql = "SELECT id FROM users WHERE username = %s"
            existing_user = DatabaseManager.execute_query(check_sql, (username,))
            if existing_user:
                raise ValueError("用户名已存在")

            # 加密密码
            hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

            # 创建新用户
            sql = """
            INSERT INTO users (username, password, role)
            VALUES (%s, %s, 'user')
            """
            user_id = DatabaseManager.execute_update(sql, (username, hashed_password.decode('utf-8')))

            return {
                'id': user_id,
                'username': username,
                'role': 'user'
            }
        except Exception as e:
            logger.error(f"注册用户失败: {str(e)}")
            raise

    @staticmethod
    def login(username: str, password: str) -> Dict:
        """用户登录"""
        try:
            # 获取用户信息
            sql = "SELECT id, username, password, role FROM users WHERE username = %s"
            result = DatabaseManager.execute_query(sql, (username,))
            
            if not result:
                raise ValueError("用户名或密码错误")

            user = result[0]
            
            # 验证密码
            if not bcrypt.checkpw(password.encode('utf-8'), user['password'].encode('utf-8')):
                raise ValueError("用户名或密码错误")

            # 更新最后登录时间
            update_sql = "UPDATE users SET last_login = %s WHERE id = %s"
            DatabaseManager.execute_update(update_sql, (datetime.now(), user['id']))

            # 生成 JWT token
            token = jwt.encode({
                'user_id': user['id'],
                'username': user['username'],
                'role': user['role'],
                'exp': datetime.utcnow().timestamp() + 24 * 60 * 60  # 24小时过期
            }, UserService.JWT_SECRET, algorithm=UserService.JWT_ALGORITHM)

            return {
                'token': token,
                'user': {
                    'id': user['id'],
                    'username': user['username'],
                    'role': user['role']
                }
            }
        except Exception as e:
            logger.error(f"用户登录失败: {str(e)}")
            raise

    @staticmethod
    def verify_token(token: str) -> Optional[Dict]:
        """验证 JWT token"""
        try:
            payload = jwt.decode(token, UserService.JWT_SECRET, algorithms=[UserService.JWT_ALGORITHM])
            return payload
        except jwt.ExpiredSignatureError:
            raise ValueError("Token已过期")
        except jwt.InvalidTokenError:
            raise ValueError("无效的Token")

    @staticmethod
    def get_user_by_id(user_id: int) -> Optional[Dict]:
        """根据ID获取用户信息"""
        try:
            sql = "SELECT id, username, role, created_at, last_login FROM users WHERE id = %s"
            result = DatabaseManager.execute_query(sql, (user_id,))
            return result[0] if result else None
        except Exception as e:
            logger.error(f"获取用户信息失败: {str(e)}")
            raise 