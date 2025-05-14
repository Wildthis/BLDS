import pymysql
from pymysql.cursors import DictCursor
from contextlib import contextmanager
from infra.db.config import DB_CONFIG
import logging

logger = logging.getLogger(__name__)

class DatabaseManager:
    @staticmethod
    @contextmanager
    def get_connection():
        """获取数据库连接的上下文管理器"""
        conn = None
        try:
            password = str(DB_CONFIG['password'])
            conn = pymysql.connect(
                host=DB_CONFIG['host'],
                port=int(DB_CONFIG['port']),
                user=DB_CONFIG['user'],
                password=password.encode('utf-8'),  # 将密码转换为字节类型
                database=DB_CONFIG['database'],
                charset=DB_CONFIG['charset'],
                cursorclass=DictCursor
            )
            yield conn
        except Exception as e:
            logger.error(f"Database connection error: {str(e)}", exc_info=True)
            raise
        finally:
            if conn:
                conn.close()

    @staticmethod
    def execute_query(sql, params=None):
        """执行查询操作"""
        try:
            with DatabaseManager.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(sql, params or ())
                    return cursor.fetchall()
        except Exception as e:
            logger.error(f"Query execution error: {str(e)}", exc_info=True)
            raise

    @staticmethod
    def execute_update(sql, params=None):
        """执行更新操作"""
        try:
            with DatabaseManager.get_connection() as conn:
                with conn.cursor() as cursor:
                    affected_rows = cursor.execute(sql, params or ())
                    conn.commit()
                    return affected_rows
        except Exception as e:
            logger.error(f"Update execution error: {str(e)}", exc_info=True)
            raise

    @staticmethod
    def execute_many(sql, params_list):
        """批量执行操作"""
        try:
            with DatabaseManager.get_connection() as conn:
                with conn.cursor() as cursor:
                    affected_rows = cursor.executemany(sql, params_list)
                    conn.commit()
                    return affected_rows
        except Exception as e:
            logger.error(f"Batch execution error: {str(e)}", exc_info=True)
            raise 