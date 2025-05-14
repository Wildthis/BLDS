from infra.cfg import ConfigManager
import logging

logger = logging.getLogger(__name__)

# 数据库配置
DB_CONFIG = {
    'host': ConfigManager.get_config('db', {}).get('host', 'localhost'),
    'port': ConfigManager.get_config('db', {}).get('port', 3306),
    'user': ConfigManager.get_config('db', {}).get('user', 'root'),
    'password': ConfigManager.get_config('db', {}).get('password', '123456'),
    'database': ConfigManager.get_config('db', {}).get('database', 'blds'),
    'charset': 'utf8mb4'
}

# 打印配置信息（不包含密码）
logger.info(f"Database configuration: {DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}") 