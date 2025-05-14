import yaml
import os
import threading

class ConfigManager:
    _config = None
    _config_path = './resources/application.yml'  # 默认配置文件路径
    _lock = threading.Lock()
    _initialized = False

    @classmethod
    def _initialize(cls):
        """内部初始化方法"""
        with cls._lock:
            if not cls._initialized:
                if not os.path.exists(cls._config_path):
                    raise FileNotFoundError(f"Config file not found: {cls._config_path}")
                with open(cls._config_path, 'r') as file:
                    cls._config = yaml.safe_load(file)
                cls._initialized = True

    @classmethod
    def get_config(cls, key, default=None):
        """获取配置值"""
        cls._initialize()  # 确保已初始化
        return cls._config.get(key, default)

    @classmethod
    def set_config(cls, key, value):
        """设置配置值"""
        cls._initialize()  # 确保已初始化
        with cls._lock:
            cls._config[key] = value
            cls._save_config()

    @classmethod
    def _save_config(cls):
        """保存配置到文件"""
        with open(cls._config_path, 'w') as file:
            yaml.safe_dump(cls._config, file)