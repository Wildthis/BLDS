-- 创建数据库
CREATE DATABASE IF NOT EXISTS blds DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- 使用数据库
USE blds;

-- 创建偏见检测记录表
CREATE TABLE IF NOT EXISTS bias_detection_records (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    text_content TEXT NOT NULL COMMENT '检测的文本内容',
    bias_type VARCHAR(50) COMMENT '偏见类型：race/region/gender/false',
    is_biased BOOLEAN NOT NULL DEFAULT FALSE COMMENT '是否包含偏见',
    detection_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '检测时间',
    confidence FLOAT COMMENT '检测置信度',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    user_id INT,
    FOREIGN KEY (user_id) REFERENCES users(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='偏见检测记录表';

-- 创建用户反馈表
CREATE TABLE IF NOT EXISTS user_feedback (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    record_id BIGINT NOT NULL COMMENT '关联的检测记录ID',
    is_correct BOOLEAN NOT NULL COMMENT '用户认为检测结果是否正确',
    feedback_content TEXT COMMENT '用户反馈内容',
    feedback_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '反馈时间',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    user_id INT,
    FOREIGN KEY (record_id) REFERENCES bias_detection_records(id),
    FOREIGN KEY (user_id) REFERENCES users(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='用户反馈表';

-- 创建模型训练数据表
CREATE TABLE IF NOT EXISTS training_data (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    text_content TEXT NOT NULL COMMENT '训练文本内容',
    bias_type VARCHAR(50) NOT NULL COMMENT '偏见类型：race/region/gender',
    is_biased BOOLEAN NOT NULL COMMENT '是否包含偏见',
    source VARCHAR(100) COMMENT '数据来源',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='模型训练数据表';

-- 创建用户表
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) NOT NULL UNIQUE,
    password VARCHAR(255) NOT NULL,
    role ENUM('user', 'admin') NOT NULL DEFAULT 'user',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_login DATETIME
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 添加默认管理员账号
INSERT INTO users (username, password, role) VALUES 
('admin', '$2b$12$8K1p/a0dR1xqM1M7zQzK3e6QZQZQZQZQZQZQZQZQZQZQZQZQZQZQZ', 'admin');

-- 修改偏见检测记录表，添加用户ID字段
ALTER TABLE bias_detection_records ADD COLUMN user_id INT;
ALTER TABLE bias_detection_records ADD FOREIGN KEY (user_id) REFERENCES users(id);

-- 修改用户反馈表，添加用户ID字段
ALTER TABLE user_feedback ADD COLUMN user_id INT;
ALTER TABLE user_feedback ADD FOREIGN KEY (user_id) REFERENCES users(id); 