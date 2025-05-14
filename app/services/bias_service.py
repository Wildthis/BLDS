from infra.db.db_utils import DatabaseManager
import logging

logger = logging.getLogger(__name__)

class BiasService:
    @staticmethod
    def save_detection_record(text_content, bias_type, is_biased, confidence, user_id):
        """保存检测记录"""
        try:
            sql = """
                INSERT INTO bias_detection_records 
                (text_content, bias_type, is_biased, confidence, user_id)
                VALUES (%s, %s, %s, %s, %s)
            """
            params = (text_content, bias_type, is_biased, confidence, user_id)
            return DatabaseManager.execute_update(sql, params)
        except Exception as e:
            logger.error(f"保存检测记录失败: {str(e)}")
            raise

    @staticmethod
    def save_user_feedback(record_id, is_correct, feedback_content, user_id):
        """保存用户反馈"""
        try:
            sql = """
                INSERT INTO user_feedback 
                (record_id, is_correct, feedback_content, user_id)
                VALUES (%s, %s, %s, %s)
            """
            params = (record_id, is_correct, feedback_content, user_id)
            return DatabaseManager.execute_update(sql, params)
        except Exception as e:
            logger.error(f"保存用户反馈失败: {str(e)}")
            raise

    @staticmethod
    def get_detection_history(limit=10, offset=0, user_id=None):
        """获取检测历史"""
        try:
            sql = """
                SELECT r.*, f.is_correct, f.feedback_content, f.feedback_time
                FROM bias_detection_records r
                LEFT JOIN user_feedback f ON r.id = f.record_id
                WHERE r.user_id = %s
                ORDER BY r.created_at DESC
                LIMIT %s OFFSET %s
            """
            params = (user_id, limit, offset)
            records = DatabaseManager.execute_query(sql, params)
            
            # 获取总记录数
            count_sql = """
                SELECT COUNT(*) as total
                FROM bias_detection_records
                WHERE user_id = %s
            """
            count_result = DatabaseManager.execute_query(count_sql, (user_id,))
            total = count_result[0]['total'] if count_result else 0
            
            return {
                'records': records,
                'total': total,
                'page': offset // limit + 1,
                'page_size': limit
            }
        except Exception as e:
            logger.error(f"获取检测历史失败: {str(e)}")
            raise

    @staticmethod
    def get_detection_stats():
        """获取检测统计信息"""
        try:
            sql = """
                SELECT 
                    COUNT(*) as total_detections,
                    SUM(CASE WHEN is_biased = 1 THEN 1 ELSE 0 END) as biased_count,
                    COUNT(DISTINCT bias_type) as bias_types_count,
                    AVG(confidence) as avg_confidence
                FROM bias_detection_records
            """
            result = DatabaseManager.execute_query(sql)
            return result[0] if result else {
                'total_detections': 0,
                'biased_count': 0,
                'bias_types_count': 0,
                'avg_confidence': 0
            }
        except Exception as e:
            logger.error(f"获取检测统计信息失败: {str(e)}")
            raise

    @staticmethod
    def get_feedback_list(page=1, page_size=10, feedback_type='all', start_date=None, end_date=None, user_id=None):
        """获取反馈列表"""
        try:
            offset = (page - 1) * page_size
            where_clauses = []
            params = []
            
            if feedback_type != 'all':
                where_clauses.append("f.is_correct = %s")
                params.append(feedback_type == 'correct')
            
            if start_date:
                where_clauses.append("f.feedback_time >= %s")
                params.append(start_date)
            
            if end_date:
                where_clauses.append("f.feedback_time <= %s")
                params.append(end_date)
            
            if user_id:
                where_clauses.append("f.user_id = %s")
                params.append(user_id)
            
            where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"
            
            sql = f"""
                SELECT f.*, r.text_content, r.bias_type, r.confidence
                FROM user_feedback f
                JOIN bias_detection_records r ON f.record_id = r.id
                WHERE {where_sql}
                ORDER BY f.feedback_time DESC
                LIMIT %s OFFSET %s
            """
            params.extend([page_size, offset])
            
            records = DatabaseManager.execute_query(sql, params)
            
            # 获取总记录数
            count_sql = f"""
                SELECT COUNT(*) as total
                FROM user_feedback f
                WHERE {where_sql}
            """
            count_result = DatabaseManager.execute_query(count_sql, params[:-2])
            total = count_result[0]['total'] if count_result else 0
            
            return {
                'data': records,
                'total': total,
                'page': page,
                'page_size': page_size
            }
        except Exception as e:
            logger.error(f"获取反馈列表失败: {str(e)}")
            raise

    @staticmethod
    def export_feedback():
        """导出反馈数据"""
        try:
            sql = """
                SELECT 
                    f.id,
                    f.record_id,
                    f.is_correct,
                    f.feedback_content,
                    f.feedback_time,
                    r.text_content,
                    r.bias_type,
                    r.is_biased,
                    r.confidence,
                    u.username
                FROM user_feedback f
                JOIN bias_detection_records r ON f.record_id = r.id
                JOIN users u ON f.user_id = u.id
                ORDER BY f.feedback_time DESC
            """
            return DatabaseManager.execute_query(sql)
        except Exception as e:
            logger.error(f"导出反馈数据失败: {str(e)}")
            raise

    @staticmethod
    def get_bias_type_stats(start_date=None, end_date=None):
        """获取偏见类型统计"""
        try:
            where_clauses = []
            params = []
            
            if start_date:
                where_clauses.append("created_at >= %s")
                params.append(start_date)
            
            if end_date:
                where_clauses.append("created_at <= %s")
                params.append(end_date)
            
            where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"
            
            sql = f"""
                SELECT 
                    bias_type,
                    COUNT(*) as count
                FROM bias_detection_records
                WHERE {where_sql}
                GROUP BY bias_type
            """
            
            result = DatabaseManager.execute_query(sql, params)
            
            # 转换为前端需要的格式
            stats = {
                'racial': 0,
                'regional': 0,
                'gender': 0,
                'false': 0
            }
            
            for row in result:
                stats[row['bias_type']] = row['count']
            
            return stats
        except Exception as e:
            logger.error(f"获取偏见类型统计失败: {str(e)}")
            raise

    @staticmethod
    def get_bias_rate_trend(start_date=None, end_date=None):
        """获取偏见率趋势"""
        try:
            where_clauses = []
            params = []
            
            if start_date:
                where_clauses.append("created_at >= %s")
                params.append(start_date)
            
            if end_date:
                where_clauses.append("created_at <= %s")
                params.append(end_date)
            
            where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"
            
            sql = f"""
                SELECT 
                    DATE(created_at) as date,
                    COUNT(*) as total,
                    SUM(CASE WHEN is_biased = 1 THEN 1 ELSE 0 END) as biased_count
                FROM bias_detection_records
                WHERE {where_sql}
                GROUP BY DATE(created_at)
                ORDER BY date
            """
            
            result = DatabaseManager.execute_query(sql, params)
            
            # 转换为前端需要的格式
            trend = {
                'dates': [],
                'rates': []
            }
            
            for row in result:
                trend['dates'].append(row['date'].strftime('%Y-%m-%d'))
                rate = (row['biased_count'] / row['total']) * 100 if row['total'] > 0 else 0
                trend['rates'].append(round(rate, 2))
            
            return trend
        except Exception as e:
            logger.error(f"获取偏见率趋势失败: {str(e)}")
            raise