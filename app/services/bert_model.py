import torch
from transformers import BertTokenizer, BertForSequenceClassification

from app.script.classfier import BertBiasClassifier
from infra.cfg import ConfigManager


class BertModelSingleton:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(BertModelSingleton, cls).__new__(cls, *args, **kwargs)
            cls._instance._initialize_model()
        return cls._instance

    def _initialize_model(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # 初始化模型
        config = ConfigManager.get_config('torch').get('device')
        model = BertBiasClassifier(num_labels=2).to(config)
        # 加载保存的模型权重
        model.load_state_dict(torch.load('../resources/models/model.pt'))
        model.eval()  # 设置为评估模式
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

    def preprocess_text(self, text, topic, max_length=512):
        device = self.device
        # 编码文本
        encoded_pair = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,  # 添加特殊标记（如[CLS]和[SEP]）
            max_length=max_length,  # 设置最大长度
            padding='max_length',  # 填充到最大长度
            truncation=True,  # 截断过长的文本
            return_attention_mask=True,  # 返回注意力掩码
            return_tensors='pt'  # 返回PyTorch张量
        )
        input_ids = encoded_pair['input_ids'].to(device)
        attention_mask = encoded_pair['attention_mask'].to(device)

        # 假设topic_embedding是通过某种方式生成的，这里简化为随机张量
        topic_embedding = torch.randn(1, 768).to(device)  # 假设topic_embedding是768维向量
        return input_ids, attention_mask, topic_embedding

    def classify(self, text):
        topics = ['race', 'region', 'gender']
        device = self.device

        # 预处理文本
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        for topic in topics:
            topic_embedding = torch.zeros(5)  # 创建一维张量
            if topic == 'race':
                topic_embedding[0] = 1
            elif topic == 'region':
                topic_embedding[1] = 1
            elif topic == 'gender':
                topic_embedding[2] = 1
            else:
                topic_embedding[4] = 1

            # 将一维张量转换为二维张量
            topic_embedding = topic_embedding.unsqueeze(0).to(device)

            # 将数据输入模型
            with torch.no_grad():  # 关闭梯度计算
                logits = self.model(input_ids, attention_mask, topic_embedding)

            # 应用 softmax 函数，将 logits 转换为概率分布
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            predicted_label = torch.argmax(probabilities, dim=1)

            print(f"topic: {topic}, probabilities:{probabilities}, Predicted label: {predicted_label.item()}")