import torch
from torch.utils.data import Dataset


# 定义数据集类
class BiasDataset(Dataset):
    def __init__(self, texts, topics, labels, tokenizer, max_len=128):
        self.texts = texts
        self.topics = topics
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        topic = self.topics[idx]
        label = self.labels[idx]

        # 文本分词
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        # 话题嵌入（简单独热编码）
        topic_embedding = torch.zeros(5)  # 假设有5个话题类别
        if topic == 'race':
            topic_embedding[0] = 1
        elif topic == 'region':
            topic_embedding[1] = 1
        # 添加其他话题类别
        else:
            topic_embedding[4] = 1

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'topic_embedding': topic_embedding,
            'labels': torch.tensor(label, dtype=torch.long)
        }