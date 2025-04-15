import torch
from torch.utils.data import Dataset

class BiasDataset(Dataset):
    def __init__(self, texts, topics, labels, tokenizer, max_len=128):
        self.texts = texts
        self.topics = topics
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.topic_list = ['race', 'region', 'gender']  # 假设有这些话题类别

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        original_topic = self.topics[idx]
        original_label = self.labels[idx]

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

        # 生成增强数据
        augmented_data = []
        for topic in self.topic_list:
            topic_embedding = torch.zeros(5)
            if topic == original_topic:
                label = original_label
            else:
                label = 0  # 设置为 False 或其他负标签
            topic_embedding[self.topic_list.index(topic)] = 1

            augmented_data.append({
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'topic_embedding': topic_embedding,
                'labels': torch.tensor(label, dtype=torch.long)
            })

        return augmented_data