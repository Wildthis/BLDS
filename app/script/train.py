import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import progressbar

# 加载数据
data = pd.read_csv('../../data/COLDataset/train.csv')  # 替换为你的数据文件路径
dev = pd.read_csv('../../data/COLDataset/dev.csv')  # 替换为你的数据文件路径
test = pd.read_csv('../../data/COLDataset/test.csv')  # 替换为你的数据文件路径

# 数据集划分
train_data = data[data['split'] == 'train']
dev_data = dev[dev['split'] == 'dev']
test_data = test[test['split'] == 'test']

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

# 初始化分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 创建数据集
train_dataset = BiasDataset(train_data['TEXT'], train_data['topic'], train_data['label'], tokenizer)
dev_dataset = BiasDataset(dev_data['TEXT'], dev_data['topic'], dev_data['label'], tokenizer)
test_dataset = BiasDataset(test_data['TEXT'], test_data['topic'], test_data['label'], tokenizer)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

class BertBiasClassifier(torch.nn.Module):
    def __init__(self, num_labels):
        super(BertBiasClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size + 5, num_labels)  # 加上话题嵌入的维度

    def forward(self, input_ids, attention_mask, topic_embedding):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = torch.cat((pooled_output, topic_embedding), dim=1)  # 将话题嵌入与BERT输出拼接
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

# 初始化模型和优化器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
model = BertBiasClassifier(num_labels=2).to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)

# 训练模型
def train_model(model, train_loader, dev_loader, optimizer, epochs=3):
    print('train_size', len(train_loader), 'dev_size', len(dev_loader))
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        progress_bar = progressbar.ProgressBar().start()
        x = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            topic_embedding = batch['topic_embedding'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, topic_embedding)
            loss = torch.nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            x = x + 1
            progress_bar.update(int(x / len(train_loader) * 100))
        progress_bar.finish()
        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}')

        # 验证模型
        model.eval()
        predictions, true_labels = [], []
        with torch.no_grad():
            for batch in dev_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                topic_embedding = batch['topic_embedding'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask, topic_embedding)
                logits = outputs
                predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        print(f'Validation Accuracy: {accuracy_score(true_labels, predictions)}')
        print(f'Validation F1 Score: {f1_score(true_labels, predictions)}')

# 训练模型
train_model(model, train_loader, dev_loader, optimizer, epochs=3)
# 保存模型
torch.save(model.state_dict(), 'model1.pt')