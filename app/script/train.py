import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score
import progressbar

from app.script.classfier import BertBiasClassifier
from app.script.dataset import BiasDataset

# 加载数据
data = pd.read_csv('../../data/COLDataset/train.csv')  # 替换为你的数据文件路径
dev = pd.read_csv('../../data/COLDataset/dev.csv')  # 替换为你的数据文件路径
test = pd.read_csv('../../data/COLDataset/test.csv')  # 替换为你的数据文件路径

# 数据集划分
train_data = data[data['split'] == 'train']
dev_data = dev[dev['split'] == 'dev']
test_data = test[test['split'] == 'test']


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
            for sample in batch:
                input_ids = sample['input_ids'].to(device)
                attention_mask = sample['attention_mask'].to(device)
                topic_embedding = sample['topic_embedding'].to(device)
                labels = sample['labels'].to(device)

                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask, topic_embedding)
                loss = torch.nn.CrossEntropyLoss()(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            x += 1
            progress_bar.update(int(x / len(train_loader) * 100))
        progress_bar.finish()
        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}')

        # 验证模型
        model.eval()
        predictions, true_labels = [], []
        with torch.no_grad():
            for batch in dev_loader:
                for sample in batch:
                    input_ids = sample['input_ids'].to(device)
                    attention_mask = sample['attention_mask'].to(device)
                    topic_embedding = sample['topic_embedding'].to(device)
                    labels = sample['labels'].to(device)

                    outputs = model(input_ids, attention_mask, topic_embedding)
                    logits = outputs
                    predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
                    true_labels.extend(labels.cpu().numpy())

        print(f'Validation Accuracy: {accuracy_score(true_labels, predictions)}')
        print(f'Validation F1 Score: {f1_score(true_labels, predictions)}')

# 训练模型
train_model(model, train_loader, dev_loader, optimizer, epochs=1)
# 保存模型
torch.save(model.state_dict(), '../../resources/models/model.pt')