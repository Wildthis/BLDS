import torch
from transformers import BertModel

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