#!/usr/bin/env python
# 加载数据：将数据加载到 Pandas DataFrame 中，然后将其转换为 PyTorch Dataset 和 Dataloader

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW

class MyDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        text = self.df.iloc[index]['text']
        label = self.df.iloc[index]['label']
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_length,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        return {
            'input_ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'token_type_ids': torch.tensor(inputs['token_type_ids'], dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def main():
    df = pd.read_csv('data.csv')
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    dataset = MyDataset(df, tokenizer, max_length=128)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    # 加载预训练模型：加载 Hugging Face 的 BERT 中文预训练模型，然后根据需要进行微调。
    model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=2)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    train(model, dataloader, optimizer)


# 定义微调过程：在微调过程中，将输入传递给模型，计算损失函数，然后反向传播误差和更新模型参数。
def train(model, dataloader, optimizer):
    from tqdm import tqdm
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    for epoch in range(5):
        total_loss = 0
        model.train()
        for batch in tqdm(dataloader, total=len(dataloader)):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print('Epoch: {}, Loss: {}'.format(epoch+1, total_loss/len(dataloader)))


if __name__ == '__main__':
    main()
