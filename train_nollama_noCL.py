# -*- coding: utf-8 -*-

import pandas as pd
from transformers import BertModel, BertTokenizer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
import datetime  
import os

# Set random seed
seed_value = 29
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)

# Dataset and hyperparameters
N_AXIS = 4
MAX_SEQ_LEN = 512
BERT_NAME = "./bert-base-uncased/"
batch_size = 64
max_epochs = 10
learning_rate = 3e-5

axes = ["I-E", "N-S", "T-F", "J-P"]
classes = {"I": 0, "E": 1, "N": 0, "S": 1, "T": 0, "F": 1, "J": 0, "P": 1}

def calculate_metrics(preds, labels):
    preds = preds > 0.5
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')  
    return acc, f1
    
def text_preprocessing(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    if text.startswith("'"):
        text = text[1:-1]
    return text

class MBTIDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer, max_len):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, item):
        sentence = str(self.sentences[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }

class BERTClassifier(nn.Module):
    def __init__(self, bert_model, n_classes):
        super(BERTClassifier, self).__init__()
        self.bert = bert_model
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled_output = outputs.last_hidden_state[:, 0]
        return self.out(pooled_output)

data = pd.read_csv("./kaggle/mbti_1.csv")
data = data.sample(frac=1).reset_index(drop=True)
data['posts'] = data['posts'].apply(lambda x: text_preprocessing(str(x)))

labels = []
for personality in data["type"]:
    pers_vect = []
    for p in personality:
        pers_vect.append(classes[p])
    labels.append(pers_vect)

sentences = data["posts"].tolist()
labels = np.array(labels, dtype="float32")

train_sentences, val_test_sentences, y_train, val_test_labels = train_test_split(sentences, labels, test_size=0.33, random_state=seed_value)
val_sentences, test_sentences, y_val, y_test = train_test_split(val_test_sentences, val_test_labels, test_size=0.5, random_state=seed_value)

tokenizer = BertTokenizer.from_pretrained(BERT_NAME)

train_dataset = MBTIDataset(train_sentences, y_train, tokenizer, MAX_SEQ_LEN)
val_dataset = MBTIDataset(val_sentences, y_val, tokenizer, MAX_SEQ_LEN)
test_dataset = MBTIDataset(test_sentences, y_test, tokenizer, MAX_SEQ_LEN)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

bert_model = BertModel.from_pretrained(BERT_NAME)
model = BERTClassifier(bert_model, N_AXIS)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
criterion = nn.BCEWithLogitsLoss()

def evaluate(model, dataloader, num_labels):
    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            pred_labels = (outputs.sigmoid() > 0.5).int()
            predictions.append(pred_labels.cpu().numpy())
            true_labels.append(labels.int().cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    predictions = np.vstack(predictions)
    true_labels = np.vstack(true_labels)
    
    accuracies = []
    f1_scores = []
    
    for i in range(num_labels):
        accuracies.append(accuracy_score(true_labels[:, i], predictions[:, i]))
        f1_scores.append(f1_score(true_labels[:, i], predictions[:, i], average='binary'))
    
    return avg_loss, accuracies, f1_scores
    
best_avg_f1 = 0.0

model_save_dir = "checkpoint"
os.makedirs(model_save_dir, exist_ok=True)  

num_labels = N_AXIS  

for epoch in range(max_epochs):
    model.train()
    total_loss = 0
    for batch in train_dataloader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids, attention_mask, token_type_ids)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()
    
    avg_train_loss = total_loss / len(train_dataloader)
    val_loss, val_accuracies, val_f1_scores = evaluate(model, val_dataloader, num_labels)
    
    avg_f1_score = np.mean(val_f1_scores)    
    print(f"Epoch {epoch+1}/{max_epochs}, Training Loss: {avg_train_loss:.4f}")
    print(f"Validation Loss: {val_loss:.4f}")
    for i, axis in enumerate(axes):
        print(f"Accuracy for {axis}: {val_accuracies[i]:.4f}, F1 Score for {axis}: {val_f1_scores[i]:.4f}")
    
    if avg_f1_score > best_avg_f1:
        best_avg_f1 = avg_f1_score
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_filename = f"{current_time}_F1-{avg_f1_score:.4f}.pth"
        model_path = os.path.join(model_save_dir, model_filename)
        torch.save(model.state_dict(), model_path)
        print(f"Saved new best model to {model_path} with avg F1: {best_avg_f1:.4f}")
        