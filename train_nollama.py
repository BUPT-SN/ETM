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
import json
import torch.nn.functional as F

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
max_epochs = 20
learning_rate = 3e-5

axes = ["I-E", "N-S", "T-F", "J-P"]
classes = {"I": 0, "E": 1, "N": 0, "S": 1, "T": 0, "F": 1, "J": 0, "P": 1}

def label_to_mbti(label, dim_map=classes):
    dim_order = ["I-E", "N-S", "T-F", "J-P"]  
    mbti_str = ""
    for i, dim in enumerate(dim_order):
        if label[i].item() == dim_map[dim[0]]:
            mbti_str += dim[0]  
        else:
            mbti_str += dim[2]  
    return mbti_str

def calculate_metrics(preds, labels):
    preds = preds > 0.5
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')  
    return acc, f1
    
def text_preprocessing_pure(text):
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

MBTIs = ('INTJ', 'INTP', 'INFP', 'ENTP', 'ISTP', 'ISFP', 'ESTJ', 'ISTJ', 'ESTP', 'ISFJ', 'ENFP', 'ESFP', 'ESFJ', 'ENFJ', 'INFJ', 'ENTJ')
token = '<mask>' 

def find_all_MBTIs(post, mbti):
    return [(match.start(), match.end()) for match in re.finditer(mbti, post)]


def text_preprocessing(text, mbti_type):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    if text.startswith("'"):
        text = text[1:-1]

    mbti_idx_list = find_all_MBTIs(text, mbti_type.lower())
    delete_idx = 0
    for start, end in mbti_idx_list:
        text = text[:start - delete_idx] + token + text[end - delete_idx:]
        delete_idx += end - start + len(token)

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bert_model = BertModel.from_pretrained(BERT_NAME)
bert_model = bert_model.to(device) 

class EmbeddingTransform(nn.Module):
    def __init__(self, input_dim):
        super(EmbeddingTransform, self).__init__()
        self.linear = nn.Linear(input_dim, input_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        return self.tanh(self.linear(x))

transform = EmbeddingTransform(bert_model.config.hidden_size).to(device)

data = pd.read_csv("./kaggle/mbti_1.csv")
data = data.sample(frac=1).reset_index(drop=True)
data['posts'] = data.apply(lambda row: text_preprocessing(str(row['posts']), row['type']), axis=1)

labels = []
for personality in data["type"]:
    pers_vect = []
    for p in personality:
        pers_vect.append(classes[p])
    labels.append(pers_vect)

sentences = data["posts"].tolist()
labels = np.array(labels, dtype="float32")

train_sentences, val_test_sentences, y_train, val_test_labels = train_test_split(sentences, labels, test_size=0.4, random_state=seed_value)
val_sentences, test_sentences, y_val, y_test = train_test_split(val_test_sentences, val_test_labels, test_size=0.5, random_state=seed_value)

tokenizer = BertTokenizer.from_pretrained(BERT_NAME)

train_dataset = MBTIDataset(train_sentences, y_train, tokenizer, MAX_SEQ_LEN)
val_dataset = MBTIDataset(val_sentences, y_val, tokenizer, MAX_SEQ_LEN)
test_dataset = MBTIDataset(test_sentences, y_test, tokenizer, MAX_SEQ_LEN)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = BERTClassifier(bert_model, N_AXIS)

with open('./MBTI_label_embedding.json', 'r') as f:
    mbti_descriptions = json.load(f)

mbti_embeddings = {}
for mbti_type, description in mbti_descriptions.items():
    processed_description = text_preprocessing_pure(description)
    inputs = tokenizer.encode_plus(
        processed_description,
        add_special_tokens=True,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        max_length=MAX_SEQ_LEN,
        return_attention_mask=True,
        return_tensors='pt',
    )
    with torch.no_grad():
        bert_output = bert_model(
            input_ids=inputs['input_ids'].to(device),
            attention_mask=inputs['attention_mask'].to(device)
        )
        pooled_output = bert_output.last_hidden_state.sum(dim=1)
    mbti_embeddings[mbti_type] = pooled_output

model = model.to(device)
transform = transform.to(device)

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

tau = 0.07
lambda_contrastive = 0.3

def contrastive_loss(text_embeddings, positive_embeddings, negative_embeddings, tau=0.07):
    
    z_texts = F.normalize(transform(text_embeddings), p=2, dim=1)
    z_pos = F.normalize(transform(positive_embeddings), p=2, dim=1).squeeze(1)

    pos_sims = torch.cosine_similarity(z_texts, z_pos, dim=1)

    neg_emb_tensor = torch.cat(negative_embeddings, dim=0).view(-1, 15, z_texts.size(1))

    neg_emb_tensor = F.normalize(neg_emb_tensor, p=2, dim=2)

    neg_sims = torch.bmm(neg_emb_tensor, z_texts.unsqueeze(2)).squeeze(2)

    max_sim = torch.max(torch.cat((pos_sims.unsqueeze(1), neg_sims), dim=1), dim=1)[0]
    
    pos_sims_adj = pos_sims - max_sim
    neg_sims_adj = neg_sims - max_sim.unsqueeze(1)

    pos_sims_adj = pos_sims_adj.clamp(min=-10)
    neg_sims_adj = neg_sims_adj.clamp(min=-10)

    numerator = torch.exp(pos_sims_adj / tau)
    denominator = numerator + torch.sum(torch.exp(neg_sims_adj / tau), dim=1) + 1e-6
    losses = -torch.log(numerator / denominator)
    
    return losses  

num_labels = N_AXIS  

for epoch in range(max_epochs):
    model.train()
    total_loss = 0
    total_contrastive_loss = 0
    total_original_loss = 0

    for batch in train_dataloader:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        original_loss = criterion(outputs, labels)

        with torch.no_grad():
            bert_output = bert_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            text_embedding = bert_output.last_hidden_state[:, 0]  
        
        positive_embeddings = []
        
        all_negative_embeddings = []
        
        for label in labels:
            
            mbti_type = label_to_mbti(label)
            positive_embeddings.append(mbti_embeddings[mbti_type])
        
            negatives = [mbti_embeddings[key] for key, emb in mbti_embeddings.items() if key != mbti_type]
            all_negative_embeddings.append(torch.stack(negatives))  
        
        positive_embeddings = torch.stack(positive_embeddings)    

        contrastive_loss_val = contrastive_loss(text_embedding, positive_embeddings, all_negative_embeddings).mean()

        total_loss = original_loss + contrastive_loss_val * lambda_contrastive
        
        total_loss.backward()
        optimizer.step()

        total_original_loss += original_loss.item()
        total_contrastive_loss += torch.sum(contrastive_loss_val).item()

    print(f"Epoch {epoch+1}/{max_epochs}, Original Loss: {total_original_loss:.4f}, Contrastive Loss: {total_contrastive_loss:.4f}, Total Loss: {total_loss:.4f}")

    val_loss, val_accuracies, val_f1_scores = evaluate(model, val_dataloader, num_labels)
    avg_f1_score = np.mean(val_f1_scores)    
    print(f"Validation Loss: {val_loss:.4f}")
    for i, axis in enumerate(axes):
        print(f"Accuracy for {axis}: {val_accuracies[i]:.4f}, F1 Score for {axis}: {val_f1_scores[i]:.4f}")
    
    if avg_f1_score > best_avg_f1:
        best_avg_f1 = avg_f1_score
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_filename = f"{current_time}_F1-{avg_f1_score:.4f}_CL.pth"
        model_path = os.path.join(model_save_dir, model_filename)
        torch.save(model.state_dict(), model_path)
        print(f"Saved new best model to {model_path} with avg F1: {best_avg_f1:.4f}")
