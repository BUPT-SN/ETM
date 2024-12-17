# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from transformers import BertModel, BertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import random
import os
import datetime
import json
import re
import torch.nn.functional as F

# Set random seed
seed_value = 29
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)

# Dataset and hyperparameters
N_AXIS = 4
MAX_SEQ_LEN = 128
BERT_NAME = "./bert-base-uncased/"
batch_size = 4
max_epochs = 10
learning_rate = 3e-5
MAX_CHUNKS = 50  # Define maximum chunk number

# Emotion axes
axes = ["I-E", "N-S", "T-F", "J-P"]
classes = {"I": 0, "E": 1, "N": 0, "S": 1, "T": 0, "F": 1, "J": 0, "P": 1}

def label_to_mbti(label, dim_map=classes):
    dim_order = ["I-E", "N-S", "T-F", "J-P"]  # Ensure order matches MBTI labels
    mbti_str = ""
    for i, dim in enumerate(dim_order):
        if label[i].item() == dim_map[dim[0]]:
            mbti_str += dim[0]  # The first letter represents 0
        else:
            mbti_str += dim[2]  # The second letter represents 1
    return mbti_str

# Accuracy and F1 score calculation functions
def calculate_metrics(preds, labels):
    preds = preds > 0.5
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')  # Use macro average
    return acc, f1

# Text preprocessing
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

def custom_collate_fn(batch):
    input_ids, attention_mask, token_type_ids, labels, llama_embeddings = [], [], [], [], []

    max_num_chunks = min(max(len(item['input_ids']) for item in batch), MAX_CHUNKS)  # Find max chunk number and limit to MAX_CHUNKS
    max_chunk_length = MAX_SEQ_LEN  # Preset chunk length

    for item in batch:
        num_chunks = len(item['input_ids'])

        # Ensure all chunk sizes are consistent
        ids_cat = [chunk for chunk in item['input_ids']]
        mask_cat = [chunk for chunk in item['attention_mask']]
        type_ids_cat = [chunk for chunk in item['token_type_ids']]

        # If the number of chunks is less than the max number of chunks, pad
        if num_chunks < max_num_chunks:
            pad_length = max_num_chunks - num_chunks
            ids_cat.extend([torch.zeros(max_chunk_length, dtype=torch.long)] * pad_length)
            mask_cat.extend([torch.zeros(max_chunk_length, dtype=torch.long)] * pad_length)
            type_ids_cat.extend([torch.zeros(max_chunk_length, dtype=torch.long)] * pad_length)

        ids_cat = torch.stack(ids_cat, dim=0)
        mask_cat = torch.stack(mask_cat, dim=0)
        type_ids_cat = torch.stack(type_ids_cat, dim=0)

        input_ids.append(ids_cat)
        attention_mask.append(mask_cat)
        token_type_ids.append(type_ids_cat)
        labels.append(item['labels'])
        llama_embeddings.append(item['llama_embedding'])  # Collect corresponding llama embedding

    # Convert lists to tensors and retain batch dimension
    input_ids = torch.stack(input_ids, dim=0)
    attention_mask = torch.stack(attention_mask, dim=0)
    token_type_ids = torch.stack(token_type_ids, dim=0)
    labels = torch.stack(labels)
    llama_embeddings = torch.stack(llama_embeddings)  # Convert llama embedding to tensor

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'token_type_ids': token_type_ids,
        'labels': labels,
        'llama_embedding': llama_embeddings  # Return data containing llama embeddings
    }

# Text preprocessing
MBTIs = ('INTJ', 'INTP', 'INFP', 'ENTP', 'ISTP', 'ISFP', 'ESTJ', 'ISTJ', 'ESTP', 'ISFJ', 'ENFP', 'ESFP', 'ESFJ', 'ENFJ', 'INFJ', 'ENTJ')
token = '<mask>'

def find_all_MBTIs(post):
    mbti_idx_list = []
    for mbti in MBTIs:
        mbti_idx_list.extend([(match.start(), match.end()) for match in re.finditer(mbti.lower(), post)])
    mbti_idx_list.sort()
    return mbti_idx_list

def text_preprocessing(text):
    # Basic text preprocessing
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    if text.startswith("'"):
        text = text[1:-1]

    # Detect and replace all MBTI types
    mbti_idx_list = find_all_MBTIs(text)
    delete_idx = 0
    for start, end in mbti_idx_list:
        text = text[:start - delete_idx] + token + text[end - delete_idx:]
        delete_idx += end - start + len(token)

    return text

def split_into_chunks(text):
    return text.split("|||")

# Custom dataset
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

        posts = split_into_chunks(sentence)
        split_input_ids, split_attention_masks, split_token_type_ids = [], [], []

        for post in posts[:MAX_CHUNKS]:  # Limit max chunk number
            encoding = self.tokenizer.encode_plus(
                post,
                add_special_tokens=True,
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_token_type_ids=True,
                return_tensors='pt',
            )
            split_input_ids.append(encoding['input_ids'].squeeze(0))
            split_attention_masks.append(encoding['attention_mask'].squeeze(0))
            split_token_type_ids.append(encoding['token_type_ids'].squeeze(0))

        # If less than MAX_CHUNKS, pad
        num_current_posts = len(posts)
        if num_current_posts < MAX_CHUNKS:
            additional = MAX_CHUNKS - num_current_posts
            for _ in range(additional):
                split_input_ids.append(torch.zeros(self.max_len, dtype=torch.long))
                split_attention_masks.append(torch.zeros(self.max_len, dtype=torch.long))
                split_token_type_ids.append(torch.zeros(self.max_len, dtype=torch.long))

        return {
            'input_ids': torch.stack(split_input_ids),
            'attention_mask': torch.stack(split_attention_masks),
            'token_type_ids': torch.stack(split_token_type_ids),
            'labels': torch.tensor(label, dtype=torch.float)
        }

# Define model
class BERTClassifier(nn.Module):
    def __init__(self, bert_model, n_classes, llama_embed_dim):
        super(BERTClassifier, self).__init__()
        self.bert = bert_model
        self.llama_fc = nn.Linear(llama_embed_dim, 768)
        self.llama_k = nn.Linear(768, 768)
        self.llama_v = nn.Linear(768, 768)
        self.bert_q = nn.Linear(768, 768)
        self.attention = nn.MultiheadAttention(embed_dim=768, num_heads=8)
        self.relu = nn.ReLU()
        self.out = nn.Linear(768, n_classes)
        self.embedding_transform = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size),
            nn.Tanh()
        )
    
    def forward(self, input_ids, attention_mask, token_type_ids, llama_embeddings, return_embedding=False):
        if input_ids.dim() == 3 and attention_mask.dim() == 3 and token_type_ids.dim() == 3:
            batch_size, num_chunks, seq_length = input_ids.size()
            input_ids = input_ids.view(-1, seq_length)
            attention_mask = attention_mask.view(-1, seq_length)
            token_type_ids = token_type_ids.view(-1, seq_length)
        
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = pooled_output.view(batch_size, num_chunks, -1)
        pooled_output = pooled_output.mean(dim=1)
        
        if return_embedding:
            return self.embedding_transform(pooled_output)  # Return BERT embeddings only
        else:
            llama_transformed = self.relu(self.llama_fc(llama_embeddings))
            k = self.relu(self.llama_k(llama_transformed))
            v = self.relu(self.llama_v(llama_transformed))
            q = self.relu(self.bert_q(pooled_output))
            
            attn_output, attn_weights = self.attention(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0))
            attn_output = attn_output.squeeze(0)
            
            combined_output = pooled_output + attn_output
            return self.out(combined_output)  # Return classification result

# Load data
data = pd.read_csv(".kaggle/mbti_1.csv")
data = data.sample(frac=1).reset_index(drop=True)
data['posts'] = data.apply(lambda row: text_preprocessing(str(row['posts'])), axis=1)

labels = []
for personality in data["type"]:
    pers_vect = []
    for p in personality:
        pers_vect.append(classes[p])
    labels.append(pers_vect)

sentences = data["posts"].tolist()
labels = np.array(labels, dtype="float32")

# Load LLAMA embeddings
llama_embeddings = torch.load("./features_tensor/features_tensor_5.pt")

# Split dataset
train_sentences, val_test_sentences, y_train, val_test_labels, train_llama_embeds, val_test_llama_embeds = train_test_split(
    sentences, labels, llama_embeddings, test_size=0.4, random_state=seed_value)
val_sentences, test_sentences, y_val, y_test, val_llama_embeds, test_llama_embeds = train_test_split(
    val_test_sentences, val_test_labels, val_test_llama_embeds, test_size=0.5, random_state=seed_value)

# Initialize BERT tokenizer
tokenizer = BertTokenizer.from_pretrained(BERT_NAME)

# Create DataLoader
train_dataset = MBTIDataset(train_sentences, y_train, tokenizer, MAX_SEQ_LEN)
val_dataset = MBTIDataset(val_sentences, y_val, tokenizer, MAX_SEQ_LEN)
test_dataset = MBTIDataset(test_sentences, y_test, tokenizer, MAX_SEQ_LEN)

class LLAMADataset(Dataset):
    def __init__(self, original_dataset, llama_embeds):
        self.original_dataset = original_dataset
        self.llama_embeds = llama_embeds

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        original_item = self.original_dataset[idx]
        llama_embedding = self.llama_embeds[idx]
        original_item['llama_embedding'] = llama_embedding
        return original_item

train_dataset = LLAMADataset(train_dataset, train_llama_embeds)
val_dataset = LLAMADataset(val_dataset, val_llama_embeds)
test_dataset = LLAMADataset(test_dataset, test_llama_embeds)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

# Initialize BERT model
bert_model = BertModel.from_pretrained(BERT_NAME)
model = BERTClassifier(bert_model, N_AXIS, llama_embed_dim=4096)

# If GPU is available, use GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Define optimizer and loss function
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

# Define Focal Loss with different alpha for each axis
class FocalLoss(nn.Module):
    def __init__(self, alphas, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alphas = alphas
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce
    
    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
        
        pt = torch.exp(-BCE_loss)  
        F_loss = torch.zeros_like(BCE_loss)
    
        for i in range(len(self.alphas)):
            alpha_factor = self.alphas[i] * targets[:, i] + (1 - self.alphas[i]) * (1 - targets[:, i])
            F_loss[:, i] = alpha_factor * (1 - pt[:, i])**self.gamma * BCE_loss[:, i]
    
        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

# Set alpha values for each axis
alphas = [0.768, 0.863, 0.46, 0.39]  
criterion = FocalLoss(alphas=alphas, logits=True)

# Load MBTI label descriptions
with open('./nohua.json', 'r') as f:
    mbti_descriptions = json.load(f)

# Preprocess MBTI descriptions and create embeddings
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
        # Use sum pooling as embedding
        pooled_output = bert_output.last_hidden_state.sum(dim=1)
    mbti_embeddings[mbti_type] = pooled_output

# Define contrastive loss function
def contrastive_loss(text_embeddings, positive_embeddings, negative_embeddings, tau=0.07):
    # Normalize embedding vectors
    z_texts = F.normalize(text_embeddings, p=2, dim=1)
    z_pos = F.normalize(positive_embeddings, p=2, dim=1).squeeze(1)

    # Calculate positive sample similarity
    pos_sims = torch.cosine_similarity(z_texts, z_pos, dim=1)

    # Prepare negative sample embeddings
    neg_emb_tensor = torch.cat(negative_embeddings, dim=0).view(-1, 15, z_texts.size(1))

    # Normalize negative sample embeddings
    neg_emb_tensor = F.normalize(neg_emb_tensor, p=2, dim=2)

    # Calculate negative sample similarity
    neg_sims = torch.bmm(neg_emb_tensor, z_texts.unsqueeze(2)).squeeze(2)

    # Stabilize numerically: subtract maximum value from all similarities
    max_sim = torch.max(torch.cat((pos_sims.unsqueeze(1), neg_sims), dim=1), dim=1)[0]
    pos_sims_adj = pos_sims - max_sim
    neg_sims_adj = neg_sims - max_sim.unsqueeze(1)

    # Avoid negative infinity values
    pos_sims_adj = pos_sims_adj.clamp(min=-10)
    neg_sims_adj = neg_sims_adj.clamp(min=-10)

    # Calculate contrastive loss
    numerator = torch.exp(pos_sims_adj / tau)
    denominator = numerator + torch.sum(torch.exp(neg_sims_adj / tau), dim=1) + 1e-6
    losses = -torch.log(numerator / denominator)
    
    return losses  # Return loss per sample

# Add accuracy and F1 score calculation in model evaluation function
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
            llama_embeddings = batch['llama_embedding'].to(device)
            
            outputs = model(input_ids, attention_mask, token_type_ids, llama_embeddings)
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

# Set initial highest average F1 score
best_avg_f1 = 0.0

# Determine model save directory
model_save_dir = "checkpoint"
os.makedirs(model_save_dir, exist_ok=True)

# Determine hyperparameters ¦Ó and ¦Ë
tau = 0.07
lambda_contrastive = 1.0

# Modify training and evaluation loop
num_labels = N_AXIS  # Number of labels

for epoch in range(max_epochs):
    model.train()
    total_loss = 0
    total_contrastive_loss = 0
    total_classification_loss = 0

    for batch_index, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['labels'].to(device)
        llama_embeddings = batch['llama_embedding'].to(device)

        # Get embeddings for contrastive learning
        bert_embeddings = model(input_ids, attention_mask, token_type_ids, llama_embeddings, return_embedding=True)
        
        # Get classification output
        class_outputs = model(input_ids, attention_mask, token_type_ids, llama_embeddings, return_embedding=False)
        
        # Calculate regular classification loss
        classification_loss = criterion(class_outputs, labels)

        # Contrastive learning logic
        positive_embeddings = []  # To be defined
        all_negative_embeddings = []  # To be defined
        for label in labels:
            mbti_type = label_to_mbti(label)
            positive_embeddings.append(mbti_embeddings[mbti_type])
            negatives = [mbti_embeddings[key] for key, emb in mbti_embeddings.items() if key != mbti_type]
            all_negative_embeddings.append(torch.stack(negatives))
        
        positive_embeddings = torch.stack(positive_embeddings)
        contrastive_loss_val = contrastive_loss(bert_embeddings, positive_embeddings, all_negative_embeddings).mean()

        # Calculate total loss and optimize
        total_loss = classification_loss + contrastive_loss_val * lambda_contrastive
        total_loss.backward()
        optimizer.step()

        total_classification_loss += classification_loss.item()
        total_contrastive_loss += contrastive_loss_val.item()
        total_loss += total_loss.item()

    # Output loss for each epoch
    print(f"Epoch {epoch + 1}/{max_epochs}, Classification Loss: {total_classification_loss:.4f}, Contrastive Loss: {total_contrastive_loss:.4f}, Total Loss: {total_loss:.4f}")

    # Validation set evaluation
    val_loss, val_accuracies, val_f1_scores = evaluate(model, val_dataloader, num_labels)
    avg_f1_score = np.mean(val_f1_scores)    
    print(f"Validation Loss: {val_loss:.4f}")
    for i, axis in enumerate(axes):
        print(f"Accuracy for {axis}: {val_accuracies[i]:.4f}, F1 Score for {axis}: {val_f1_scores[i]:.4f}")
    
    # If the current average F1 score is the highest, save the model
    if avg_f1_score > best_avg_f1:
        best_avg_f1 = avg_f1_score
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_filename = f"{current_time}_F1-{avg_f1_score:.4f}_CL.pth"
        model_path = os.path.join(model_save_dir, model_filename)
        torch.save(model.state_dict(), model_path)
        print(f"Saved new best model to {model_path} with avg F1: {best_avg_f1:.4f}")

final_model_path = os.path.join(model_save_dir, "final_mbti_model.pth")
torch.save(model.state_dict(), final_model_path)
print("Saved final model to", final_model_path)