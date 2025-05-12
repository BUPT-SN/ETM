# -*- coding: utf-8 -*-
import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from tqdm.auto import tqdm
import re
from torch.nn import DataParallel

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = "./Meta-Llama-3-8B-Instruct/"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=AutoConfig.from_pretrained(model_path, output_hidden_states=True),
        torch_dtype=torch.float16
    ).to(device)

    model.eval()

    data_path = "./kaggle/mbti_1.csv"  
    data = pd.read_csv(data_path)
    data['posts'] = data.apply(lambda row: text_preprocessing(str(row['posts']), row['type']), axis=1)

    features_tensor = extract_features(data, tokenizer, model, device, max_length=8192)
    output_dir = 'features_tensor'
    os.makedirs(output_dir, exist_ok=True)
    torch.save(features_tensor, os.path.join(output_dir, 'features_tensor_5.pt'))
    print(f"Features saved to {os.path.join(output_dir, 'features_tensor_5.pt')}.")

def text_preprocessing(text, mbti_type):
    token = ' '
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    if text.startswith("'"):
        text = text[1:-1]

    mbti_idx_list = [(match.start(), match.end()) for match in re.finditer(re.escape(mbti_type.lower()), text)]
    for start, end in mbti_idx_list:
        text = text[:start] + token + text[end:]
    return text

def extract_features(data, tokenizer, model, device, max_length):
    all_features = []
    space_token_id = tokenizer(' ', add_special_tokens=False).input_ids[0]
    for post in tqdm(data['posts'], desc="Processing posts"):
        encoded = tokenizer(post, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
        encoded = {k: v.to(device) for k, v in encoded.items()}

        out_of_vocab_mask = encoded['input_ids'] >= tokenizer.vocab_size
        encoded['input_ids'][out_of_vocab_mask] = space_token_id

        attention_mask = encoded['attention_mask']
        with torch.no_grad():
            outputs = model(**encoded)
            last_five_layers = outputs.hidden_states[-5:]
            layer_mean = torch.stack(last_five_layers).mean(dim=0)
            masked_layer_mean = layer_mean * attention_mask.unsqueeze(-1).float()
            sum_embeddings = masked_layer_mean.sum(dim=1)
            sum_mask = attention_mask.sum(dim=1, keepdim=True)
            avg_embeddings = sum_embeddings / sum_mask.clamp(min=1)
            all_features.append(avg_embeddings.cpu())

    return torch.cat(all_features, dim=0) if all_features else torch.tensor([])

if __name__ == "__main__":
    main()
