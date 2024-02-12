# ---- coding: utf-8 ----
# @author: Ziyang Zhang et al.

from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, Dataset
import torch
import json


class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


def encode_texts(batch_texts, device='cpu'):
    tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    model = AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    model.to(device)

    encoded_tensors = []
    for text in batch_texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        outputs = model(**inputs)
        encoded_tensors.append(outputs.last_hidden_state.mean(dim=1).squeeze().detach().cpu())

    return torch.stack(encoded_tensors)


def process_data(data_loader, device='cpu'):
    encoded_tensors = []
    for batch_texts in data_loader:
        batch_encoded = encode_texts(batch_texts, device=device)
        encoded_tensors.append(batch_encoded)
        torch.cuda.empty_cache()
    return torch.cat(encoded_tensors)

def encode_bert(dataset='mimic3', text_dim=128):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # load text json
    file_path = f'../data/raw_data/{dataset}/node_text.json'
    with open(file_path, 'r') as file:
        data = json.load(file)
    dataset = TextDataset(list(data.values()))
    data_loader = DataLoader(dataset, batch_size=1024, shuffle=False)
    encoded_matrix = process_data(data_loader, device=device)
    pca = PCA(n_components=text_dim)
    encoded_matrix_reduced = pca.fit_transform(encoded_matrix.detach().numpy())
    return torch.tensor(encoded_matrix_reduced)


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load text json
    file_path = '../data/raw_data/cradle/node_text.json'
    with open(file_path, 'r') as file:
        data = json.load(file)
    dataset = TextDataset(list(data.values()))
    data_loader = DataLoader(dataset, batch_size=1024, shuffle=False)
    encoded_matrix = process_data(data_loader, device=device)
    print(encoded_matrix.shape)
